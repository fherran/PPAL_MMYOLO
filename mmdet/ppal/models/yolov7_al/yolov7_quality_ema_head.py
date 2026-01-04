from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS
from mmyolo.models.dense_heads.yolov7_head import YOLOv7Head


def all_reduce_sum_(x: torch.Tensor) -> torch.Tensor:
    """In-place all-reduce sum over DDP ranks (no-op for single process)."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


@MODELS.register_module()
class YOLOv7QualityEMAHead(YOLOv7Head):
    """
    YOLOv7 head with class-wise quality EMA.
    """

    def __init__(self, *args, base_momentum: float = 0.999, quality_xi: float = 0.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_momentum = float(base_momentum)
        self.quality_xi = float(quality_xi)

        # Buffers (must be registered so DCUSSampler can load them from checkpoint)
        self.register_buffer('class_momentum',
                             torch.ones((self.num_classes,), dtype=torch.float32) * self.base_momentum)
        self.register_buffer('class_quality',
                             torch.zeros((self.num_classes,), dtype=torch.float32))

    def loss_by_feat(
        self,
        cls_scores: Sequence[Union[Tensor, Sequence[Tensor]]],
        bbox_preds: Sequence[Union[Tensor, Sequence[Tensor]]],
        objectnesses: Sequence[Union[Tensor, Sequence[Tensor]]],
        batch_gt_instances: Sequence[InstanceData],
        batch_img_metas: Sequence[dict],
        batch_gt_instances_ignore=None,
    ) -> dict:
        # Same structure as upstream YOLOv7Head.loss_by_feat, but we add EMA update.
        if isinstance(cls_scores[0], Sequence):
            with_aux = True
            batch_size = cls_scores[0][0].shape[0]
            device = cls_scores[0][0].device

            bbox_preds_main, bbox_preds_aux = zip(*bbox_preds)
            objectnesses_main, objectnesses_aux = zip(*objectnesses)
            cls_scores_main, cls_scores_aux = zip(*cls_scores)

            head_preds = self._merge_predict_results(bbox_preds_main, objectnesses_main, cls_scores_main)
            head_preds_aux = self._merge_predict_results(bbox_preds_aux, objectnesses_aux, cls_scores_aux)
        else:
            with_aux = False
            batch_size = cls_scores[0].shape[0]
            device = cls_scores[0].device
            head_preds = self._merge_predict_results(bbox_preds, objectnesses, cls_scores)
            head_preds_aux = None

        batch_targets_normed = self._convert_gt_to_norm_format(batch_gt_instances, batch_img_metas)
        scaled_factors = [
            torch.tensor(head_pred.shape, device=device)[[3, 2, 3, 2]] for head_pred in head_preds
        ]

        # Main loss + update quality EMA
        loss_cls, loss_obj, loss_box = self.calc_loss_with_quality(
            head_preds=head_preds,
            head_preds_aux=None,
            batch_targets_normed=batch_targets_normed,
            near_neighbor_thr=self.near_neighbor_thr,
            scaled_factors=scaled_factors,
            batch_img_metas=batch_img_metas,
            device=device,
            update_quality=True,
        )

        # Aux loss (NO quality update to avoid double-counting)
        if with_aux and head_preds_aux is not None:
            loss_cls_aux, loss_obj_aux, loss_box_aux = self.calc_loss_with_quality(
                head_preds=head_preds,
                head_preds_aux=head_preds_aux,
                batch_targets_normed=batch_targets_normed,
                near_neighbor_thr=self.near_neighbor_thr * 2,
                scaled_factors=scaled_factors,
                batch_img_metas=batch_img_metas,
                device=device,
                update_quality=False,
            )
            loss_cls += self.aux_loss_weights * loss_cls_aux
            loss_obj += self.aux_loss_weights * loss_obj_aux
            loss_box += self.aux_loss_weights * loss_box_aux

        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * batch_size * world_size,
            loss_obj=loss_obj * batch_size * world_size,
            loss_bbox=loss_box * batch_size * world_size,
        )

    def calc_loss_with_quality(
        self,
        head_preds,
        head_preds_aux,
        batch_targets_normed,
        near_neighbor_thr,
        scaled_factors,
        batch_img_metas,
        device,
        update_quality: bool = False,
    ):
        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)

        # Accumulators for class-wise EMA update
        if update_quality:
            collected_counts = torch.zeros((self.num_classes,), device=device, dtype=torch.float32)
            collected_qualities = torch.zeros((self.num_classes,), device=device, dtype=torch.float32)

        assigner_results = self.assigner(
            head_preds,
            batch_targets_normed,
            batch_img_metas[0]['batch_input_shape'],
            self.priors_base_sizes,
            self.grid_offset,
            near_neighbor_thr=near_neighbor_thr)
        mlvl_positive_infos = assigner_results['mlvl_positive_infos']
        mlvl_priors = assigner_results['mlvl_priors']
        mlvl_targets_normed = assigner_results['mlvl_targets_normed']

        if head_preds_aux is not None:
            head_preds = head_preds_aux

        for i, head_pred in enumerate(head_preds):
            batch_inds, proir_idx, grid_x, grid_y = mlvl_positive_infos[i].T
            num_pred_positive = batch_inds.shape[0]
            target_obj = torch.zeros_like(head_pred[..., 0])

            if num_pred_positive == 0:
                loss_box += head_pred[..., :4].sum() * 0
                loss_cls += head_pred[..., 5:].sum() * 0
                loss_obj += self.loss_obj(head_pred[..., 4], target_obj) * self.obj_level_weights[i]
                continue

            priors = mlvl_priors[i]
            targets_normed_lvl = mlvl_targets_normed[i]

            head_pred_positive = head_pred[batch_inds, proir_idx, grid_y, grid_x]

            # bbox loss + iou
            grid_xy = torch.stack([grid_x, grid_y], dim=1)
            decoded_pred_bbox = self._decode_bbox_to_xywh(head_pred_positive[:, :4], priors, grid_xy)
            target_bbox_scaled = targets_normed_lvl[:, 2:6] * scaled_factors[i]
            loss_box_i, iou = self.loss_bbox(decoded_pred_bbox, target_bbox_scaled)
            loss_box += loss_box_i

            # obj loss
            target_obj[batch_inds, proir_idx, grid_y, grid_x] = iou.detach().clamp(0).type(target_obj.dtype)
            loss_obj += self.loss_obj(head_pred[..., 4], target_obj) * self.obj_level_weights[i]

            # cls loss
            if self.num_classes > 1:
                pred_cls = targets_normed_lvl[:, 1].long()
                target_class = torch.full_like(head_pred_positive[:, 5:], 0., device=device)
                target_class[range(num_pred_positive), pred_cls] = 1.
                loss_cls += self.loss_cls(head_pred_positive[:, 5:], target_class)
            else:
                pred_cls = torch.zeros((num_pred_positive,), device=device, dtype=torch.long)
                loss_cls += head_pred_positive[:, 5:].sum() * 0

            if update_quality:
                with torch.no_grad():
                    # p = sigmoid(obj) * sigmoid(cls[target])
                    obj_p = torch.sigmoid(head_pred_positive[:, 4])
                    cls_p_all = torch.sigmoid(head_pred_positive[:, 5:])
                    cls_p = cls_p_all[torch.arange(num_pred_positive, device=device), pred_cls]
                    p = (obj_p * cls_p).clamp(0.0, 1.0)
                    q = torch.pow(p, self.quality_xi) * torch.pow(iou.clamp(min=0.0, max=1.0),
                                                                 1.0 - self.quality_xi)
                    # accumulate per class
                    for c in range(self.num_classes):
                        m = (pred_cls == c)
                        if m.any():
                            collected_counts[c] += float(m.sum())
                            collected_qualities[c] += q[m].sum().float()

        if update_quality:
            with torch.no_grad():
                all_reduce_sum_(collected_counts)
                all_reduce_sum_(collected_qualities)
                avg_qualities = collected_qualities / (collected_counts + 1e-5)

                if self.class_quality.device != avg_qualities.device:
                    self.class_quality = self.class_quality.to(avg_qualities.device)
                if self.class_momentum.device != avg_qualities.device:
                    self.class_momentum = self.class_momentum.to(avg_qualities.device)

                self.class_quality = self.class_momentum * self.class_quality + \
                                     (1. - self.class_momentum) * avg_qualities
                self.class_momentum = torch.where(
                    avg_qualities > 0,
                    torch.zeros_like(self.class_momentum) + self.base_momentum,
                    self.class_momentum * self.base_momentum,
                )

        return loss_cls, loss_obj, loss_box


