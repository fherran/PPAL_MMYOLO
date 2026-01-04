# Copyright (c) OpenMMLab. All rights reserved.
import torch
from typing import List, Tuple

# MMEngine / MMCV imports
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from mmengine.dist import get_dist_info

# MMDet 3.x imports
from mmdet.registry import MODELS
from mmdet.models.dense_heads.retina_head import RetinaHead
from mmdet.structures.bbox import bbox_overlaps
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import DetDataSample

# Task modules (Replacements for mmdet.core)
from mmdet.models.utils import images_to_levels, multi_apply

# Custom PPAL utils
from mmdet.ppal.models.utils import concat_all_sum

@MODELS.register_module()
class RetinaQualityEMAHead(RetinaHead):
    """RetinaHead with Quality EMA for Active Learning.
    
    This head calculates an Exponential Moving Average of prediction quality 
    per class to help balance the active learning selection process.
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 stacked_convs: int = 4,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 base_momentum: float = 0.999,
                 quality_xi: float = 0.6,
                 **kwargs):
        
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            stacked_convs=stacked_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            **kwargs)
            
        self.quality_xi = quality_xi
        self.base_momentum = base_momentum

        # Buffers for Classwise Quality EMA
        self.register_buffer('class_momentum', torch.ones((num_classes,)) * base_momentum)
        self.register_buffer('class_quality', torch.zeros((num_classes,)))

    def loss_by_feat(self,
                     cls_scores: List[torch.Tensor],
                     bbox_preds: List[torch.Tensor],
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptConfigType = None,
                     **kwargs) -> dict:
        """Compute losses of the head using the MMDet 3.x internal API.
        
        The base class BaseDenseHead.loss() calls this method after unpacking 
        the DetDataSample objects.
        """
        device = cls_scores[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        
        # 1. Get Targets (Anchors, Labels, Bbox Targets)
        # 3.x uses the extracted gt_instances directly
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            return_sampling_results=False)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         avg_factor) = cls_reg_targets

        # 2. Handle multi-level anchor structures
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)

        # 3. Compute Loss Single via multi_apply
        # Note: the return includes classwise_quality for PPAL logic
        losses_cls, losses_bbox, classwise_quality = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=avg_factor)

        # 4. PPAL Quality EMA Update Logic
        with torch.no_grad():
            classwise_quality = torch.cat(classwise_quality, dim=0)
            if classwise_quality.numel() > 0:
                _classes = classwise_quality[:, 0]
                _qualities = classwise_quality[:, 1]

                collected_counts = classwise_quality.new_full((self.num_classes,), 0)
                collected_qualities = classwise_quality.new_full((self.num_classes,), 0)
                
                for i in range(self.num_classes):
                    mask = (_classes == i)
                    cq = _qualities[mask]
                    if cq.numel() > 0:
                        collected_counts[i] += cq.numel()
                        collected_qualities[i] += cq.sum()
                
                # Global sync across GPUs (Custom PPAL util)
                collected_counts = concat_all_sum(collected_counts)
                collected_qualities = concat_all_sum(collected_qualities)
                
                avg_qualities = collected_qualities / (collected_counts + 1e-5)
                
                # Update EMA buffer
                self.class_quality = self.class_momentum * self.class_quality + \
                                     (1. - self.class_momentum) * avg_qualities
                
                # Reset/Decay momentum
                self.class_momentum = torch.where(
                    avg_qualities > 0,
                    torch.zeros_like(self.class_momentum) + self.base_momentum,
                    self.class_momentum * self.base_momentum)

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level."""
        
        # Classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # Regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        
        anchors = anchors.reshape(-1, 4)
        
        # PPAL Quality Calculation
        with torch.no_grad():
            # Decode predictions and targets to calculate IoU Quality
            _bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            _bbox_gt = self.bbox_coder.decode(anchors, bbox_targets)

            valid_inds = (labels >= 0) & (labels < self.num_classes)
            if valid_inds.any():
                _labels = labels[valid_inds]
                iou = bbox_overlaps(_bbox_pred[valid_inds], _bbox_gt[valid_inds], is_aligned=True)
                
                # Probability p for the ground truth class
                p = torch.sigmoid(cls_score[valid_inds])[torch.arange(iou.shape[0], device=iou.device), _labels]
                
                # quality = p^xi * iou^(1-xi)
                quality = torch.pow(p, self.quality_xi) * torch.pow(iou, 1. - self.quality_xi)
                classwise_quality = torch.stack((_labels.float(), quality), dim=-1)
            else:
                classwise_quality = cls_score.new_zeros((0, 2))

        if self.reg_decoded_bbox:
            bbox_pred = _bbox_pred # Use decoded if loss requires it (e.g. GIoU)

        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
            
        return loss_cls, loss_bbox, classwise_quality