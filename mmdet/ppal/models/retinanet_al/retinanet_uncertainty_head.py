# Copyright (c) OpenMMLab. All rights reserved.
import torch
from typing import List, Optional

# MMEngine / MMCV imports
from mmengine.structures import InstanceData

# MMDet 3.x imports
from mmdet.registry import MODELS
from mmdet.models.dense_heads.retina_head import RetinaHead
from mmdet.models.utils import multi_apply
from mmdet.utils import OptConfigType, ConfigType

@MODELS.register_module()
class RetinaHeadUncertainty(RetinaHead):
    """RetinaHead with Uncertainty Calculation for Active Learning.
    
    This head implements the 3.x internal API to correctly handle 
    entropy-based uncertainty calculation during the inference phase.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # --- TRAINING LOGIC (Sync with 3.x) ---
    def loss_by_feat(self,
                     cls_scores: List[torch.Tensor],
                     bbox_preds: List[torch.Tensor],
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptConfigType = None,
                     **kwargs) -> dict:
        """Calculate loss using unpacked DetDataSample components."""
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         avg_factor) = cls_reg_targets

        losses_cls, losses_bbox = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor=avg_factor)

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    # --- INFERENCE LOGIC (Fix for TypeError) ---
    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigType,
                           rescale: bool = False,
                           img_meta: Optional[dict] = None,
                           **kwargs) -> InstanceData:
        """3.x compatible post-processing with uncertainty calculation.
        
        Args:
            results (InstanceData): Initial detection results before NMS.
            cfg (ConfigType): Test config containing NMS thresholds.
        """
        # 1. Call the base class NMS logic first to get the final detections.
        # This resolves the argument mismatch and leverages optimized kernels.
        results = super()._bbox_post_process(
            results=results, 
            cfg=cfg, 
            rescale=rescale, 
            img_meta=img_meta, 
            **kwargs)

        # 2. Calculate Entropy-based Uncertainty on the final filtered scores.
        if results.scores.numel() > 0:
            p = results.scores
            # Entropy Formula: -[p*log(p) + (1-p)*log(1-p)]
            # 1e-10 added for numerical stability to prevent log(0).
            uncertainty = -1 * (p * torch.log(p + 1e-10) + 
                               (1 - p) * torch.log((1 - p) + 1e-10))
            
            # Attach Custom Active Learning metadata to the results
            results.cls_uncertainties = uncertainty
            results.box_uncertainties = torch.zeros_like(uncertainty)
        else:
            # Handle empty detection case
            results.cls_uncertainties = results.scores.new_zeros(0)
            results.box_uncertainties = results.scores.new_zeros(0)

        return results