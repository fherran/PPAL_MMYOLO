# Copyright (c) OpenMMLab. All rights reserved.
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
from mmcv.ops import batched_nms
from mmdet.utils import ConfigType, OptInstanceList
from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData
from typing import List, Optional, Sequence, Tuple, Union
from torch import Tensor

from mmyolo.models.dense_heads.yolov7_head import YOLOv7Head
from mmyolo.registry import MODELS
from mmdet.ppal.models.utils import concat_all_gather, get_inter_feats


@MODELS.register_module()
class YOLOv7HeadFeat(YOLOv7Head):
    """YOLOv7Head with feature extraction for diversity sampling.
    
    This head extends YOLOv7Head to support diversity-based active learning
    by extracting features from detection boxes and computing image distance matrices.
    """
    
    def __init__(self, total_images, max_det, feat_dim, output_path, **kwargs):
        super().__init__(**kwargs)
        
        rank, world_size = get_dist_info()
        assert total_images % world_size == 0, f"total_images ({total_images}) must be divisible by world_size ({world_size})"
        
        self.rank = rank
        self.world_size = world_size
        self.total_images = total_images
        self.queue_length = total_images
        self.current_images = 0
        self.max_det = max_det
        self.feat_dim = feat_dim
        self.output_path = output_path
        
    
        if self.rank == 0:
            self.det_label_queue = torch.zeros((self.queue_length, max_det), dtype=torch.long, device='cpu')
            self.det_score_queue = torch.zeros((self.queue_length, max_det), dtype=torch.float32, device='cpu')
            self.det_feat_queue = torch.zeros((self.queue_length, max_det, feat_dim), dtype=torch.float16, device='cpu')
            self.image_id_queue = (torch.zeros((self.queue_length, 1), dtype=torch.long, device='cpu') - 1)
        else:
            self.det_label_queue = None
            self.det_score_queue = None
            self.det_feat_queue = None
            self.image_id_queue = None
        
        # Store neck features for feature extraction
        self.neck_feats = None
        
        # Projection layers to map neck features to common dimension
        # Will be initialized when neck features are first set
        self.feat_projections = None

    def set_neck_feats(self, neck_feats: Tuple[Tensor]):
        """Store neck features for later feature extraction.
        
        Args:
            neck_feats (Tuple[Tensor]): Features from the neck (before head processing).
        """
        self.neck_feats = neck_feats
        
        # Initialize projection layers if not already done
        if self.feat_projections is None:
            self.feat_projections = nn.ModuleList()
            for feat in neck_feats:
                in_channels = feat.shape[1]  # Get channel dimension
                proj = nn.Conv2d(in_channels, self.feat_dim, kernel_size=1)
                self.feat_projections.append(proj)
            self.feat_projections = self.feat_projections.to(neck_feats[0].device)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigType] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        """Transform a batch of output features extracted by the head into bbox results.
        
        Overrides the base method to extract features for diversity sampling.
        """
        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        import copy
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes

        mlvl_priors = self.mlvl_priors
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        mlvl_objectnesses = []

        for cls_score, bbox_pred, objectness, priors, stride in zip(
                cls_scores, bbox_preds, objectnesses, mlvl_priors,
                self.featmap_strides):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert objectness.size()[-2:] == bbox_pred.size()[-2:]

            # Permute from (batch, C, H, W) to (batch, H, W, C) and reshape
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.num_classes).sigmoid()
            objectness = objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1).sigmoid()

            # Process each image in the batch
            level_bboxes = []
            level_scores = []
            level_labels = []
            level_objectnesses = []
            
            for img_id in range(num_imgs):
                img_bbox_pred = bbox_pred[img_id]  # (num_priors, 4)
                img_cls_score = cls_score[img_id]  # (num_priors, num_classes)
                img_objectness = objectness[img_id]  # (num_priors,)

                if self.num_classes == 1:
                    img_scores = img_objectness.unsqueeze(1)
                else:
                    img_scores = (img_objectness.unsqueeze(1) * img_cls_score)

                nms_pre = cfg.get('nms_pre', -1)
                if nms_pre > 0 and img_scores.shape[0] > nms_pre:
                    max_scores, _ = img_scores.max(dim=1)
                    _, topk_inds = max_scores.topk(nms_pre)
                    img_priors = priors[topk_inds, :]
                    img_bbox_pred = img_bbox_pred[topk_inds, :]
                    img_scores = img_scores[topk_inds, :]
                    img_objectness = img_objectness[topk_inds]
                else:
                    img_priors = priors

                img_bboxes = self.bbox_coder.decode(
                    img_priors, img_bbox_pred, stride)
                level_bboxes.append(img_bboxes)
                level_scores.append(img_scores)
                level_labels.append(img_scores.argmax(dim=1, keepdim=False))
                level_objectnesses.append(img_objectness)
            
            mlvl_bboxes.append(level_bboxes)
            mlvl_scores.append(level_scores)
            mlvl_labels.append(level_labels)
            mlvl_objectnesses.append(level_objectnesses)

        results_list = []
        for img_id in range(num_imgs):
            img_meta = batch_img_metas[img_id]
            bboxes = torch.cat([mlvl_bboxes[i][img_id] for i in range(len(mlvl_bboxes))])
            scores = torch.cat([mlvl_scores[i][img_id] for i in range(len(mlvl_scores))])
            labels = torch.cat([mlvl_labels[i][img_id] for i in range(len(mlvl_labels))])
            objectnesses = torch.cat([mlvl_objectnesses[i][img_id] for i in range(len(mlvl_objectnesses))])

            if rescale:
                scale_factor = img_meta.get('scale_factor', [1, 1, 1, 1])
                if isinstance(scale_factor, (list, tuple)):
                    scale_factor = list(scale_factor)
                    # If scale_factor is 2 elements (w, h), repeat to 4 (w, h, w, h)
                    if len(scale_factor) == 2:
                        scale_factor = scale_factor * 2
                    scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = bboxes / scale_factor

            if with_nms:
                det_bboxes, keep_idxs = batched_nms(bboxes, scores.max(dim=1)[0], labels, cfg.nms)
                det_bboxes = det_bboxes[:cfg.max_per_img]
                det_labels = labels[keep_idxs][:cfg.max_per_img]
                det_scores = scores[keep_idxs][:cfg.max_per_img].max(dim=1)[0]
                
                # Extract features for diversity sampling
                if self.neck_feats is not None:
                    # Determine which level each detection came from
                    # This is approximate - we use the detection's scale to guess the level
                    img_shape = img_meta['img_shape']
                    det_boxes_unscaled = bboxes[keep_idxs][:cfg.max_per_img]
                    
                    # Calculate which feature level each box likely came from
                    box_areas = (det_boxes_unscaled[:, 2] - det_boxes_unscaled[:, 0]) * \
                                (det_boxes_unscaled[:, 3] - det_boxes_unscaled[:, 1])
                    box_areas = box_areas / (img_shape[0] * img_shape[1])
                    
                    # Assign to levels based on area (larger boxes -> lower levels)
                    lvl_inds = torch.zeros(len(det_boxes_unscaled), dtype=torch.long, device=det_boxes_unscaled.device)
                    for i, area in enumerate(box_areas):
                        if area > 0.1:
                            lvl_inds[i] = 0  # Large objects -> P3/8
                        elif area > 0.05:
                            lvl_inds[i] = 1  # Medium objects -> P4/16
                        else:
                            lvl_inds[i] = 2  # Small objects -> P5/32
                    
                    # Project neck features to common dimension if needed
                    if self.feat_projections is not None:
                        projected_feats = []
                        for i, feat in enumerate(self.neck_feats):
                            projected = self.feat_projections[i](feat)
                            projected_feats.append(projected)
                    else:
                        projected_feats = self.neck_feats
                    
                    # Extract features using get_inter_feats
                    det_feats = get_inter_feats(projected_feats, lvl_inds, det_boxes_unscaled, img_shape)
                    
                    # Pad or truncate to max_det
                    if len(det_feats) < self.max_det:
                        padding = torch.zeros(self.max_det - len(det_feats), self.feat_dim, 
                                            device=det_feats.device, dtype=det_feats.dtype)
                        det_feats = torch.cat([det_feats, padding], dim=0)
                        det_labels_padded = torch.cat([det_labels, torch.zeros(self.max_det - len(det_labels), 
                                                                             dtype=det_labels.dtype, 
                                                                             device=det_labels.device)], dim=0)
                        det_scores_padded = torch.cat([det_scores, torch.zeros(self.max_det - len(det_scores), 
                                                                              dtype=det_scores.dtype, 
                                                                              device=det_scores.device)], dim=0)
                    else:
                        det_feats = det_feats[:self.max_det]
                        det_labels_padded = torch.cat([det_labels[:self.max_det], 
                                                      torch.zeros(max(0, self.max_det - len(det_labels)), 
                                                                 dtype=det_labels.dtype, 
                                                                 device=det_labels.device)], dim=0)
                        det_scores_padded = torch.cat([det_scores[:self.max_det], 
                                                      torch.zeros(max(0, self.max_det - len(det_scores)), 
                                                                 dtype=det_scores.dtype, 
                                                                 device=det_scores.device)], dim=0)
                    
                    # Collect detection info for diversity computation
                    self.collect_det_info(img_meta, det_labels_padded, det_scores_padded, det_feats)
                else:
                    # If no neck features available, create dummy features
                    det_feats = torch.zeros(self.max_det, self.feat_dim, device=det_bboxes.device)
                    det_labels_padded = torch.cat([det_labels, torch.zeros(self.max_det - len(det_labels), 
                                                                           dtype=det_labels.dtype, 
                                                                           device=det_labels.device)], dim=0)
                    det_scores_padded = torch.cat([det_scores, torch.zeros(self.max_det - len(det_scores), 
                                                                          dtype=det_scores.dtype, 
                                                                          device=det_scores.device)], dim=0)
                    self.collect_det_info(img_meta, det_labels_padded, det_scores_padded, det_feats)
                
                rank, world_size = get_dist_info()
                self.current_images += world_size
                
                if self.current_images >= (self.total_images - world_size + 1):
                    if rank == 0:
                        print(f"\033[92m>> SAVING YOLOv7 DIVERSITY DATA TO: {self.output_path}\033[0m")
                        self.compute_al()
            else:
                det_bboxes = bboxes
                det_labels = labels
                det_scores = scores.max(dim=1)[0]

            results = InstanceData()
            results.bboxes = det_bboxes
            results.labels = det_labels
            results.scores = det_scores
            results_list.append(results)

        return results_list

    def collect_det_info(self, img_meta, det_labels, det_scores, det_feats):
        """Collect detection information for diversity computation.
        
        Args:
            img_meta (dict): Image metadata.
            det_labels (Tensor): Detection labels.
            det_scores (Tensor): Detection scores.
            det_feats (Tensor): Detection features.
        """
        rank, world_size = get_dist_info()
        
        if isinstance(img_meta, list):
            img_meta = img_meta[0]

        img_id_val = img_meta.get('img_id', None)
        if img_id_val is None:
            img_id_val = img_meta.get('image_id', None)
        if img_id_val is None:
            img_id_val = img_meta.get('id', None)
        if img_id_val is not None:
            img_id_val = int(img_id_val)
        else:
            # last resort: hash a stable basename (prefer img_path if filename isn't present)
            basename = os.path.basename(img_meta.get('filename') or img_meta.get('img_path') or '')
            img_id_val = int(hashlib.md5(basename.encode()).hexdigest(), 16) % (10**8)

        # Create tensors for this specific rank on the model device (for NCCL all_gather)
        img_id_tensor = torch.tensor([[img_id_val]], dtype=torch.long, device=det_feats.device)
        
        # Gather everything from all GPUs
        collected_ids = concat_all_gather(img_id_tensor)
        collected_labels = concat_all_gather(det_labels.reshape(1, self.max_det))
        collected_scores = concat_all_gather(det_scores.reshape(1, self.max_det).contiguous())
        collected_feats = concat_all_gather(det_feats.reshape(1, self.max_det, self.feat_dim).contiguous())

        # Write to queues at the correct global index
        start_idx = self.current_images
        end_idx = self.current_images + world_size
        
        if self.rank == 0 and end_idx <= self.queue_length:
            self.image_id_queue[start_idx:end_idx].copy_(collected_ids.detach().cpu())
            self.det_label_queue[start_idx:end_idx].copy_(collected_labels.detach().cpu().to(dtype=torch.long))
            self.det_score_queue[start_idx:end_idx].copy_(collected_scores.detach().cpu().to(dtype=torch.float32))
            self.det_feat_queue[start_idx:end_idx].copy_(collected_feats.detach().cpu().to(dtype=torch.float16))
        
        return

    def compute_al(self):
        """Compute active learning diversity matrix and save to file."""
        if self.rank != 0:
            return

        valid_inds = (self.image_id_queue >= 0).reshape(-1)
        image_id_queue = self.image_id_queue[valid_inds]

        det_label_queue = self.det_label_queue[valid_inds]
        det_score_queue = self.det_score_queue[valid_inds]
        det_feat_queue = self.det_feat_queue[valid_inds].to(dtype=torch.float32)

        num_imgs = det_feat_queue.size(0)
        num_classes = int(self.num_classes)
        feat_dim = int(self.feat_dim)
        score_thr = 0.001
        eps = 1e-12

        feats = det_feat_queue
        labels = det_label_queue.to(dtype=torch.long)
        scores = det_score_queue.to(dtype=torch.float32)
        valid = scores > score_thr

        emb = feats.new_zeros((num_imgs, num_classes * feat_dim), dtype=torch.float32, device='cpu')
        for c in range(num_classes):
            m = (labels == c) & valid
            w = scores * m.to(dtype=torch.float32)
            denom = w.sum(dim=1, keepdim=True).clamp_min(eps)
            pooled = (feats * w.unsqueeze(-1)).sum(dim=1) / denom
            emb[:, c * feat_dim:(c + 1) * feat_dim] = pooled

        emb = F.normalize(emb, p=2, dim=1)
        img_dis_mat = torch.empty((num_imgs, num_imgs), dtype=torch.float32, device='cpu')
        chunk = 512
        emb_t = emb.t().contiguous()
        for i in range(0, num_imgs, chunk):
            j = min(i + chunk, num_imgs)
            img_dis_mat[i:j] = 1.0 - emb[i:j].mm(emb_t)
        img_dis_mat.fill_diagonal_(0.0)

        img_dis_mat = img_dis_mat.numpy()
        img_ids = image_id_queue.detach().cpu().numpy()

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'wb') as fwb:
            np.save(fwb, img_dis_mat)
            np.save(fwb, img_ids)
        return