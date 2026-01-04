# Copyright (c) OpenMMLab. All rights reserved.
import os
import numpy as np
import torch
import torch.nn.functional as F
import hashlib
from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.models.dense_heads.retina_head import RetinaHead
from mmdet.models.utils import filter_scores_and_topk, select_single_mlvl
from mmdet.ppal.models.utils import get_img_score_distance_matrix_slow, concat_all_gather, get_inter_feats

@MODELS.register_module()
class RetinaHeadFeat(RetinaHead):
    def __init__(self, total_images, max_det, feat_dim, output_path, **kwargs):
        super(RetinaHeadFeat, self).__init__(**kwargs)
        self.total_images = total_images
        self.queue_length = total_images
        self.current_images = 0
        self.max_det = max_det
        self.feat_dim = feat_dim
        self.output_path = output_path

        self.register_buffer("det_label_queue", torch.zeros((self.queue_length, max_det)))
        self.register_buffer("det_score_queue", torch.zeros((self.queue_length, max_det)))
        self.register_buffer("det_feat_queue", torch.zeros((self.queue_length, max_det, feat_dim)))
        self.register_buffer("image_id_queue", torch.zeros((self.queue_length, 1), dtype=torch.int) - 1)

    def predict_by_feat(self, cls_scores, bbox_preds, score_factors=None, batch_img_metas=None, cfg=None, rescale=False, with_nms=True, **kwargs):
        if cfg is None: cfg = self.test_cfg
        fpn_feats = kwargs.pop('fpn_feats', None)

        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device)

        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            fpn_feats_single = select_single_mlvl(fpn_feats, img_id) if fpn_feats is not None else None

            mlvl_bboxes, mlvl_scores, mlvl_labels, mlvl_lvl_ids = [], [], [], []
            for lvl, (cls_s, bbox_p, priors) in enumerate(zip(cls_score_list, bbox_pred_list, mlvl_priors)):
                cls_s = cls_s.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
                scores = cls_s.sigmoid() if self.use_sigmoid_cls else cls_s.softmax(-1)[:, :-1]
                bbox_p = bbox_p.permute(1, 2, 0).reshape(-1, 4)
                
                results = filter_scores_and_topk(scores, 0.05, cfg.get('nms_pre', 1000), dict(bbox_pred=bbox_p, priors=priors))
                s, l, keep, filtered = results
                bboxes = self.bbox_coder.decode(filtered['priors'], filtered['bbox_pred'], max_shape=img_meta['img_shape'])
                
                mlvl_bboxes.append(bboxes)
                mlvl_scores.append(s)
                mlvl_labels.append(l)
                mlvl_lvl_ids.append(torch.full_like(l, lvl))

            results = InstanceData()
            results.bboxes, results.scores, results.labels = torch.cat(mlvl_bboxes), torch.cat(mlvl_scores), torch.cat(mlvl_labels)
            results.level_ids = torch.cat(mlvl_lvl_ids)

            img_results = self._bbox_post_process(
                results=results, 
                cfg=cfg, 
                rescale=rescale, 
                with_nms=with_nms, 
                img_meta=img_meta, 
                fpn_feats=fpn_feats_single, 
                **kwargs)
            result_list.append(img_results)
            
        return result_list

    def _bbox_post_process(self, results, cfg, rescale=False, with_nms=True, img_meta=None, **kwargs):
        fpn_feats = kwargs.pop('fpn_feats', None)
        if cfg is None: cfg = self.test_cfg
        if img_meta is None: img_meta = {}

        results = super()._bbox_post_process(
            results=results, cfg=cfg, rescale=rescale, 
            with_nms=with_nms, img_meta=img_meta, **kwargs)

        if results.bboxes.numel() > 0 and fpn_feats is not None:
            print(f"\033[92m>>I AMMMM HEEEEERE INSIDE IF \033[0m")
            level_ids = getattr(results, 'level_ids', torch.zeros_like(results.labels))
            det_feats = get_inter_feats(fpn_feats, level_ids, results.bboxes, img_meta['img_shape'])
            
            # 1. Collect and Sync across all GPUs
            self.collect_det_info(img_meta, results.labels, results.scores, det_feats)
            
            # 2. DYNAMIC INCREMENT
            rank, world_size = get_dist_info()
            # Every time we process a batch, we've collectively processed 'world_size' images
            self.current_images += world_size 

            # 3. DYNAMIC SAVING TRIGGER
            # Instead of a hard-coded number, we check if we've filled the queue
            # We use a safety margin (- world_size + 1) to account for the final batch
            if self.current_images >= (self.queue_length - world_size + 1):
                if rank == 0:
                    print(f"\033[92m>> DYNAMIC TARGET REACHED: {self.current_images}/{self.queue_length}\033[0m")
                    print(f">> SAVING PPAL DIVERSITY DATA TO: {self.output_path}")
                    self.compute_al()
                    
                    # Reset counter for next potential round in the same script execution
                    self.current_images = 0 
        else:
            print(f"\033[92m>>I AMMMM HEEEEERE INSIDE ELSE \033[0m")
            print(f'results.bboxes.numel() = {results.bboxes.numel()} , fpn_feats: {fpn_feats}')
        results.cls_uncertainties = torch.zeros_like(results.scores)
        return results

    def collect_det_info(self, img_meta, det_labels, det_scores, det_feats):
        rank, world_size = get_dist_info()
        filename = img_meta.get('img_path', img_meta.get('filename', ''))
        img_id_val = int(hashlib.md5(os.path.basename(filename).encode()).hexdigest(), 16) % (10**8)
        img_id_tensor = torch.tensor([[img_id_val]], dtype=torch.int, device=self.image_id_queue.device)
        
        num_dets = min(det_labels.shape[0], self.max_det)
        pad_labels = F.pad(det_labels[:num_dets], (0, self.max_det - num_dets), value=-1)
        pad_scores = F.pad(det_scores[:num_dets], (0, self.max_det - num_dets), value=0)
        pad_feats = F.pad(det_feats[:num_dets], (0, 0, 0, self.max_det - num_dets), value=0)

        c_ids = concat_all_gather(img_id_tensor)
        c_labels = concat_all_gather(pad_labels.reshape(1, self.max_det))
        c_scores = concat_all_gather(pad_scores.reshape(1, self.max_det))
        c_feats = concat_all_gather(pad_feats.reshape(1, self.max_det, self.feat_dim))

        idx = self.current_images
        if idx + world_size <= self.queue_length:
            self.image_id_queue[idx:idx+world_size] = c_ids
            self.det_label_queue[idx:idx+world_size] = c_labels
            self.det_score_queue[idx:idx+world_size] = c_scores
            self.det_feat_queue[idx:idx+world_size] = c_feats

    def compute_al(self):
        # Only save if we actually have data
        valid = (self.image_id_queue >= 0).reshape(-1)
        if not valid.any():
            print(">> WARNING: No valid detections collected. Cannot save diversity matrix.")
            return

        print(f"\033[92m>> SAVING PPAL DIVERSITY DATA ({sum(valid)} images) TO: {self.output_path}\033[0m")
        
        img_dis_mat = get_img_score_distance_matrix_slow(
            self.det_label_queue[valid], 
            self.det_score_queue[valid], 
            self.det_feat_queue[valid], 
            score_thr=0.05)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'wb') as f:
            np.save(f, img_dis_mat.detach().cpu().numpy())
            np.save(f, self.image_id_queue[valid].detach().cpu().numpy())