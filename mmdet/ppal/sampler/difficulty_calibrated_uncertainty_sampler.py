import json
import numpy as np
import os
import torch
from collections import OrderedDict
from mmdet.ppal.builder import SAMPLER
from mmdet.ppal.sampler.al_sampler_base import BaseALSampler
from mmdet.ppal.utils.running_checks import sys_echo

eps = 1e-10

@SAMPLER.register_module()
class DCUSSampler(BaseALSampler):
    def __init__(
        self,
        n_sample_images,
        oracle_annotation_path,
        score_thr,
        class_weight_ub,
        class_weight_alpha,
        dataset_type,
    ):
        super(DCUSSampler, self).__init__(
            n_sample_images,
            oracle_annotation_path,
            is_random=False,
            dataset_type=dataset_type)

        self.score_thr = score_thr
        self.class_weight_ub = class_weight_ub
        self.class_weight_alpha = class_weight_alpha
        self.log_init_info()

    def _get_classwise_weight(self, results_json):
        ckpt_path = os.path.join(os.path.dirname(results_json), 'latest.pth')

        if not os.path.exists(ckpt_path):
            sys_echo(f'WARNING: Checkpoint not found at {ckpt_path}, using default class weights')
            _weights = np.array([1/len(self.CLASSES)] * len(self.CLASSES))
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            sd = ckpt.get('state_dict', {}) if isinstance(ckpt, dict) else {}
            # Support common prefixes (DDP/module and EMA wrappers) to avoid false warnings.
            key_candidates = [
                'bbox_head.class_quality',
                'module.bbox_head.class_quality',
                'ema/bbox_head.class_quality',
                'ema/module.bbox_head.class_quality',
            ]
            q = None
            for k in key_candidates:
                if k in sd:
                    q = sd[k]
                    break
            if q is not None:
                if isinstance(q, torch.Tensor):
                    class_qualities = q.detach().cpu().numpy()
                else:
                    class_qualities = np.asarray(q)
                # Simplified weight calculation based on class quality
                _weights = 1.0 - class_qualities
                _weights = _weights / (_weights.sum() + eps)
            else:
                sys_echo(f'WARNING: class_quality not found in checkpoint, using default class weights')
                _weights = np.array([1/len(self.CLASSES)] * len(self.CLASSES))

        class_weights = dict()
        for i in range(len(_weights)):
            cid = self.class_name2id[self.CLASSES[i]]
            class_weights[cid] = _weights[i]
        return class_weights

    def al_acquisition(self, result_json, last_label_path):
        # 1. Get weights
        class_weights = self._get_classwise_weight(result_json)

        # 2. Load inference results
        with open(result_json) as f:
            results = json.load(f)

        # 3. Load previously labeled IDs (INTEGER IDs, no hashing)
        with open(last_label_path) as f:
            last_labeled_data = json.load(f)
        last_labeled_ids = set(int(x['id']) for x in last_labeled_data['images'])

        # 4. Pass 1: Calculate Category average uncertainty
        category_uncertainty = OrderedDict()
        category_count = OrderedDict()

        for res in results:
            img_id = int(res['image_id'])
            
            # Skip if already labeled or not in oracle
            if img_id in last_labeled_ids or img_id not in self.oracle_data:
                continue
                
            img_size = (self.oracle_data[img_id]['image']['width'], self.oracle_data[img_id]['image']['height'])
            if not self.is_box_valid(res['bbox'], img_size) or res['score'] < self.score_thr:
                continue
                
            # Fallback to entropy if cls_uncertainty is missing
            if 'cls_uncertainty' in res:
                uncertainty = float(res['cls_uncertainty'])
            else:
                s = float(res['score'])
                uncertainty = -1 * (s * np.log(s + eps) + (1 - s) * np.log((1 - s) + eps))
            
            label = res['category_id']
            category_uncertainty[label] = category_uncertainty.get(label, 0.) + uncertainty
            category_count[label] = category_count.get(label, 0.) + 1

        # 5. Pass 2: Image-level calibrated uncertainty
        image_uncertainties = {img_id: [0.] for img_id in self.oracle_data.keys() if img_id not in last_labeled_ids}

        for res in results:
            img_id = int(res['image_id'])
            if img_id not in image_uncertainties:
                continue
                
            img_size = (self.oracle_data[img_id]['image']['width'], self.oracle_data[img_id]['image']['height'])
            if not self.is_box_valid(res['bbox'], img_size) or res['score'] < self.score_thr:
                continue
                
            if 'cls_uncertainty' in res:
                uncertainty = float(res['cls_uncertainty'])
            else:
                s = float(res['score'])
                uncertainty = -1 * (s * np.log(s + eps) + (1 - s) * np.log((1 - s) + eps))
            
            label = res['category_id']
            # Apply class weights to calibrate
            image_uncertainties[img_id].append(uncertainty * class_weights.get(label, 1.0))

        # 6. Sum and select top K
        final_scores = {img_id: np.sum(scores) for img_id, scores in image_uncertainties.items()}
        sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        sampled_img_ids = [item[0] for item in sorted_items[:self.n_images]]
        
        # 7. Remaining pool
        all_oracle_ids = set(self.oracle_data.keys())
        rest_img_ids = list(all_oracle_ids - last_labeled_ids - set(sampled_img_ids))

        print(f"\033[1;32mDEBUG Uncertainty: Pool Size {self.n_images}, Actually Selected {len(sampled_img_ids)}\033[0m")
        return sampled_img_ids, rest_img_ids

    def create_jsons(self, sampled_img_ids, unsampled_img_ids, last_labeled_json, out_label_path, out_unlabeled_path):
        # NOTE: For uncertainty, the "label" file is just a candidate pool for Diversity
        labeled_data = dict(images=[], annotations=[], categories=self.categories)
        
        for img_id in sampled_img_ids:
            if img_id in self.oracle_data:
                labeled_data['images'].append(self.oracle_data[img_id]['image'])
                # We don't add annotations here yet, diversity will do it for the final set

        with open(out_label_path, 'w') as f:
            json.dump(labeled_data, f)
        
        # Unlabeled set for the next round's base
        unlabeled_data = dict(images=[], categories=self.categories)
        for img_id in unsampled_img_ids:
            if img_id in self.oracle_data:
                unlabeled_data['images'].append(self.oracle_data[img_id]['image'])
        
        with open(out_unlabeled_path, 'w') as f:
            json.dump(unlabeled_data, f)

    def al_round(self, result_path, last_label_path, out_label_path, out_unlabeled_path):
        sys_echo('\n\n>> Starting DCUS Uncertainty Acquisition!!!')
        self.round += 1
        sampled_img_ids, rest_img_ids = self.al_acquisition(result_path, last_label_path)
        self.create_jsons(sampled_img_ids, rest_img_ids, last_label_path, out_label_path, out_unlabeled_path)
        sys_echo('>> DCUS Acquisition Complete!!!\n\n')
    
    def log_info(self, result_path,  out_label_path, out_unlabeled_path):
        sys_echo('>>>> Round: %d' % self.round)
        sys_echo('>>>> Dataset: %s' % self.dataset_type)
        sys_echo('>>>> Oracle annotation path: %s' % self.oracle_path)
        sys_echo('>>>> Image pool size: %d' % self.image_pool_size)
        sys_echo('>>>> Uncertainty pool size per Round: %d (%.2f%%)'%(self.n_images, 100.*float(self.n_images)/self.image_pool_size))
        sys_echo('>>>> Unlabeled results path: %s' % result_path)
        sys_echo('>>>> Uncertainty pool image info file path: %s' % out_label_path)
        sys_echo('>>>> Score threshold: %s' % self.score_thr)
        sys_echo('>>>> Class weight upper bound: %.2f' % self.class_weight_ub)
        sys_echo('>>>> Class quality alpha: %.2f' % self.class_weight_alpha)

    def log_init_info(self):
        sys_echo('>> %s initialized:'%self.__class__.__name__)
        sys_echo('>>>> Dataset: %s' % self.dataset_type)
        sys_echo('>>>> Oracle annotation path: %s' % self.oracle_path)
        sys_echo('>>>> Image pool size: %d' % self.image_pool_size)
        sys_echo('>>>> Uncertainty pool size per round: %d (%.2f%%)'%(self.n_images, 100.*float(self.n_images)/self.image_pool_size))
        sys_echo('>>>> Score threshold: %s' % self.score_thr)
        sys_echo('>>>> Class weight upper bound: %.2f' % self.class_weight_ub)
        sys_echo('>>>> Class quality alpha: %.2f' % self.class_weight_alpha)
        sys_echo('\n')