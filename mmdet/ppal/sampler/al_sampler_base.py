import json
import numpy as np
import os
from mmdet.ppal.utils.running_checks import sys_echo

eps = 1e-10

class BaseALSampler(object):
    def __init__(self, n_sample_images, oracle_annotation_path, is_random, dataset_type='buc', **kwargs):
        self.CLASSES = ['Hanagers', 'Other', 'Fenced', 'Not fenced']
        self.dataset_type = dataset_type
        self.n_images = n_sample_images
        self.is_random = is_random

        with open(oracle_annotation_path) as f:
            data = json.load(f)

        self.image_pool_size = len(data['images'])
        self.oracle_data = dict()
        self.categories = data['categories']
        self.categories_dict = {c['id']: c['name'] for c in self.categories}
        
        # FIX 1: Initialize class mapping dictionaries needed by child samplers
        self.class_id2name = {c['id']: c['name'] for c in self.categories if c['name'] in self.CLASSES}
        self.class_name2id = {c['name']: c['id'] for c in self.categories if c['name'] in self.CLASSES}

        # print(f'self.class_id2name = {self.class_id2name}')
        # print(f'self.class_name2id = {self.class_name2id}')
        self.valid_categories = list(self.class_id2name.keys())

        # Store Images using integer IDs
        for img in data['images']:
            img_id = int(img['id'])
            self.oracle_data[img_id] = {'image': img, 'annotations': []}

        # Store Annotations using integer IDs
        for ann in data['annotations']:
            img_id = int(ann['image_id'])
            if img_id in self.oracle_data and self.categories_dict.get(ann['category_id']) in self.CLASSES:
                self.oracle_data[img_id]['annotations'].append(ann)

        self.oracle_cate_prob = self.cate_prob_stat(input_json=None)
        self.round = 1
        self.size_thr = 16
        self.ratio_thr = 5.
        self.oracle_path = oracle_annotation_path
        self.requires_result = True
        self.latest_labeled = None

    def cate_prob_stat(self, input_json=None):
        cate_freqs = {cid: 0. for cid in self.valid_categories}
        if input_json is None:
            for img_id in self.oracle_data.keys():
                for ann in self.oracle_data[img_id]['annotations']:
                    if ann['category_id'] in cate_freqs:
                        cate_freqs[ann['category_id']] += 1.
        else:
            with open(input_json) as f:
                data = json.load(f)
            for ann in data['annotations']:
                if ann['category_id'] in self.valid_categories:
                    cate_freqs[ann['category_id']] += 1
        total = sum(cate_freqs.values()) + eps
        return {k: v / total for k, v in cate_freqs.items()}

    def set_round(self, new_round):
        self.round = new_round
    
    def is_box_valid(self, box, img_size):
        # clip box and filter out outliers
        img_w, img_h = img_size
        x1, y1, w, h = box
        if (x1 > img_w) or (y1 > img_h):
            return False
        x2 = min(img_w, x1+w)
        y2 = min(img_h, y1+h)
        w = x2 - x1
        h = y2 - y1
        return (np.sqrt(w*h) > self.size_thr) and (w/(h+eps) < self.ratio_thr) and (h/(w+eps) < self.ratio_thr)

    def al_acquisition(self, result_json, last_label_path):
        """Placeholder to match the signature required by child samplers."""
        pass

    def create_jsons(self, sampled_img_ids, unsampled_img_ids, last_labeled_json, out_label_path, out_unlabeled_path):
        with open(last_labeled_json) as f:
            last_labeled_data = json.load(f)

        last_labeled_ids = set(int(x['id']) for x in last_labeled_data['images'])
        new_sampled_ids = set(int(s) for s in sampled_img_ids)
        
        all_labeled_set = last_labeled_ids.union(new_sampled_ids)
        all_oracle_ids = set(self.oracle_data.keys())

        YELLOW = "\033[33m"
        RESET = "\033[0m"

        print(f"{YELLOW}{out_label_path}{RESET}")
        print(f"{YELLOW}all_labeled_set length: {len(all_labeled_set)}{RESET}")
        print(f"{YELLOW}all_oracle_ids length: {len(all_oracle_ids)}{RESET}")
        
        final_labeled_ids = list(all_labeled_set & all_oracle_ids)
        final_unlabeled_ids = list(all_oracle_ids - all_labeled_set)

        labeled_data = dict(images=[], annotations=[], categories=self.categories)
        unlabeled_data = dict(images=[], categories=self.categories)

        for img_id in final_labeled_ids:
            labeled_data['images'].append(self.oracle_data[img_id]['image'])
            labeled_data['annotations'].extend(self.oracle_data[img_id]['annotations'])
            
        for img_id in final_unlabeled_ids:
            unlabeled_data['images'].append(self.oracle_data[img_id]['image'])

        print(f"\033[1;36mDEBUG: Labeled count {len(labeled_data['images'])}, Unlabeled {len(unlabeled_data['images'])}\033[0m")

        with open(out_label_path, 'w') as f:
            json.dump(labeled_data, f)
        with open(out_unlabeled_path, 'w') as f:
            json.dump(unlabeled_data, f)

    def al_round(self, result_path, last_label_path, out_label_path, out_unlabeled_path):
        sys_echo('\n>> Starting AL acquisition...')
        self.round += 1
        # FIX 2: Child samplers like DCUS need the last_label_path to filter duplicates
        sampled_img_ids, rest_img_ids = self.al_acquisition(result_path, last_label_path)
        self.create_jsons(sampled_img_ids, rest_img_ids, last_label_path, out_label_path, out_unlabeled_path)
        sys_echo('>> AL acquisition complete!\n')

    def log_info(self, *args, **kwargs):
        pass

    def log_init_info(self):
        pass