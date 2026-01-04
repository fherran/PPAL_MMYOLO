import json
import numpy as np
import os

from mmdet.ppal.builder import SAMPLER
from mmdet.ppal.sampler.al_sampler_base import BaseALSampler
from mmdet.ppal.utils.running_checks import sys_echo


eps = 1e-10


@SAMPLER.register_module()
class DiversitySampler(BaseALSampler):
    def __init__(
        self,
        n_sample_images,
        oracle_annotation_path,
        dataset_type,
    ):
        super(DiversitySampler, self).__init__(
            n_sample_images,
            oracle_annotation_path,
            is_random=False,
            dataset_type=dataset_type)

        self.log_init_info()

    @staticmethod
    def k_centroid_greedy(dis_matrix, K):
        N = dis_matrix.shape[0]
        centroids = []
        c = np.random.randint(0, N, (1,))[0]
        centroids.append(c)
        i = 1
        while i < K:
            centroids_diss = dis_matrix[:, centroids].copy()
            centroids_diss = centroids_diss.min(axis=1)
            centroids_diss[centroids] = -1
            new_c = np.argmax(centroids_diss)
            centroids.append(new_c)
            i += 1
        return centroids

    @staticmethod
    def kmeans(dis_matrix, K, n_iter=100):
        N = dis_matrix.shape[0]
        # Safety check: K cannot be larger than the number of images available
        K = min(K, N)
        if K <= 0: return []

        centroids = DiversitySampler.k_centroid_greedy(dis_matrix, K)
        data_indices = np.arange(N)

        for _ in range(n_iter):
            centroid_dis = dis_matrix[:, centroids]
            cluster_assign = np.argmin(centroid_dis, axis=1)
            
            new_centroids = []
            for i in range(K):
                cluster_i = data_indices[cluster_assign == i]
                
                if len(cluster_i) == 0:
                    remaining = list(set(data_indices) - set(new_centroids))
                    if remaining:
                        new_centroids.append(np.random.choice(remaining))
                    continue
                
                dis_mat_i = dis_matrix[cluster_i][:, cluster_i]
                sorted_cluster_indices = np.argsort(dis_mat_i.sum(axis=1))
                
                found_unique = False
                for idx in sorted_cluster_indices:
                    potential_centroid = cluster_i[idx]
                    if potential_centroid not in new_centroids:
                        new_centroids.append(potential_centroid)
                        found_unique = True
                        break
                
                if not found_unique:
                    remaining = list(set(data_indices) - set(new_centroids))
                    if remaining:
                        new_centroids.append(np.random.choice(remaining))
                    
            centroids = np.array(new_centroids)
        return centroids.tolist()


    def al_acquisition(self, image_dis_path, last_label_path):
        # 1. Load distance matrix and candidate IDs (Standard COCO Integers)
        with open(image_dis_path, 'rb') as frb:
            image_dis_matrix = np.load(frb)
            candidate_ids = np.load(frb).reshape(-1).astype(int)

        # 2. Load previously labeled images using their REAL IDs
        with open(last_label_path) as f:
            last_round_data = json.load(f)
        
        last_labeled_ids = set()
        for img in last_round_data['images']:
            # FIX: Use the actual 'id' field as an integer
            last_labeled_ids.add(int(img['id']))

        # 3. CRITICAL FILTER: Filter out images already in your labeled set
        # This ensures you don't pick duplicates.
        unlabeled_mask = [i for i, cid in enumerate(candidate_ids) if cid not in last_labeled_ids]
        
        if len(unlabeled_mask) == 0:
            print("\033[1;31mWARNING: No new images found in the candidate pool!\033[0m")
            return [], list(set(self.oracle_data.keys()) - last_labeled_ids)

        # Subset the distance matrix to only include truly unlabeled candidates
        sub_dis_matrix = image_dis_matrix[unlabeled_mask][:, unlabeled_mask]
        sub_candidate_ids = candidate_ids[unlabeled_mask]

        # 4. Perform Diversity Sampling on the filtered pool
        K = min(self.n_images, len(unlabeled_mask))
        centroids = DiversitySampler.kmeans(sub_dis_matrix, K=K)
        sampled_img_ids = sub_candidate_ids[centroids].tolist()

        # 5. Calculate remaining unlabeled pool
        all_oracle_ids = set(self.oracle_data.keys())
        new_total_labeled = last_labeled_ids.union(set(sampled_img_ids))
        rest_image_ids = list(all_oracle_ids - new_total_labeled)

        # Debug Prints
        print(f"\033[1;32mDEBUG: Oracle Total images: {self.image_pool_size}\033[0m")
        print(f"\033[1;32mDEBUG: Already Labeled: {len(last_labeled_ids)}\033[0m")
        print(f"\033[1;32mDEBUG: Successfully Sampled: {len(sampled_img_ids)}\033[0m")
        print(f"\033[1;32mDEBUG: New Unlabeled Pool Size: {len(rest_image_ids)}\033[0m")
        
        return sampled_img_ids, rest_image_ids

    def al_round(self, result_path, image_dis_path, last_label_path, out_label_path, out_unlabeled_path):
        sys_echo('\n\n>> Starting Diversity Sampling Acquisition!!!')
        self.round += 1
        self.log_info(result_path, image_dis_path, out_label_path, out_unlabeled_path)
        self.latest_labeled = last_label_path
        
        sampled_img_ids, rest_image_ids = self.al_acquisition(image_dis_path, last_label_path)
        
        # This calls create_jsons in al_sampler_base.py
        self.create_jsons(sampled_img_ids, rest_image_ids, last_label_path, out_label_path, out_unlabeled_path)
        sys_echo('>> Diversity Sampling Complete!!!\n\n')

    def log_info(self, result_path, image_dis_path, out_label_path, out_unlabeled_path):
        sys_echo('>>>> Round: %d' % self.round)
        sys_echo('>>>> Oracle Path: %s' % self.oracle_path)
        sys_echo('>>>> Sampled images per round: %d' % self.n_images)
        sys_echo('>>>> Image distance cache: %s' % image_dis_path)

    def log_init_info(self):
        sys_echo('>> %s initialized:'%self.__class__.__name__)
        sys_echo('>>>> Image pool size: %d' % self.image_pool_size)
        sys_echo('>>>> Target sample size: %d' % self.n_images)
        sys_echo('\n')