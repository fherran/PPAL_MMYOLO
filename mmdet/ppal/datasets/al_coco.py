# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from typing import List, Optional

# MMDet 3.x / MMEngine Imports
from mmdet.registry import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmdet.structures import DetDataSample

@DATASETS.register_module()
class ALCocoDataset(CocoDataset):
    """CocoDataset for Active Learning (PPAL) in MMDet 3.x.
    
    This dataset integrates with the PPAL framework by extracting custom 
    uncertainty metrics from prediction instances for active selection rounds.
    """

    def __init__(self, **kwargs):
        """Initialize the Active Learning COCO dataset."""
        super().__init__(**kwargs)

    def _det2json(self, results: List[DetDataSample]) -> List[dict]:
        """Convert detection results to COCO JSON style for PPAL.
        
        Replaces the 2.x nested list iteration with DetDataSample 
        and pred_instances attribute access.
        """
        json_results = []
        for idx in range(len(self)):
            # In 3.x, metadata is retrieved via get_data_info
            img_info = self.get_data_info(idx)
            img_id = img_info.get('img_id', idx)
            
            # results[idx] is now a DetDataSample object
            data_sample = results[idx]
            pred_instances = data_sample.pred_instances
            
            # Convert tensors to numpy for JSON serialization
            bboxes = pred_instances.bboxes.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            labels = pred_instances.labels.cpu().numpy()
            
            # Extract PPAL-specific uncertainties added during head refactoring
            if 'cls_uncertainties' in pred_instances:
                cls_uncertainties = pred_instances.cls_uncertainties.cpu().numpy()
            else:
                cls_uncertainties = np.zeros_like(scores)

            if 'box_uncertainties' in pred_instances:
                box_uncertainties = pred_instances.box_uncertainties.cpu().numpy()
            else:
                box_uncertainties = np.zeros_like(scores)

            for i in range(len(labels)):
                data = dict()
                data['image_id'] = img_id
                
                # Convert [x1, y1, x2, y2] to COCO [x, y, w, h] format
                x1, y1, x2, y2 = bboxes[i]
                data['bbox'] = [
                    float(x1), 
                    float(y1), 
                    float(x2 - x1), 
                    float(y2 - y1)
                ]
                
                data['score'] = float(scores[i])
                data['cls_uncertainty'] = float(cls_uncertainties[i])
                data['box_uncertainty'] = float(box_uncertainties[i])
                data['category_id'] = self.cat_ids[labels[i]]
                json_results.append(data)
                
        return json_results

    def format_results(self, 
                       results: List[DetDataSample], 
                       jsonfile_prefix: Optional[str] = None, 
                       **kwargs) -> tuple:
        """Format results to JSON for the active learning sampler."""
        import mmengine
        import os.path as osp
        import tempfile
        
        assert isinstance(results, list), 'results must be a list'
        
        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
            
        result_files = dict()
        json_results = self._det2json(results)
        
        result_files['bbox'] = f'{jsonfile_prefix}.bbox.json'
        mmengine.dump(json_results, result_files['bbox'])
        
        return result_files, tmp_dir