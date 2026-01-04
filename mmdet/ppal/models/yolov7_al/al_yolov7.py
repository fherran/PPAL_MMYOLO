# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from torch import Tensor

from mmyolo.models.detectors.yolo_detector import YOLODetector
from mmyolo.registry import MODELS


@MODELS.register_module()
class ALYOLODetector(YOLODetector):
    """YOLODetector with support for active learning feature extraction.
    
    This detector extends YOLODetector to pass neck features to the head
    for diversity-based active learning.
    """
    
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples,
                rescale: bool = True):
        """Predict results from a batch of inputs and data samples.
        
        Overrides the base method to store neck features for feature extraction.
        """
        x = self.extract_feat(batch_inputs)
        
        # Store neck features if head supports it
        if hasattr(self.bbox_head, 'set_neck_feats'):
            self.bbox_head.set_neck_feats(x)
        
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

