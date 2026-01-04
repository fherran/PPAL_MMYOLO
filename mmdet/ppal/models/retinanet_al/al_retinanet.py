import torch
import inspect
from typing import List
from mmdet.registry import MODELS
from mmdet.models.detectors.retinanet import RetinaNet
from mmdet.structures import DetDataSample

@MODELS.register_module()
class ALRetinaNet(RetinaNet):
    def extract_feat(self, batch_inputs: torch.Tensor):
        x = self.backbone(batch_inputs)
        if self.with_neck: x = self.neck(x)
        return x

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        x = self.extract_feat(batch_inputs)
        
        outs = self.bbox_head(x)
        cls_scores, bbox_preds = outs[:2]
        
        batch_img_metas = [s.metainfo for s in batch_data_samples]

        # Check if predict_by_feat accepts **kwargs (RetinaHeadFeat does, standard RetinaHead doesn't)
        sig = inspect.signature(self.bbox_head.predict_by_feat)
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        
        if has_kwargs:
            # RetinaHeadFeat - pass fpn_feats via kwargs
            results_list = self.bbox_head.predict_by_feat(
                cls_scores, 
                bbox_preds,
                batch_img_metas=batch_img_metas,
                rescale=rescale,
                cfg=self.test_cfg, 
                fpn_feats=x)
        else:
            # Standard head - don't pass fpn_feats
            results_list = self.bbox_head.predict_by_feat(
                cls_scores, 
                bbox_preds,
                batch_img_metas=batch_img_metas,
                rescale=rescale,
                cfg=self.test_cfg)

        return self.add_pred_to_datasample(batch_data_samples, results_list)

    def loss(self, batch_inputs, batch_data_samples):
        x = self.extract_feat(batch_inputs)
        return self.bbox_head.loss(x, batch_data_samples)