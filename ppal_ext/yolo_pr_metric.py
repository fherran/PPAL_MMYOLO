from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmyolo.registry import METRICS


def bbox_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one box (4,) and boxes (N,4), xyxy."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-9
    return inter / union


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "tensor"):
        x = x.tensor
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@METRICS.register_module()
class YoloPRMetric(BaseMetric):
    """YOLO-style Precision/Recall/F1 at a fixed IoU threshold (default 0.5).

    Computes per-class PR curve by sorting predictions by score and matching
    to GT (same class, same image) with greedy IoU matching.
    Reports mean(P), mean(R), mean(F1) at the per-class best F1 point.
    """

    default_prefix: Optional[str] = "yolo_pr"

    def __init__(self,
                 iou_thr: float = 0.5,
                 collect_device: str = "cpu",
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thr = float(iou_thr)

    def process(self, data_batch: dict, data_samples: List[dict]) -> None:
        for ds in data_samples:
            pred = getattr(ds, "pred_instances", None)
            gt = getattr(ds, "gt_instances", None)
            if pred is None or gt is None:
                continue

            pb = to_numpy(pred.bboxes)
            ps = to_numpy(pred.scores)
            pl = to_numpy(pred.labels).astype(np.int64)

            gb = to_numpy(gt.bboxes)
            gl = to_numpy(gt.labels).astype(np.int64)

            self.results.append(
                dict(
                    pred_bboxes=pb,
                    pred_scores=ps,
                    pred_labels=pl,
                    gt_bboxes=gb,
                    gt_labels=gl,
                )
            )

    def compute_for_class(self, results: List[dict], cls: int) -> Tuple[float, float, float]:
        gt_by_img: Dict[int, List[np.ndarray]] = {}
        pred_list: List[Tuple[float, int, np.ndarray]] = []

        for img_i, r in enumerate(results):
            gl = r["gt_labels"]
            gb = r["gt_bboxes"]
            if gb.size and gl.size:
                mask = gl == cls
                if np.any(mask):
                    gt_by_img[img_i] = [b for b in gb[mask]]

            pl = r["pred_labels"]
            pb = r["pred_bboxes"]
            ps = r["pred_scores"]
            if pb.size and pl.size:
                pmask = pl == cls
                if np.any(pmask):
                    for b, s in zip(pb[pmask], ps[pmask]):
                        pred_list.append((float(s), img_i, b))

        n_gt = sum(len(v) for v in gt_by_img.values())
        if n_gt == 0:
            return float("nan"), float("nan"), float("nan")

        pred_list.sort(key=lambda x: x[0], reverse=True)
        # GT exists but no predictions for this class.
        if len(pred_list) == 0:
            return 0.0, 0.0, 0.0

        matched: Dict[int, np.ndarray] = {
            img_i: np.zeros(len(gts), dtype=bool) for img_i, gts in gt_by_img.items()
        }

        tp = np.zeros(len(pred_list), dtype=np.float32)
        fp = np.zeros(len(pred_list), dtype=np.float32)

        for i, (_, img_i, box) in enumerate(pred_list):
            gts = gt_by_img.get(img_i, [])
            if not gts:
                fp[i] = 1.0
                continue
            gts_arr = np.stack(gts, axis=0).astype(np.float32)
            ious = bbox_iou_xyxy(box.astype(np.float32), gts_arr)
            j = int(np.argmax(ious))
            if ious[j] >= self.iou_thr and not matched[img_i][j]:
                matched[img_i][j] = True
                tp[i] = 1.0
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / (n_gt + 1e-9)
        precision = tp_cum / (tp_cum + fp_cum + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        if f1.size == 0:
            return 0.0, 0.0, 0.0
        best = int(np.nanargmax(f1))
        return float(precision[best]), float(recall[best]), float(f1[best])

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        logger = MMLogger.get_current_instance()
        if not results:
            logger.warning("YoloPRMetric got 0 results.")
            return dict(precision=0.0, recall=0.0, f1=0.0)

        all_labels = []
        for r in results:
            all_labels.append(r["gt_labels"])
        all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.int64)
        num_classes = int(all_labels.max() + 1) if all_labels.size else 0

        if num_classes <= 0:
            return dict(precision=0.0, recall=0.0, f1=0.0)

        ps, rs, fs = [], [], []
        for c in range(num_classes):
            p, r, f = self.compute_for_class(results, c)
            if np.isnan(p):
                continue
            ps.append(p)
            rs.append(r)
            fs.append(f)

        if not ps:
            return dict(precision=0.0, recall=0.0, f1=0.0)

        return dict(
            precision=float(np.mean(ps)),
            recall=float(np.mean(rs)),
            f1=float(np.mean(fs)),
        )


