from __future__ import annotations

from typing import Any, Dict, Optional

from mmyolo.registry import HOOKS

try:
    from mmengine.hooks import Hook  # type: ignore
except Exception:  # pragma: no cover
    class Hook:  # type: ignore
        pass


def _to_float(x: Any) -> Optional[float]:
    try:
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return None


def _is_main_process() -> bool:
    try:
        import importlib

        dist = importlib.import_module("mmengine.dist")
        return bool(dist.is_main_process())
    except Exception:
        return True


@HOOKS.register_module()
class YoloStyleMetricsHook(Hook):
    priority = "LOW"

    def __init__(self, imgsz: Optional[int] = None):
        self.imgsz = imgsz
        self._sum: Dict[str, float] = {}
        self._count: int = 0

    def before_train_epoch(self, runner) -> None:
        self._sum = {}
        self._count = 0
        try:
            import importlib

            torch = importlib.import_module("torch")
            if torch.cuda.is_available():  # type: ignore[attr-defined]
                torch.cuda.reset_peak_memory_stats()  # type: ignore[attr-defined]
        except Exception:
            pass

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        if not isinstance(outputs, dict):
            return
        log_vars = outputs.get("log_vars")
        if not isinstance(log_vars, dict):
            return
        for k, v in log_vars.items():
            if not isinstance(k, str) or not k.startswith("loss"):
                continue
            fv = _to_float(v)
            if fv is None:
                continue
            self._sum[k] = self._sum.get(k, 0.0) + fv
        self._count += 1

    def _mean_loss(self, key: str) -> Optional[float]:
        if self._count > 0 and key in self._sum:
            return self._sum[key] / max(1, self._count)
        return None

    def _try_hub_scalar(self, runner, keys) -> Optional[float]:
        mh = getattr(runner, "message_hub", None)
        if mh is None:
            return None
        for k in keys:
            try:
                v = mh.get_scalar(k).current()
                fv = _to_float(v)
                if fv is not None:
                    return fv
            except Exception:
                continue
        return None

    def after_train_epoch(self, runner) -> None:
        if not _is_main_process():
            return

        epoch = int(getattr(runner, "epoch", 0)) + 1
        max_epochs = getattr(runner, "max_epochs", None)
        if max_epochs is None:
            max_epochs = getattr(getattr(runner, "train_loop", None), "max_epochs", None)

        box = self._mean_loss("loss_bbox")
        cls = self._mean_loss("loss_cls")
        dfl = self._mean_loss("loss_dfl")

        if box is None:
            box = self._try_hub_scalar(runner, ["train/loss_bbox", "train/loss_box", "train/loss_bbox_total"])
        if cls is None:
            cls = self._try_hub_scalar(runner, ["train/loss_cls", "train/loss_cls_total"])
        if dfl is None:
            dfl = self._try_hub_scalar(runner, ["train/loss_dfl"])

        try:
            import importlib

            torch = importlib.import_module("torch")
            mem_gb = ((torch.cuda.max_memory_reserved() / (1024**3))
                      if torch.cuda.is_available() else 0.0)  # type: ignore[attr-defined]
        except Exception:
            mem_gb = 0.0

        def _fmt(x: Optional[float]) -> str:
            return f"{x:.3f}" if isinstance(x, (int, float)) else "-"

        size_str = f"{self.imgsz}" if self.imgsz is not None else "-"
        mepochs = f"{max_epochs}" if max_epochs is not None else "?"
        runner.logger.info(
            f"Epoch {epoch}/{mepochs}  GPU_mem {mem_gb:.2f}G  "
            f"box_loss {_fmt(box)}  cls_loss {_fmt(cls)}  dfl_loss {_fmt(dfl)}  Size {size_str}"
        )

    def after_val_epoch(self, runner, metrics: Optional[dict] = None) -> None:
        if not _is_main_process():
            return
        metrics = metrics or {}

        map50 = metrics.get("coco/bbox_mAP_50", metrics.get("coco/bbox_mAP50"))
        map5095 = metrics.get("coco/bbox_mAP", metrics.get("coco/bbox_mAP_50_95"))
        prec = metrics.get("yolo_pr/precision")
        rec = metrics.get("yolo_pr/recall")

        map50_f = _to_float(map50)
        map5095_f = _to_float(map5095)
        prec_f = _to_float(prec)
        rec_f = _to_float(rec)

        def _fmt(x: Optional[float]) -> str:
            return f"{x:.3f}" if isinstance(x, (int, float)) else "-"

        runner.logger.info(
            f"Val  P {_fmt(prec_f)}  R {_fmt(rec_f)}  mAP50 {_fmt(map50_f)}  mAP50-95 {_fmt(map5095_f)}"
        )


