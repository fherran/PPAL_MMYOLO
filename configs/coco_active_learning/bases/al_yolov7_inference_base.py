_base_ = "al_yolov7_base.py"

custom_imports = dict(
    imports=[
        'mmdet.ppal.datasets',
        'mmdet.ppal.models.yolov7_al',
    ],
    allow_failed_imports=False,
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,  # Allow mismatched parameters
        priority=49)
]

