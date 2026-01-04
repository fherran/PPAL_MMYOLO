_base_ = "../bases/al_yolov7_base.py"

data_root = '...' # Change ... to the path of the training data
data_root_val = '...' # Change ... to the path of the validation data
num_classes = len(_base_.metainfo['classes'])
# ---------------------------------------------------------
# 1. Model & Head Setting
# ---------------------------------------------------------
model = dict(
    bbox_head=dict(
        type='YOLOv7QualityEMAHead',
        base_momentum=0.999,
        quality_xi=0.6,
        head_module=dict(num_classes=num_classes),
        test_cfg=dict(
            multi_label=True,
            nms_pre=30000,
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.65),
            max_per_img=300)
    )
)

# ---------------------------------------------------------
# 2. Optimizer Wrapper (YOLOv7 specific)
# ---------------------------------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=16),
    constructor='YOLOv7OptimWrapperConstructor',
    clip_grad=dict(max_norm=35, norm_type=2),
)

# ---------------------------------------------------------
# 3. Learning Rate Scheduler & Checkpoint Configuration
# ---------------------------------------------------------
max_epochs = 26 # Change ... to the number of epochs

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='cosine',
        lr_factor=0.01,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,  
        max_keep_ckpts=3,  
        save_best=None,
        save_last=True,  
        save_param_scheduler=False),
    logger=dict(type='LoggerHook', interval=5)
)

custom_hooks = [
    dict(type='YoloStyleMetricsHook', imgsz=864),
]

# ---------------------------------------------------------
# 4. Training Loops
# ---------------------------------------------------------
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=max_epochs, 
    val_interval=5)  # Validate every 10 epochs
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root=data_root,
        ann_file='...', # Change ... to the path of the initial labeled json file
        data_prefix=dict(img='...'), # Change ... to the path of the images
        metainfo=_base_.metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=32)
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root=data_root_val,
        ann_file='...', # Change ... to the path of the validation json file
        data_prefix=dict(img='...'), # Change ... to the path of the images
        metainfo=_base_.metainfo,
        test_mode=True,
        pipeline=_base_.test_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32)
    )
)

val_evaluator = [
    dict(
        type='mmdet.CocoMetric',
        ann_file=data_root_val + '...', # Change ... to the path of the validation json file
        metric='bbox',
        classwise=True,
    ),
    dict(type='YoloPRMetric', iou_thr=0.5),
]

# Added test_dataloader for Active Learning Inference rounds
# test_dataloader = ... # Change ... to the path of the test data


# test_evaluator = [
#     dict(
#         type='mmdet.CocoMetric',
#         ann_file=data_root_val + 'val.json',
#         metric='bbox',
#         classwise=True,
#     ),
#     dict(type='YoloPRMetric', iou_thr=0.5),
# ]


# Load trained model
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco/yolov7_tiny_syncbn_fast_8x16b-300e_coco_20221126_102719-0ee5bbdf.pth'