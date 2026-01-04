# configs/coco_active_learning/bases/al_yolov7_base.py
# MMDet 3.x / MMEngine compatible YOLOv7 config for PPAL (COCO format)

custom_imports = dict(
    imports=[
        'mmyolo.models',
        'mmyolo.engine.hooks',
        'ppal_ext.yolo_pr_metric',
        'mmdet.ppal.models.yolov7_al',
        'mmdet.ppal.models.yolov7_al.yolov7_quality_ema_head',
    ],
    allow_failed_imports=False,
)

_base_ = [
    '../../../mmyolo/configs/yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco.py'
]

# ----------------------
# Dataset / paths
# ----------------------
dataset_type = 'YOLOv5CocoDataset'
data_root = '...' # Change ... to the path of the training data
data_root_val = '...' # Change ... to the path of the validation data


metainfo = dict(classes=('...', '...', '...', '...')) # Change ... to the classes
num_classes = len(metainfo['classes'])

img_scale = (864, 864)
loss_cls_weight = 0.5
loss_obj_weight = 1.0

imgsz = img_scale[0]
obj_imgsz_scale = (imgsz / 640) ** 2


max_translate_ratio = 0.1
scaling_ratio_range = (0.5, 1.5)
mixup_prob = 0.05
randchoice_mosaic_prob = [1.0, 0.0]
mixup_alpha = 8.0
mixup_beta = 8.0

# Validation batch-shape policy (used by YOLOv5Collate / letterbox padding)
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=1,
    img_size=img_scale[0],
    size_divisor=32,
    extra_pad_ratio=0.5,
)

pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
]

mosiac4_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0, pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=max_translate_ratio,
        scaling_ratio_range=scaling_ratio_range,
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114),
    ),
]

mosiac9_pipeline = [
    dict(type='Mosaic9', img_scale=img_scale, pad_val=114.0, pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=max_translate_ratio,
        scaling_ratio_range=scaling_ratio_range,
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114),
    ),
]

randchoice_mosaic_pipeline = dict(
    type='RandomChoice',
    transforms=[mosiac4_pipeline, mosiac9_pipeline],
    prob=randchoice_mosaic_prob,
)

train_pipeline = [
    *pre_transform,
    randchoice_mosaic_pipeline,
    dict(
        type='YOLOv5MixUp',
        alpha=mixup_alpha,
        beta=mixup_beta,
        prob=mixup_prob,
        pre_transform=[*pre_transform, randchoice_mosaic_pipeline],
    ),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'flip_direction'),
    ),
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LetterResize', scale=img_scale, allow_scale_up=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
        )),
]

# ----------------------
# Dataloaders (MMEngine)
# ----------------------
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='...', # Change ... to the path of the initial labeled json file
        data_prefix=dict(img='...'), # Change ... to the path of the images
        metainfo=metainfo,
        pipeline=train_pipeline
    ),
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_val,
        ann_file='...', # Change ... to the path of the validation json file
        batch_shapes_cfg=batch_shapes_cfg,
        data_prefix=dict(img='...'), # Change ... to the path of the images
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

# test_dataloader = dict(
#     batch_size=2,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root_val,
#         ann_file='...', # Change ... to the path of the validation json file
#         batch_shapes_cfg=batch_shapes_cfg,
#         data_prefix=dict(img='...'), # Change ... to the path of the images
#         metainfo=metainfo,
#         test_mode=True,
#         pipeline=test_pipeline,
#     ),
# )

# ----------------------
# Evaluators (MMDet 3.x)
# ----------------------
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root_val + '...', # Change ... to the path of the validation json file
    metric='bbox',
    classwise=True,
)

# test_evaluator = dict(
#     type='mmdet.CocoMetric',
#     ann_file=data_root_val + '...', # Change ... to the path of the validation json file
#     metric='bbox',
#     classwise=True,
# )


model = dict(
    bbox_head=dict(
        type='YOLOv7QualityEMAHead',
        # RetinaQualityEMAHead-like EMA params
        base_momentum=0.999,
        quality_xi=0.6,
        head_module=dict(num_classes=len(metainfo['classes'])),
        loss_cls=dict(loss_weight=loss_cls_weight),
        loss_obj=dict(loss_weight=loss_obj_weight * obj_imgsz_scale),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=[
                [(10, 13), (16, 30), (33, 23)],       # P3/8
                [(30, 61), (62, 45), (59, 119)],      # P4/16
                [(116, 90), (156, 198), (373, 326)],  # P5/32
            ],
            strides=[8, 16, 32]),
        test_cfg=dict(
            multi_label=True,
            nms_pre=30000,
            score_thr=0.001,  # Critical for fixing the "All Zeros" image_dis.npy
            nms=dict(type='nms', iou_threshold=0.65),
            max_per_img=300)
    )
)


default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1,
        max_keep_ckpts=2,
        save_last=True,
        # Ensure 'latest.pth' is created so your script doesn't FileNotFoundError
        published_keys=['meta', 'state_dict']), 
    logger=dict(type='LoggerHook', interval=50),
)

log_level = 'INFO'
