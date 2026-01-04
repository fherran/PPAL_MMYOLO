_base_ = "../bases/al_yolov7_inference_base.py"

data_root = '...' # Change ... to the path of the training data
num_classes = len(_base_.metainfo['classes'])
model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        test_cfg=dict(
            multi_label=True,
            nms_pre=3000,
            min_bbox_size=0,
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=200)
    )
)

test_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ALCocoDataset',
        data_root=data_root,
        ann_file='...', # Change ... to the path of the initial unlabeled json file
        data_prefix=dict(img='...'), # Change ... to the path of the images
        metainfo=_base_.metainfo,
        test_mode=True,
        pipeline=_base_.test_pipeline
    )
)

test_evaluator = dict(
    type='mmdet.CocoMetric',    
    ann_file=data_root + '...', # Change ... to the path of the initial unlabeled json file
    metric='bbox',
    format_only=True,
    outfile_prefix='unlabeled_inference_result'
)

custom_hooks = []