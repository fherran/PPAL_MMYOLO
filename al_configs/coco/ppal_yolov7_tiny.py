# al_configs/coco/ppal_yolov7_coco_mod.py

# Paths
config_dir  = 'configs/coco_active_learning/'
work_dir    = 'work_dirs/...' # Change ... to the name of the work directory

# Environment setting
python_path = 'python'
port        = 29500
gpus        = 4

# Data setting
# Ensure these paths are relative to your project root
oracle_path         = '...' # Change ... to the path of the original json file
init_label_json     = '...' # Change ... to the path of the initial labeled json file
init_unlabeled_json = '...' # Change ... to the path of the initial unlabeled json file
init_model          = None


# You can use a pretrained checkpoint from the OpenMMLab model zoo, if you want to start from a strong checkpoint.
# You can put None if you want to start from scratch.
pretrained_ckpt = 'https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco/yolov7_tiny_syncbn_fast_8x16b-300e_coco_20221126_102719-0ee5bbdf.pth'

# Config setting - Ensure these reference the YOLOv7 config files
train_config             = config_dir + 'al_train/yolov7_26e.py'
uncertainty_infer_config = config_dir + 'al_inference/yolov7_uncertainty.py'
diversity_infer_config   = config_dir + 'al_inference/yolov7_diversity.py'

# Active learning setting
round_num             = 2 + 1 # + 1 for the initial round
budget                = 15000

budget_expand_ratio   = 1
_unc_pool_target = budget * budget_expand_ratio
uncertainty_pool_size = _unc_pool_target + (-_unc_pool_target) % gpus

# Sampler setting (MMEngine Registry compatible)
uncertainty_sampler_config = dict(
    type='DCUSSampler',  # Registered in mmdet.ppal.core or models
    n_sample_images=uncertainty_pool_size,
    oracle_annotation_path=oracle_path,
    score_thr=0.001,
    class_weight_ub=0.2,
    class_weight_alpha=0.3,
    dataset_type='YOLOv5CocoDataset')  # Updated to YOLOv5CocoDataset for YOLOv7

diversity_sampler_config = dict(
    type='DiversitySampler',  # Registered in mmdet.ppal.core or models
    n_sample_images=budget,
    oracle_annotation_path=oracle_path,
    dataset_type='YOLOv5CocoDataset')  # Updated to YOLOv5CocoDataset for YOLOv7

output_dir  = work_dir
