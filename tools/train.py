# Copyright (c) OpenMMLab.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch

from mmengine.config import Config, DictAction
from mmengine.dist import get_dist_info, init_dist
from mmengine.runner import Runner
from mmengine.runner.utils import set_random_seed

# from mmdet.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector (MMEngine)')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int)
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='set deterministic options for CUDNN')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in the config')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # --------------------
    # Load config
    # --------------------
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # --------------------
    # CUDNN settings
    # --------------------
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # --------------------
    # Work dir
    # --------------------
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(
            './work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume

    # --------------------
    # Distributed setup
    # --------------------
    if args.launcher == 'none':
        distributed = False
        if args.gpu_ids is not None:
            cfg.gpu_ids = args.gpu_ids
        else:
            cfg.gpu_ids = range(1 if args.gpus is None else args.gpus)
    else:
        distributed = True
        init_dist(args.launcher, **cfg.get('dist_params', {}))
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # --------------------
    # Create work dir & logger
    # --------------------
    # mkdir_or_exist(osp.abspath(cfg.work_dir))
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # --------------------
    # Log environment
    # --------------------
    # env_info = collect_env()
    # logger.info('Environment info:\n' +
    #             '\n'.join([f'{k}: {v}' for k, v in env_info.items()]))

    # logger.info(f'Distributed training: {distributed}')
    # logger.info(f'Config:\n{cfg.pretty_text}')

    # --------------------
    # Random seed
    # --------------------
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
        cfg.randomness = dict(seed=args.seed, deterministic=args.deterministic)

    # --------------------
    # Build runner & train
    # --------------------
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
