# Copyright (c) OpenMMLab.
import argparse
import os
import os.path as osp
import time
import warnings

from mmengine.config import Config, DictAction
from mmengine.dist import get_dist_info, init_dist
from mmengine.runner import Runner


def _parse_kv_list(kv_list):
    """Parse ['k=v', 'a=b'] into dict."""
    out = {}
    if not kv_list:
        return out
    for item in kv_list:
        if '=' not in item:
            continue
        k, v = item.split('=', 1)
        v = v.strip()
        # basic bool/int/float parsing
        if v.lower() in ('true', 'false'):
            vv = v.lower() == 'true'
        else:
            try:
                vv = int(v)
            except ValueError:
                try:
                    vv = float(v)
                except ValueError:
                    vv = v
        out[k.strip()] = vv
    return out


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet/MMYOLO test (MMEngine)')

    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')

    parser.add_argument('--work-dir', help='directory to save metrics/results')
    parser.add_argument('--out', help='output result file (best-effort)')

    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', help='directory to save painted images')
    parser.add_argument('--show-score-thr', type=float, default=0.3)

    # Compatibility flags
    parser.add_argument('--eval', type=str, nargs='+')
    parser.add_argument('--format-only', action='store_true')

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config settings, key=value format')

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none')
    parser.add_argument('--local_rank', type=int, default=0)

    # IMPORTANT: swallow PPAL/MMDet2 legacy flags
    # e.g. --eval-option classwise=True, --eval-options jsonfile_prefix=...
    args, unknown = parser.parse_known_args()

    # Make LOCAL_RANK available (torchrun uses env)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args, unknown


def _inject_visualization(cfg: Config, show: bool, show_dir: str, score_thr: float):
    if not (show or show_dir):
        return

    default_hooks = cfg.get('default_hooks', None)
    if default_hooks is None:
        cfg.default_hooks = {}
        default_hooks = cfg.default_hooks

    vis_hook = default_hooks.get('visualization', None)
    if vis_hook is None:
        default_hooks['visualization'] = dict(type='VisualizationHook')
        vis_hook = default_hooks['visualization']

    vis_hook['draw'] = True
    vis_hook['interval'] = 1
    vis_hook['show'] = bool(show)
    if show_dir is not None:
        vis_hook['out_dir'] = show_dir
    vis_hook['score_thr'] = float(score_thr)


def _inject_result_dump(cfg: Config, out_path: str):
    if not out_path:
        return

    prefix = out_path
    for ext in ('.pkl', '.pickle', '.json'):
        if prefix.endswith(ext):
            prefix = prefix[:-len(ext)]
            break

    if cfg.get('test_evaluator', None) is not None and isinstance(cfg.test_evaluator, dict):
        cfg.test_evaluator.setdefault('outfile_prefix', prefix)
    elif cfg.get('val_evaluator', None) is not None and isinstance(cfg.val_evaluator, dict):
        cfg.val_evaluator.setdefault('outfile_prefix', prefix)
    else:
        warnings.warn(
            'No test_evaluator/val_evaluator found; --out may be ignored.'
        )


def _apply_legacy_eval_options(cfg: Config, unknown_args):
    """Convert legacy MMDet2 args to MMDet3 evaluator keys (best-effort)."""
    # Accept both:
    # --eval-option k=v
    # --eval-options k=v k2=v2
    legacy = []

    i = 0
    while i < len(unknown_args):
        tok = unknown_args[i]
        if tok in ('--eval-option', '--eval-options', '--eval-option=',
                   '--eval-options='):
            # next tokens until next flag
            j = i + 1
            while j < len(unknown_args) and not unknown_args[j].startswith('--'):
                legacy.append(unknown_args[j])
                j += 1
            i = j
            continue
        # also handle --eval-options=k=v (rare)
        if tok.startswith('--eval-options=') or tok.startswith('--eval-option='):
            legacy.append(tok.split('=', 1)[1])
        i += 1

    if not legacy:
        return

    opts = _parse_kv_list(legacy)

    # Map MMDet2 jsonfile_prefix -> MMDet3 outfile_prefix
    if 'jsonfile_prefix' in opts and 'outfile_prefix' not in opts:
        opts['outfile_prefix'] = opts.pop('jsonfile_prefix')

    # Apply to evaluator dict
    for key in ('test_evaluator', 'val_evaluator'):
        if cfg.get(key, None) is not None and isinstance(cfg[key], dict):
            # best-effort merge
            for k, v in opts.items():
                cfg[key][k] = v
            return

    warnings.warn(
        f'Legacy eval options provided {opts}, but no evaluator found in config.'
    )


def main():
    args, unknown = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Work dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # Dist init
    if args.launcher != 'none':
        init_dist(args.launcher, **cfg.get('dist_params', {}))

    rank, _ = get_dist_info()
    if rank == 0:
        os.makedirs(cfg.work_dir, exist_ok=True)

    # Optional visualization
    _inject_visualization(cfg, args.show, args.show_dir, args.show_score_thr)

    # Output dumping (best-effort)
    _inject_result_dump(cfg, args.out)

    # Apply legacy eval-options coming from PPAL/MMDet2 scripts
    _apply_legacy_eval_options(cfg, unknown)

    # Load checkpoint
    cfg.load_from = args.checkpoint

    # Optional metric override (best-effort)
    if args.eval:
        for key in ('test_evaluator', 'val_evaluator'):
            if cfg.get(key, None) is not None and isinstance(cfg[key], dict):
                cfg[key]['metric'] = args.eval
                break

    runner = Runner.from_cfg(cfg)

    if cfg.get('test_dataloader', None) is not None:
        metrics = runner.test()
    else:
        warnings.warn('No test_dataloader found; running validation (runner.val()).')
        metrics = runner.val()

    if rank == 0 and metrics is not None:
        ts = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        metrics_path = osp.join(cfg.work_dir, f'metrics_{ts}.json')
        try:
            import mmengine
            mmengine.dump(
                dict(
                    config=args.config,
                    checkpoint=args.checkpoint,
                    metrics=metrics,
                    unknown_args=unknown,
                ),
                metrics_path
            )
            print(f'\nSaved metrics to: {metrics_path}')
        except Exception as e:
            print(f'\nCould not save metrics json: {e}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
