import sys
import os
import glob
from mmengine.config import Config

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from mmdet.ppal.sampler import *
from mmdet.ppal.builder import builder_al_sampler
from mmdet.ppal.utils.running_checks import (
    display_latest_results,
    command_with_time,
    sys_echo
)

import argparse
parser = argparse.ArgumentParser(description='Active learning arguments')
parser.add_argument('--config', required=True, type=str, help='active learning config')
parser.add_argument('--resume', required=False, type=bool, default=False, help='whether to resume training')
parser.add_argument('--model', required=True, type=str, help='running model (e.g. retinanet)')
args = parser.parse_args()

cfg = Config.fromfile(args.config)

sys_echo('>> Start COCO active learning')
sys_echo('>> Working path: %s' % cfg.get('output_dir'))
sys_echo('>> Config: %s' % args.config)
sys_echo('\n')

PYTHON = cfg.get('python_path', 'python')

ENABLE_HASH_ID_FALLBACK = False


def cleanup_cuda() -> None:
    """Best-effort CUDA memory cleanup for the *current* Python process.

    Note: this cannot free memory held by a still-running torchrun subprocess.
    """
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass

def evaluator_annfile_cfgopts(cfg_obj, key: str, ann_file: str) -> str:
    """Return cfg-options snippet to override evaluator ann_file for dict or list evaluators."""
    try:
        ev = cfg_obj.get(key)
    except Exception:
        ev = None
    if isinstance(ev, list):
        parts = []
        for i, item in enumerate(ev):
            if isinstance(item, dict) and 'ann_file' in item:
                parts.append(f'{key}.{i}.ann_file={ann_file}')
        return ' '.join(parts)
    if isinstance(ev, dict):
        return f'{key}.ann_file={ann_file}' if 'ann_file' in ev else ''
    return ''

def load_coco(json_path):
    import json
    with open(json_path, 'r') as f:
        return json.load(f)

def write_coco(json_path, data):
    import json
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(data, f)

def trim_coco_to_n_images(src_json, dst_json, n_images):
    """Trim a COCO json to first n_images (keeps categories + matching annotations)."""
    src_json = os.path.abspath(src_json)
    dst_json = os.path.abspath(dst_json)
    data = load_coco(src_json)
    imgs = data.get('images', []) or []
    anns = data.get('annotations', []) or []
    cats = data.get('categories', []) or []

    keep_imgs = imgs[:max(0, int(n_images))]
    keep_ids = set(im.get('id') for im in keep_imgs if 'id' in im)
    keep_anns = [a for a in anns if a.get('image_id') in keep_ids]

    out = dict(categories=cats, images=keep_imgs, annotations=keep_anns)
    # preserve optional keys if present
    for k in ('info', 'licenses'):
        if k in data:
            out[k] = data[k]

    write_coco(dst_json, out)
    return dst_json, len(keep_imgs)

def hash_basename_to_int(basename: str) -> int:
    """Match YOLOv7HeadFeat's legacy hashing (md5 % 1e8)."""
    import hashlib
    return int(hashlib.md5(basename.encode()).hexdigest(), 16) % (10**8)

def find_latest(path_glob: str):
    matches = glob.glob(path_glob, recursive=True)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)

def materialize_bbox_json(expected_json: str, work_dir: str, outfile_prefix: str):
    expected_json = os.path.abspath(expected_json)
    work_dir = os.path.abspath(work_dir)
    if os.path.isfile(expected_json):
        return expected_json

    cand = find_latest(os.path.join(work_dir, '**', os.path.basename(outfile_prefix) + '.bbox.json'))
    if cand and os.path.isfile(cand):
        os.makedirs(os.path.dirname(expected_json), exist_ok=True)
        try:
            import shutil
            shutil.copy2(cand, expected_json)
        except Exception as e:
            sys_echo(f'>> [WARN] Failed to copy bbox json from {cand} -> {expected_json}: {e}')
    else:
        sys_echo(f'>> [WARN] Could not find bbox json for prefix={outfile_prefix} under work_dir={work_dir}')
    return expected_json

def enrich_and_persist_bbox_json(raw_bbox_json: str, enriched_out_path: str):
    """Ensure a bbox json has the requested AL fields and persist it to a stable path.

    Required keys per det:
      - image_id (int)
      - bbox (xywh list[4])
      - score (float)
      - category_id (int)
      - cls_uncertainty (float)  [entropy fallback]
      - box_uncertainty (float)  [default 0.0]
    """
    import json
    import math
    raw_bbox_json = os.path.abspath(raw_bbox_json)
    enriched_out_path = os.path.abspath(enriched_out_path)
    if not os.path.isfile(raw_bbox_json):
        sys_echo(f'>> [WARN] Cannot enrich missing bbox json: {raw_bbox_json}')
        return False

    try:
        data = json.loads(open(raw_bbox_json, 'r').read())
    except Exception as e:
        sys_echo(f'>> [WARN] Cannot parse bbox json: {raw_bbox_json} ({e})')
        return False

    if not isinstance(data, list):
        sys_echo(f'>> [WARN] Unexpected bbox json format (expected list): {raw_bbox_json}')
        return False

    eps = 1e-10
    kept = []
    dropped = 0
    for d in data:
        if not isinstance(d, dict):
            dropped += 1
            continue
        if 'image_id' not in d or 'bbox' not in d or 'score' not in d or 'category_id' not in d:
            dropped += 1
            continue
        b = d.get('bbox')
        if not (isinstance(b, list) and len(b) == 4):
            dropped += 1
            continue

        if 'cls_uncertainty' not in d:
            s = float(d.get('score', 0.0))
            s = min(max(s, 0.0), 1.0)
            u = -1.0 * (s * math.log(s + eps) + (1.0 - s) * math.log((1.0 - s) + eps))
            d['cls_uncertainty'] = float(u)
        if 'box_uncertainty' not in d:
            d['box_uncertainty'] = 0.0
        kept.append(d)

    if dropped:
        sys_echo(f'>> [WARN] Dropped {dropped} malformed detections while enriching {os.path.basename(raw_bbox_json)}')

    os.makedirs(os.path.dirname(enriched_out_path), exist_ok=True)
    with open(enriched_out_path, 'w') as f:
        json.dump(kept, f)
    return True

def build_hash_to_oracle_id(oracle_json_path: str):
    """Build mapping: md5(file_name basename) -> COCO image_id."""
    oracle = load_coco(os.path.abspath(oracle_json_path))
    h2id = {}
    collisions = 0
    for im in oracle.get('images', []) or []:
        img_id = int(im.get('id'))
        fn = im.get('file_name') or ''
        h = hash_basename_to_int(os.path.basename(fn))
        if h in h2id and h2id[h] != img_id:
            collisions += 1
            # keep first; collisions are extremely unlikely, but we warn
            continue
        h2id[h] = img_id
    return h2id, collisions

def get_latest_ckpt(work_dir):
    """Robustly finds the latest checkpoint in MMDetection 3.x."""
    # 1. Check for the legacy 'latest.pth'
    legacy_path = os.path.join(work_dir, 'latest.pth')
    if os.path.exists(legacy_path):
        return legacy_path
    
    # 2. Check for MMEngine 'last_checkpoint' metadata file
    last_ckpt_meta = os.path.join(work_dir, 'last_checkpoint')
    if os.path.exists(last_ckpt_meta):
        with open(last_ckpt_meta, 'r') as f:
            return f.read().strip()
            
    # 3. Fallback: Find the highest numbered epoch file
    ckpts = glob.glob(os.path.join(work_dir, 'epoch_*.pth'))
    if ckpts:
        # Returns the most recently modified epoch file
        return max(ckpts, key=os.path.getmtime)
        
    return legacy_path # Final fallback to original expectation

def get_start_round():
    start_round = 0
    if args.resume:
        if not os.path.isdir(cfg.get('output_dir')):
            pass  
        else:
            k = 0
            while k < cfg.get('round_num'):
                round_work_dir = os.path.join(cfg.get('output_dir'), 'round%d' % (k + 1))
                if os.path.isfile(os.path.join(round_work_dir, 'annotations', 'new_labeled.json')):
                    k += 1
                else:
                    break
            start_round = k
    return start_round

uncertainty_sampler = builder_al_sampler(cfg.uncertainty_sampler_config)
diversity_sampler = builder_al_sampler(cfg.diversity_sampler_config)

def run(round, run_al):
    # Base Directories
    output_dir = os.path.abspath(cfg.get('output_dir'))
    last_round_work_dir = os.path.join(output_dir, 'round%d'%(round-1))
    round_work_dir      = os.path.join(output_dir, 'round%d'%round)
    
    # Converting all annotation paths to ABSOLUTE paths to avoid URI errors
    oracle_path          = os.path.abspath(cfg.get('oracle_path'))
    oracle_img_prefix    = cfg.get('oracle_img_prefix', 'train/images/')
    if isinstance(oracle_img_prefix, str) and oracle_img_prefix and not oracle_img_prefix.endswith('/'):
        oracle_img_prefix = oracle_img_prefix + '/'
    round_labeled_json   = os.path.abspath(os.path.join(round_work_dir, 'annotations', 'labeled.json'))
    round_unlabeled_json = os.path.abspath(os.path.join(round_work_dir, 'annotations', 'unlabeled.json'))
    
    round_eval_log       = os.path.join(round_work_dir, 'eval.txt')

    round_uncertainty_inference_json_prefix = os.path.abspath(os.path.join(round_work_dir, 'unlabeled_inference_result'))
    round_uncertainty_inference_json        = os.path.abspath(os.path.join(round_work_dir, 'unlabeled_inference_result.bbox.json'))
    round_uncertainty_new_labeled_json      = os.path.abspath(os.path.join(round_work_dir, 'annotations', 'uncertainty_new_labeled.json'))
    round_uncertainty_new_unlabeled_json    = os.path.abspath(os.path.join(round_work_dir, 'annotations', 'uncertainty_new_unlabeled.json'))

    round_diversity_image_dis_npy           = os.path.abspath(os.path.join(round_work_dir, 'image_dis.npy'))
    round_diversity_inference_json_prefix   = os.path.abspath(os.path.join(round_work_dir, 'diversity_inference_result'))
    round_diversity_inference_json          = os.path.abspath(os.path.join(round_work_dir, 'diversity_inference_result.bbox.json'))
    round_diversity_new_labeled_json        = os.path.abspath(os.path.join(round_work_dir, 'annotations', 'new_labeled.json'))
    round_diversity_new_unlabeled_json      = os.path.abspath(os.path.join(round_work_dir, 'annotations', 'new_unlabeled.json'))

    # Default: validate/evaluate using the datasets defined inside the train config.
    # Opt-in: validate/evaluate on oracle(train) set by setting these in the AL config.
    validate_on_oracle = bool(cfg.get('validate_on_oracle', False))
    evaluate_on_oracle = bool(cfg.get('evaluate_on_oracle', False))

    val_overrides = ''
    if validate_on_oracle:
        val_overrides = (
            f' val_dataloader.dataset.ann_file={oracle_path}'
            f' val_dataloader.dataset.data_prefix.img={oracle_img_prefix}'
            f' {evaluator_annfile_cfgopts(cfg, "val_evaluator", oracle_path)}'
        )

    train_command = 'torchrun --nproc_per_node=%d --master_port=%d ' % (int(cfg.get('gpus')), int(cfg.get('port'))) + \
                    ' tools/train_mmengine.py ' + \
                    ' %s ' % cfg.get('train_config') + \
                    ' --work-dir %s ' % round_work_dir + \
                    ' --launcher pytorch ' + \
                    (' --cfg-options train_dataloader.dataset.ann_file=%s%s' % (round_labeled_json, val_overrides))

    os.system('mkdir -p %s' % os.path.join(round_work_dir, 'annotations'))
    
    # Handle Training Phase
    if round == 1:
        os.system('cp %s %s' % (cfg.get('init_label_json'), round_labeled_json))
        os.system('cp %s %s' % (cfg.get('init_unlabeled_json'), round_unlabeled_json))
        if cfg.get('init_model', None) is not None:
            os.system('cp %s %s'%(cfg.get('init_model'), os.path.join(round_work_dir, 'latest.pth')))
        else:
            # Warm-start Round-1 from COCO pretrained weights unless user disables it.
            pretrained_ckpt = cfg.get('pretrained_ckpt', None)
            if pretrained_ckpt:
                train_command = train_command + f' --cfg-options load_from={pretrained_ckpt}'
            command_with_time(train_command, 'Training')
            cleanup_cuda()
    else:
        # Check if already trained if resuming
        current_ckpt = get_latest_ckpt(round_work_dir)
        if args.resume and os.path.exists(current_ckpt):
            pass
        else:
            os.system('cp %s %s' % (os.path.abspath(os.path.join(last_round_work_dir, 'annotations', 'new_labeled.json')), round_labeled_json))
            os.system('cp %s %s' % (os.path.abspath(os.path.join(last_round_work_dir, 'annotations', 'new_unlabeled.json')), round_unlabeled_json))
            # Warm-start each AL round from the previous round checkpoint (weights only).
            # This avoids re-training from scratch every round.
            prev_ckpt = get_latest_ckpt(last_round_work_dir)
            if prev_ckpt and os.path.exists(prev_ckpt):
                train_command = train_command + f' --cfg-options load_from={os.path.abspath(prev_ckpt)}'
            command_with_time(train_command, 'Training')
            cleanup_cuda()

    # Dynamically find the checkpoint for evaluation/inference
    latest_ckpt = get_latest_ckpt(round_work_dir)
    # Create legacy latest.pth for samplers expecting it (e.g., DCUSSampler)
    try:
        latest_pth = os.path.join(round_work_dir, 'latest.pth')
        if os.path.abspath(latest_ckpt) != os.path.abspath(latest_pth):
            if os.path.islink(latest_pth) or os.path.isfile(latest_pth):
                os.remove(latest_pth)
            os.symlink(os.path.abspath(latest_ckpt), latest_pth)
    except Exception:
        pass

    # Evaluation Command
    test_overrides = ''
    if evaluate_on_oracle:
        test_overrides = (
            f' --cfg-options test_dataloader.dataset.ann_file={oracle_path}'
            f' test_dataloader.dataset.data_prefix.img={oracle_img_prefix}'
            f' {evaluator_annfile_cfgopts(cfg, "test_evaluator", oracle_path)}'
        )
    eval_command = 'torchrun --nproc_per_node=%d --master_port=%d ' % (int(cfg.get('gpus')), int(cfg.get('port'))) + \
                   ' tools/test_mmengine.py ' + \
                   ' %s ' % cfg.get('train_config') + \
                   ' %s ' % latest_ckpt + \
                   ' --work-dir %s ' % round_work_dir + \
                   ' --launcher pytorch ' + \
                   test_overrides + \
                   ' > %s' % round_eval_log

    if not (os.path.isfile(round_eval_log) and args.resume):
        command_with_time(eval_command, 'Evaluation round %d'%round)
        cleanup_cuda()
    
    display_latest_results(cfg.get('output_dir'), round, os.path.join(cfg.get('output_dir'), 'eval_results.txt'))

    if run_al:
        # Unlabeled Inference Command
        unlabeled_infer_command = 'torchrun --nproc_per_node=%d --master_port=%d ' % (int(cfg.get('gpus')), int(cfg.get('port'))) + \
                                  ' tools/test_mmengine.py ' + \
                                  ' %s ' % cfg.get('uncertainty_infer_config') + \
                                  ' %s ' % latest_ckpt + \
                                  ' --work-dir %s ' % round_work_dir + \
                                  ' --launcher pytorch ' + \
                                  ' --cfg-options test_dataloader.dataset.ann_file=%s ' % round_unlabeled_json + \
                                  ' test_evaluator.outfile_prefix=%s' % round_uncertainty_inference_json_prefix

        if not (os.path.isfile(round_uncertainty_inference_json) and args.resume):
            command_with_time(unlabeled_infer_command, 'Inference on unlabeled data')
            cleanup_cuda()

        # Materialize + enrich + persist a stable copy for samplers + debugging
        round_uncertainty_inference_json = materialize_bbox_json(
            round_uncertainty_inference_json, round_work_dir, round_uncertainty_inference_json_prefix
        )
        enriched_unc_path = os.path.join(
            round_work_dir, 'annotations', 'unlabeled_inference_result_with_uncertainty.bbox.json'
        )
        if os.path.isfile(round_uncertainty_inference_json):
            ok = enrich_and_persist_bbox_json(round_uncertainty_inference_json, enriched_unc_path)
            if ok:
                # Overwrite the raw path too, so downstream samplers always see the enriched fields.
                try:
                    import shutil
                    shutil.copy2(enriched_unc_path, round_uncertainty_inference_json)
                except Exception as e:
                    sys_echo(f'>> [WARN] Failed to overwrite raw bbox json with enriched copy: {e}')
        else:
            sys_echo(f'>> [WARN] Missing uncertainty bbox json at: {round_uncertainty_inference_json}')
        
        if not (os.path.isfile(round_uncertainty_new_labeled_json) and args.resume):
            uncertainty_sampler.al_round(round_uncertainty_inference_json, round_labeled_json, round_uncertainty_new_labeled_json, round_uncertainty_new_unlabeled_json)

        # Diversity Phase
        num_gpus = int(cfg.get('gpus'))
        diversity_pool_json = round_uncertainty_new_labeled_json
        try:
            pool_data = load_coco(round_uncertainty_new_labeled_json)
            pool_n = len(pool_data.get('images', []) or [])
        except Exception:
            pool_n = 0

        pool_size_round = (pool_n // num_gpus) * num_gpus
        if pool_size_round <= 0:
            sys_echo(f'>> [WARN] Diversity pool is empty (images={pool_n}). Skipping diversity sampling.')
            return

        if pool_size_round != pool_n:
            trimmed_path = os.path.abspath(os.path.join(round_work_dir, 'annotations', 'diversity_pool.json'))
            diversity_pool_json, kept = trim_coco_to_n_images(round_uncertainty_new_labeled_json, trimmed_path, pool_size_round)
            pool_size_round = kept

        if args.model == 'fasterrcnn': head = 'roi_head'
        else: head = 'bbox_head'

        diversity_infer_command = 'torchrun --nproc_per_node=%d --master_port=%d ' % (int(cfg.get('gpus')), int(cfg.get('port'))) + \
                                  ' tools/test_mmengine.py ' + \
                                  ' %s ' % cfg.get('diversity_infer_config') + \
                                  ' %s ' % latest_ckpt + \
                                  ' --work-dir %s ' % round_work_dir + \
                                  ' --launcher pytorch ' + \
                                  ' --cfg-options test_dataloader.dataset.ann_file=%s ' % diversity_pool_json + \
                                  ' test_evaluator.outfile_prefix=%s ' % round_diversity_inference_json_prefix + \
                                  ' model.%s.total_images=%d ' % (head, pool_size_round) + \
                                  ' model.%s.output_path=\"%s\" ' % (head, round_diversity_image_dis_npy)

        if not (os.path.isfile(round_diversity_image_dis_npy) and args.resume):
            command_with_time(diversity_infer_command, 'Inference on diversity data')
            cleanup_cuda()

        round_diversity_inference_json = materialize_bbox_json(
            round_diversity_inference_json, round_work_dir, round_diversity_inference_json_prefix
        )
        
        if not (os.path.isfile(round_diversity_new_labeled_json) and args.resume):

            sampled_ids, _ = diversity_sampler.al_acquisition(round_diversity_image_dis_npy, round_labeled_json)


            if ENABLE_HASH_ID_FALLBACK:
                oracle_data = diversity_sampler.oracle_data
                oracle_ids = set(oracle_data.keys())
                sampled_set = set(int(s) for s in sampled_ids)
                if sampled_set and len(sampled_set & oracle_ids) == 0:
                    h2id, collisions = build_hash_to_oracle_id(oracle_path)
                    mapped = []
                    for s in sampled_ids:
                        sid = int(s)
                        if sid in h2id:
                            mapped.append(h2id[sid])
                    mapped = list(dict.fromkeys(mapped))  # stable de-dupe
                    if collisions:
                        sys_echo(f'>> [WARN] Hash collisions while mapping candidate_ids -> oracle ids: {collisions}')
                    sampled_ids = mapped

            # Recompute remaining pool in oracle-id space
            import json
            with open(round_labeled_json) as f:
                last = json.load(f)
            last_labeled_ids = set(int(x['id']) for x in last.get('images', []))
            sampled_ids = [int(x) for x in sampled_ids if int(x) not in last_labeled_ids]
            all_oracle_ids = set(diversity_sampler.oracle_data.keys())
            rest_ids = list(all_oracle_ids - (last_labeled_ids.union(set(sampled_ids))))

            diversity_sampler.create_jsons(
                sampled_ids, rest_ids, round_labeled_json,
                round_diversity_new_labeled_json, round_diversity_new_unlabeled_json
            )
    

        # Cleanup
        if os.path.exists(round_uncertainty_inference_json): os.remove(round_uncertainty_inference_json)
        if os.path.exists(round_diversity_inference_json): os.remove(round_diversity_inference_json)


if __name__ == '__main__':
    start_round = get_start_round()
    os.system('mkdir -p %s' % cfg.get('output_dir'))
    os.system('cp %s %s' % (args.config, os.path.join(cfg.get('output_dir'), os.path.split(args.config)[-1])))
    uncertainty_sampler.set_round(start_round+1)
    diversity_sampler.set_round(start_round + 1)
    for i in range(start_round, int(cfg.get('round_num'))):
        run(i+1, i!=int(cfg.get('round_num'))-1)