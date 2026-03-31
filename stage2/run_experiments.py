#!/usr/bin/env python3
"""
통합 RF 실험 시스템 실행기 (v1 revised) - Final 모드 전용

사용법:
  # JSON 설정 파일로 다수 combo 실행 (주 사용 방식)
  python run_experiments_v1.py --config final_config.json

  # 단일 combo 직접 지정
  python run_experiments_v1.py --dim_reduction none --feature_type dual \\
      --index_type sp500 --r2_cut 0.1
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from config import (
    DEFAULT_TAIL_Q_LIST,
    generate_param_grid,
    get_output_root,
    load_config,
    norm_grid,
)
from data_loader import prepare_data
from pipeline import run_final_protocol


# ============================================================
# Worker cache
# ============================================================
_CACHE: Dict[Tuple[str, str, float, bool], Any] = {}


def _init_worker():
    global _CACHE
    _CACHE = {}
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except Exception:
        pass


def _filter_sigma_columns(data, feature_names, all_vars):
    drop_cols = [c for c in data.columns if c.endswith('_std')]
    data2 = data.drop(columns=drop_cols, errors='ignore').copy()
    fn2 = [c for c in feature_names if not c.endswith('_std')]
    av2 = [c for c in all_vars if not c.endswith('_std')]
    return data2, fn2, av2


def _get_data_cached(feature_type, index_type, r2_cut, use_sigma=True):
    global _CACHE
    key = (feature_type, index_type, float(r2_cut), bool(use_sigma))
    if key not in _CACHE:
        data, fn, av = prepare_data(feature_type, index_type, float(r2_cut))
        if not use_sigma:
            data, fn, av = _filter_sigma_columns(data, fn, av)
        _CACHE[key] = (data, fn, av)
    return _CACHE[key]


# ============================================================
# Final worker
# ============================================================
def _worker_final(task):
    try:
        data, feature_names, all_vars = _get_data_cached(
            task['feature_type'], task['index_type'],
            task['r2_cut'], use_sigma=bool(task.get('use_sigma', True)))

        result = run_final_protocol(
            dim_reduction=task['dim_reduction'],
            data=data, feature_names=feature_names, all_vars=all_vars,
            feature_type=task['feature_type'],
            index_type=task['index_type'],
            r2_cut=task['r2_cut'],
            param_grid=task['param_grid'],
            output_dir=task['output_dir'],
            model_params=task.get('model_params'),
            scoring=task.get('scoring', 'neg_mean_squared_error'),
            n_components=task.get('n_components'),
            topN=task.get('topN'),
            tail_q_list=task.get('tail_q_list'),
            use_sigma=bool(task.get('use_sigma', True)),
            verbose=task.get('verbose', False),
        )
        return {'status': 'success', 'result': result, 'task': task}
    except Exception as e:
        import traceback
        print(f"[ERROR] Final failed: {e}")
        traceback.print_exc()
        return {'status': 'error', 'error': str(e), 'task': task, 'result': None}


# ============================================================
# Helpers
# ============================================================
def _combo_label(dr, ft, it, rc):
    return f'{dr}/{ft}/{it}/tau={float(rc):g}'


def _safe_tau_tag(rc):
    return format(float(rc), 'g').replace('-', 'm').replace('.', 'p')


def _combo_dir_name(dr, ft, it, rc):
    return f'{dr}__{ft}__{it}__tau{_safe_tau_tag(rc)}'


def _write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def _resolve_tail_q_list(args, config):
    if args.tail_q is not None and len(args.tail_q) > 0:
        return [float(x) for x in args.tail_q]
    cfg = config.get('tail_q_list')
    if cfg and len(cfg) > 0:
        return [float(x) for x in cfg]
    return list(DEFAULT_TAIL_Q_LIST)


# ============================================================
# Config loading
# ============================================================
def _load_final_config(config_path):
    """JSON 설정 파일 로딩.

    Expected format:
    {
      "param_grid": { ... },
      "combos": [
        {"dim_reduction": "none", "feature_type": "dual", "index_type": "sp500",
         "r2_cut": 0.0},
        {"dim_reduction": "pca", ..., "n_components": 5},
        {"dim_reduction": "shap_pca", ..., "topN": 10, "n_components": 3},
        ...
      ]
    }
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    param_grid = cfg['param_grid']
    combos = cfg['combos']

    # Validate
    for i, c in enumerate(combos):
        for key in ('dim_reduction', 'feature_type', 'index_type', 'r2_cut'):
            if key not in c:
                raise ValueError(f"combo[{i}] missing required key: '{key}'")

    return param_grid, combos


# ============================================================
# Final execution
# ============================================================
def _run_final(args, config):
    # ---- Load combos ----
    if args.config:
        param_grid, combos = _load_final_config(args.config)
    else:
        # 단일 combo: CLI에서 직접 지정
        param_grid = generate_param_grid(config['param_space'])
        combos = [{
            'dim_reduction': args.dim_reduction,
            'feature_type': args.feature_type,
            'index_type': args.index_type,
            'r2_cut': float(args.r2_cut),
        }]
        if args.topN is not None:
            combos[0]['topN'] = args.topN
        if args.n_components is not None:
            combos[0]['n_components'] = args.n_components

    tail_q_list = _resolve_tail_q_list(args, config)

    # ---- Output directory ----
    first_index = combos[0]['index_type'] if combos else 'sp500'
    output_root = args.output_root or get_output_root(
        use_sigma=not args.no_sigma, index_type=first_index)
    run_tag = args.run_tag or '_'.join(sorted(set(c['index_type'] for c in combos)))
    run_dir = os.path.join(output_root, f'final_{run_tag}')

    if os.path.exists(run_dir) and os.listdir(run_dir):
        print(f"[WARN] Output dir already exists, new results will be added/updated: {run_dir}")
    os.makedirs(run_dir, exist_ok=True)

    # ---- Save batch config ----
    _write_json(os.path.join(run_dir, 'batch_config.json'), {
        'param_grid': param_grid,
        'use_sigma': not args.no_sigma,
        'scoring': config['scoring'],
        'tail_q_list': tail_q_list,
        'model': config['model'],
        'n_combos': len(combos),
    })

    # ---- Print plan ----
    print('\n' + '=' * 70)
    print('  FINAL MODE')
    print('=' * 70)
    print(f'  use_sigma:      {not args.no_sigma}')
    print(f'  scoring:        {config["scoring"]}')
    print(f'  tail_q_list:    {tail_q_list}')
    print(f'  run_dir:        {run_dir}')
    print(f'  combos:         {len(combos)}')
    for i, c in enumerate(combos, 1):
        dr, ft, it, rc = c['dim_reduction'], c['feature_type'], c['index_type'], c['r2_cut']
        topN = c.get('topN')
        nc = c.get('n_components')
        extra = ''
        if topN is not None:
            extra += f' topN={topN}'
        if nc is not None:
            extra += f' nc={nc}'
        print(f'    [{i:2d}] {_combo_label(dr, ft, it, rc)}{extra}')
    print('=' * 70)

    # ---- Build tasks ----
    tasks = []
    multi_combo = len(combos) > 1
    for c in combos:
        dr = c['dim_reduction']
        ft = c['feature_type']
        it = c['index_type']
        rc = float(c['r2_cut'])
        combo_name = _combo_dir_name(dr, ft, it, rc)
        out_dir = os.path.join(run_dir, combo_name) if multi_combo else run_dir
        if multi_combo:
            os.makedirs(out_dir, exist_ok=True)

        task = {
            'dim_reduction': dr, 'feature_type': ft,
            'index_type': it, 'r2_cut': rc,
            'param_grid': param_grid, 'output_dir': out_dir,
            'model_params': config['model'], 'scoring': config['scoring'],
            'n_components': c.get('n_components'),
            'topN': c.get('topN'),
            'tail_q_list': tail_q_list,
            'use_sigma': not args.no_sigma,
            'verbose': False,
            'combo_key': (dr, ft, it, rc),
            'combo_name': combo_name,
        }
        tasks.append(task)

        _write_json(os.path.join(out_dir, 'run_config.json'), {
            'combo_name': combo_name, 'dim_reduction': dr,
            'feature_type': ft, 'index_type': it, 'r2_cut': rc,
            'topN': c.get('topN'), 'n_components': c.get('n_components'),
            'param_grid': param_grid, 'scoring': config['scoring'],
            'tail_q_list': tail_q_list,
        })

    if args.dry_run:
        print(f'\n[DRY RUN] {len(tasks)} final tasks.')
        return

    # ---- Execute ----
    start_time = time.time()
    results_rows = []
    completed, failed = 0, 0

    if len(tasks) == 1:
        iterator = [_worker_final(tasks[0])]
        pool = None
    else:
        workers = max(1, min(args.max_workers, len(tasks)))
        pool = mp.Pool(processes=workers, initializer=_init_worker)
        iterator = pool.imap_unordered(_worker_final, tasks, chunksize=1)

    try:
        for res in iterator:
            task = res['task']
            dr, ft, it, rc = task['combo_key']
            row = {'status': res['status'],
                   'dim_reduction': dr, 'feature_type': ft,
                   'index_type': it, 'r2_cut': float(rc)}
            if res['status'] == 'success':
                completed += 1
                row.update(res['result']['metrics_summary'])
                print(f"  [✓ {completed+failed}/{len(tasks)}] {_combo_label(dr,ft,it,rc)} "
                      f"OOS_R²={res['result']['metrics_summary'].get('OOS_R2')}")
            else:
                failed += 1
                row['error'] = res.get('error')
                print(f"  [✗ {completed+failed}/{len(tasks)}] {_combo_label(dr,ft,it,rc)} "
                      f"ERROR: {str(res.get('error',''))[:160]}")
            results_rows.append(row)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if results_rows:
        pd.DataFrame(results_rows).sort_values(
            by=['feature_type', 'index_type', 'r2_cut', 'dim_reduction']
        ).to_csv(os.path.join(run_dir, 'final_runs_summary.csv'), index=False)

    elapsed = time.time() - start_time
    h, m, s = int(elapsed//3600), int((elapsed%3600)//60), int(elapsed%60)
    print(f'\n  FINAL COMPLETE: {completed} done, {failed} failed, {h}h {m}m {s}s')
    print(f'  Saved to: {run_dir}')


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='통합 RF 실험 시스템 - Final 모드')

    # Config file (주 사용 방식)
    parser.add_argument('--config', type=str, default=None,
                        help='JSON 설정 파일 경로 (param_grid + combos)')

    # 단일 combo 직접 지정 (--config 미사용 시)
    parser.add_argument('--dim_reduction', type=str, default=None,
                        choices=['none', 'pca', 'pls', 'shap_pca', 'shap_pls'])
    parser.add_argument('--feature_type', type=str, default=None,
                        choices=['dual', 'past'])
    parser.add_argument('--index_type', type=str, default=None,
                        choices=['sp500', 'crsp_index'])
    parser.add_argument('--r2_cut', type=float, default=None)
    parser.add_argument('--topN', type=int, default=None)
    parser.add_argument('--n_components', type=int, default=None)

    # Execution
    parser.add_argument('--max_workers', type=int, default=10)
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--run_tag', type=str, default=None)
    parser.add_argument('--tail_q', type=float, nargs='*', default=None)
    parser.add_argument('--no_sigma', action='store_true')
    parser.add_argument('--dry_run', action='store_true')

    args = parser.parse_args()

    # Validate: either --config or CLI args
    if args.config is None:
        for required in ('dim_reduction', 'feature_type', 'index_type', 'r2_cut'):
            if getattr(args, required) is None:
                parser.error(f'--{required} is required when --config is not specified')

    config = load_config()
    _run_final(args, config)


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
