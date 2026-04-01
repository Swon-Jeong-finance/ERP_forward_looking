"""run_stage1_D.py

Parallel runner for Stage-1 forecasting (ARIMAX / ETS / GPR).

Exogenous variables are specified via --exo_map (JSON file, default: exo.json).

How to run
----------

1) ARIMAX with exo map:

    python3 run_stage1_D.py \
      --model arimax \
      --targets default \
      --exo_map exo.json \
      --data_dir data \
      --save_dir "results(arima)" \
      --start_year 1952 --end_year 2024 \
      --initial_train_years 20 \
      --order_fixed \
      --n_jobs 4

2) ETS robustness (univariate, ignores exo_map):

    python3 run_stage1_D.py \
      --model ets \
      --targets default \
      --sigma_source garch \
      --ets_allow_seasonal \
      --n_jobs 4

3) GPR with exo map (reuse ARIMAX selections):

    python3 run_stage1_D.py \
      --model gpr \
      --targets default \
      --exo_map exo.json \
      --sigma_source native \
      --gpr_ar_lags 2 \
      --gpr_alpha 1e-2 \
      --gpr_kernel matern52 \
      --n_jobs 4

exo.json format
---------------
{
  "dp": "tbl",
  "ep": "None",
  "tbl": "tms",
  ...
}

Notes
-----
- If a target is not in the exo_map, it runs with exo_var='None'.
- If --resume is enabled, targets with an existing best.csv are skipped.
- Summary is written to stage1_best_summary.csv.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd


def _set_thread_env():
    """Keep BLAS/NumExpr from oversubscribing when we parallelize."""
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')


def _parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(',') if x.strip()]


def _best_csv_path(save_dir: str, model: str, start_year: int, order_fixed: bool, target: str) -> str:
    root = f"{start_year}(fixed order)" if order_fixed else f"{start_year}"
    m = (model or 'arimax').strip().lower()
    if m in {'arima', 'arimax', 'sarimax'}:
        return os.path.join(save_dir, root, target, 'best.csv')
    if m == 'grf':
        m = 'gpr'
    return os.path.join(save_dir, m, root, target, 'best.csv')


def _load_stage1_module(stage1_file: str):
    """Load a stage1 implementation from an explicit .py file path."""
    stage1_file = os.path.abspath(stage1_file)
    spec = importlib.util.spec_from_file_location('stage1_mod', stage1_file)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import stage1 from file: {stage1_file}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _worker(target: str, params: dict) -> dict:
    stage1_file = params.get('_stage1_file')
    if not stage1_file:
        stage1_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stage1_D.py')
    stage1 = _load_stage1_module(stage1_file)

    call_params = dict(params)
    call_params.pop('_stage1_file', None)

    # Extract exo_var for this target from the map
    exo_map = call_params.pop('_exo_map', {})
    exo_var = exo_map.get(target, 'None')

    return stage1.main(target, exo_var, **call_params)


def main():
    parser = argparse.ArgumentParser(description='Parallel runner for Stage-1 forecasting')

    parser.add_argument('--model', type=str, default='arimax',
                        choices=['arimax', 'ets', 'gpr'],
                        help="Stage-1 model: 'arimax' (default) | 'ets' | 'gpr'")
    parser.add_argument('--targets', type=str, default='default',
                        help="Comma-separated list of targets, or 'default' (default: default).")
    parser.add_argument('--targets_file', type=str, default=None,
                        help='Optional newline-separated target list file.')
    parser.add_argument('--stage1_file', type=str, default=None,
                        help='Path to the stage1 implementation .py file (default: stage1_D.py).')
    parser.add_argument('--exo_map', type=str, default='exo.json',
                        help='JSON file mapping target -> exo_var (default: exo.json).')

    # Data / output
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--start_year', type=int, default=1952)
    parser.add_argument('--end_year', type=int, default=None)
    parser.add_argument('--initial_train_years', type=int, default=20)
    parser.add_argument('--order_fixed', action=argparse.BooleanOptionalAction, default=True)

    # Sigma / volatility
    parser.add_argument('--sigma_source', type=str, default=None,
                        choices=['garch', 'native', 'resid_var'])
    parser.add_argument('--garch_order_fixed', action=argparse.BooleanOptionalAction, default=False)

    # ETS knobs
    parser.add_argument('--ets_allow_seasonal', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--ets_select_criterion', type=str, default='aicc',
                        choices=['aicc', 'aic', 'bic'])

    # GPR knobs
    parser.add_argument('--gpr_max_train_size', type=int, default=None)
    parser.add_argument('--gpr_ar_lags', type=int, default=2)
    parser.add_argument('--gpr_target_transform', type=str, default='auto')
    parser.add_argument('--gpr_add_time_index', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--gpr_alpha', type=float, default=1e-0)
    parser.add_argument('--gpr_optimizer', type=str, default=None)
    parser.add_argument('--gpr_kernel', type=str, default='matern52')
    parser.add_argument('--gpr_kernel_candidates', type=str, default=None)
    parser.add_argument('--gpr_kernel_select', type=str, default='lml', choices=['lml', 'mse'])

    # Parallelism / runtime
    default_n_jobs = max(1, (os.cpu_count() or 1) // 4)
    parser.add_argument('--n_jobs', type=int, default=default_n_jobs)
    parser.add_argument('--spawn', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--set_threads', action=argparse.BooleanOptionalAction, default=True)

    # Resume / outputs
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--force', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--summary_name', type=str, default='stage1_best_summary.csv')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    if args.set_threads:
        _set_thread_env()

    if args.spawn:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    # Load stage1 module
    stage1_file = args.stage1_file
    if stage1_file is None:
        stage1_file = os.path.join(script_dir, 'stage1.py')
    stage1 = _load_stage1_module(stage1_file)

    # Targets
    if args.targets_file:
        with open(args.targets_file, 'r') as f:
            targets = [ln.strip() for ln in f if ln.strip()]
    else:
        key = (args.targets or 'default').strip().lower()
        if key in {'default', 'all', '*'}:
            targets = list(stage1.DEFAULT_TARGETS)
        else:
            targets = _parse_csv_list(args.targets)
    targets = stage1.strip_slash(targets)

    # Load exo map
    exo_map: dict[str, str] = {}
    if os.path.exists(args.exo_map):
        with open(args.exo_map, 'r') as f:
            exo_map = json.load(f)
        if args.verbose:
            print(f"Loaded exo map from {args.exo_map} ({len(exo_map)} entries)")
    else:
        if args.verbose:
            print(f"No exo map found at {args.exo_map} -- all targets will run with exo_var='None'.")

    gpr_kernel_candidates = None
    if args.gpr_kernel_candidates:
        gpr_kernel_candidates = _parse_csv_list(args.gpr_kernel_candidates)

    # Build parameters
    params = dict(
        _stage1_file=stage1_file,
        _exo_map=exo_map,
        stage1_model=args.model,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        initial_train_years=args.initial_train_years,
        order_fixed=args.order_fixed,
        sigma_source=args.sigma_source,
        garch_order_fixed=args.garch_order_fixed,
        ets_allow_seasonal=args.ets_allow_seasonal,
        ets_select_criterion=args.ets_select_criterion,
        gpr_max_train_size=args.gpr_max_train_size,
        gpr_ar_lags=args.gpr_ar_lags,
        gpr_target_transform=args.gpr_target_transform,
        gpr_add_time_index=args.gpr_add_time_index,
        gpr_alpha=args.gpr_alpha,
        gpr_optimizer=args.gpr_optimizer,
        gpr_kernel=args.gpr_kernel,
        gpr_kernel_candidates=gpr_kernel_candidates,
        gpr_kernel_select=args.gpr_kernel_select,
        verbose=args.verbose,
    )

    # Resume logic
    results: list[dict] = []
    to_run: list[str] = []

    for t in targets:
        best_path = _best_csv_path(args.save_dir, args.model, args.start_year, args.order_fixed, t)
        if args.resume and (not args.force) and os.path.exists(best_path):
            try:
                row = pd.read_csv(best_path).iloc[0].to_dict()
                row['_status'] = 'skipped(resume)'
                results.append(row)
                continue
            except Exception:
                pass
        to_run.append(t)

    failures: list[dict] = []

    if len(to_run) and args.n_jobs == 1:
        for t in to_run:
            try:
                row = _worker(t, params)
                row['_status'] = 'ok'
                results.append(row)
            except Exception as e:
                failures.append({'variable': t, 'error': str(e), 'traceback': traceback.format_exc()})

    elif len(to_run):
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            fut_map = {ex.submit(_worker, t, params): t for t in to_run}
            for fut in as_completed(fut_map):
                t = fut_map[fut]
                try:
                    row = fut.result()
                    row['_status'] = 'ok'
                    results.append(row)
                except Exception as e:
                    failures.append({'variable': t, 'error': str(e), 'traceback': traceback.format_exc()})

    # Save summary
    root = f"{args.start_year}(fixed order)" if args.order_fixed else f"{args.start_year}"
    m = (args.model or 'arimax').strip().lower()
    if m in {'arima', 'arimax', 'sarimax'}:
        out_root = os.path.join(args.save_dir, root)
    else:
        if m == 'grf':
            m = 'gpr'
        out_root = os.path.join(args.save_dir, m, root)
    os.makedirs(out_root, exist_ok=True)

    summary_path = os.path.join(out_root, args.summary_name)
    df_summary = pd.DataFrame(results)
    if 'OOS_R2' in df_summary.columns:
        df_summary = df_summary.sort_values('OOS_R2', ascending=False).reset_index(drop=True)
    df_summary.to_csv(summary_path, index=False)

    if failures:
        err_path = os.path.join(out_root, args.summary_name.replace('.csv', '_errors.csv'))
        pd.DataFrame(failures).to_csv(err_path, index=False)

    if args.verbose:
        print(f"Saved summary: {summary_path} ({len(results)} rows)")
        if failures:
            print(f"Saved errors:  {err_path} ({len(failures)} failures)")


if __name__ == '__main__':
    main()
