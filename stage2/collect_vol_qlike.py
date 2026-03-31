"""
collect_vol_qlike.py
각 실험(combo_dir)의 volatility_summary.csv에서 avg_QLIKE를 모아
strategy × vol_model 피벗 테이블을 생성합니다.

Usage:
    python collect_vol_qlike.py --final_dir output/final_sp500
    python collect_vol_qlike.py --final_dir output/final_sp500 --metric coverage_95
    python collect_vol_qlike.py --final_dir "output(non-sigma)/final_sp500"
"""

import argparse
import json
import os

import pandas as pd


# ---- analysis.py와 동일한 파싱 로직 ----

def _parse_combo_dir(dirname):
    parts = dirname.split('__')
    if len(parts) != 4 or not parts[3].startswith('tau'):
        return None
    try:
        rc = float(parts[3][3:].replace('p', '.').replace('m', '-'))
    except ValueError:
        return None
    return parts[0], parts[1], parts[2], rc


def _strategy_label(dr, ft, rc):
    suffix = {
        'none': '', 'pca': '_pca', 'pls': '_pls',
        'shap_pca': '_pca_shap', 'shap_pls': '_pls_shap',
    }.get(dr, f'_{dr}')
    if ft == 'past':
        return f'past{suffix}'
    return f'dual{suffix}_{rc:g}'


def collect_vol_summaries(final_dir, metric='avg_QLIKE'):
    """final_dir 하위 combo 디렉토리를 스캔하여 vol metric 수집."""
    rows = []

    for entry in sorted(os.listdir(final_dir)):
        subdir = os.path.join(final_dir, entry)
        if not os.path.isdir(subdir):
            continue

        # strategy 이름 결정 (run_config.json 우선, 없으면 디렉토리명 파싱)
        cfg_path = os.path.join(subdir, 'run_config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            dr = cfg.get('dim_reduction', '')
            ft = cfg.get('feature_type', '')
            rc = float(cfg.get('r2_cut', 0))
        else:
            parsed = _parse_combo_dir(entry)
            if parsed is None:
                continue
            dr, ft, _, rc = parsed

        name = _strategy_label(dr, ft, rc)

        # volatility_summary.csv 로딩
        vpath = os.path.join(subdir, 'volatility_summary.csv')
        if not os.path.exists(vpath):
            print(f'[WARN] Missing volatility_summary.csv: {entry}')
            continue

        vdf = pd.read_csv(vpath)
        for _, row in vdf.iterrows():
            vol_model = row.get('vol_model', '')
            val = row.get(metric, None)
            is_best = row.get('is_best_model', False)
            status = row.get('status', '')
            rows.append({
                'strategy': name,
                'combo_dir': entry,
                'vol_model': vol_model,
                metric: val,
                'is_best_model': is_best,
                'status': status,
            })

    if not rows:
        raise ValueError(f'No volatility_summary.csv found in {final_dir}')

    return pd.DataFrame(rows)


def make_pivot(df, metric='avg_QLIKE'):
    """strategy × vol_model 피벗 테이블 생성."""
    pivot = df.pivot_table(
        index='strategy', columns='vol_model', values=metric, aggfunc='first'
    )

    # best model 컬럼 추가
    best = df[df['is_best_model'] == True].drop_duplicates('strategy')
    best_map = best.set_index('strategy')['vol_model']
    pivot['best_model'] = best_map

    # 컬럼 순서 정리
    vol_cols = [c for c in pivot.columns if c != 'best_model']
    vol_cols_sorted = sorted(vol_cols)
    pivot = pivot[vol_cols_sorted + ['best_model']]

    return pivot


def main():
    parser = argparse.ArgumentParser(description='Collect vol QLIKE across experiments')
    parser.add_argument('--final_dir', type=str, required=True,
                        help='final output directory (e.g. output/final_sp500)')
    parser.add_argument('--metric', type=str, default='avg_QLIKE',
                        choices=['avg_QLIKE', 'coverage_95', 'var_MSE', 'var_RMSE', 'var_MAE'],
                        help='volatility metric to pivot (default: avg_QLIKE)')
    parser.add_argument('--output', type=str, default=None,
                        help='output CSV path (default: {final_dir}/vol_qlike_table.csv)')
    args = parser.parse_args()

    final_dir = args.final_dir
    metric = args.metric

    print(f'Scanning: {final_dir}')
    print(f'Metric : {metric}')

    df = collect_vol_summaries(final_dir, metric=metric)
    print(f'Collected {len(df)} rows from {df["strategy"].nunique()} strategies')

    pivot = make_pivot(df, metric=metric)

    out_path = args.output or os.path.join(final_dir, f'vol_{metric}_table.csv')
    pivot.to_csv(out_path)
    print(f'\nPivot table saved to: {out_path}')
    print(f'\n{pivot.to_string()}')

    # long-form도 저장 (디버깅용)
    long_path = out_path.replace('.csv', '_long.csv')
    df.to_csv(long_path, index=False)
    print(f'Long-form saved to : {long_path}')


if __name__ == '__main__':
    main()
