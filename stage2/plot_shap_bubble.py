"""
plot_shap_bubble.py
SHAP importance bubble chart & bump chart 생성 (standalone)

final 모드 / search 모드 출력 모두 지원:
  - final: selected_features_by_year.csv (long format, top-N only)
  - search: validation_shap_feature_importance.csv (wide format, all features)

Usage:
  # === final 모드 출력에서 개별 combo 지정 ===
  python plot_shap_bubble.py \
      --input output/final_sp500/shap_pca__dual__sp500__tau0p2/selected_features_by_year.csv

  # === search 모드 출력에서 wide format 직접 지정 ===
  python plot_shap_bubble.py \
      --input results/RF_sp500(dual_PCA_shap_r2cut0.2)/validation_shap_feature_importance.csv

  # === final 디렉토리 전체 스캔 (shap 실험만 자동 탐지) ===
  python plot_shap_bubble.py --scan_dir output/final_sp500

  # === GW 그룹 bubble + bump chart ===
  python plot_shap_bubble.py --input <path> --gw_group

  # === Predictor bundle stability chart (코멘트 2+3 대응) ===
  python plot_shap_bubble.py --input <path> --predictor_bundle

  # === GW 그룹 + predictor bundle 동시 생성 ===
  python plot_shap_bubble.py --input <path> --gw_group --predictor_bundle

  # === 옵션 ===
  python plot_shap_bubble.py --input <path> --n_levels 20 --topN_filter 10
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# 1. Data Loading & Format Detection
# ============================================================

def _detect_and_load(csv_path):
    """
    CSV 포맷 자동 감지 후 wide format DataFrame 반환.

    Returns:
        shap_df: pd.DataFrame (index=year, columns=feature, values=importance)
        fmt: str ('long' or 'wide')
    """
    df = pd.read_csv(csv_path)

    # long format 감지: 'year', 'feature', 'shap_importance' 컬럼 존재
    if {'year', 'feature', 'shap_importance'}.issubset(df.columns):
        shap_df = _long_to_wide(df)
        return shap_df, 'long'

    # wide format: 첫 컬럼이 date/year index
    df.rename(columns={df.columns[0]: 'date'}, inplace=True)
    df.set_index('date', inplace=True)
    return df, 'wide'


def _long_to_wide(df):
    """
    selected_features_by_year.csv (long) → wide format pivot.
    선택되지 않은 (year, feature) 조합은 0으로 채움.
    """
    pivot = df.pivot_table(
        index='year', columns='feature', values='shap_importance',
        aggfunc='first'
    ).fillna(0.0)
    pivot.index.name = 'date'
    return pivot


def _parse_combo_dir(dirname):
    """analysis.py와 동일한 combo dir 파싱."""
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


def scan_final_dir(final_dir):
    """
    final 디렉토리 전체를 스캔하여 shap 실험의 SHAP 파일 목록 반환.

    Returns:
        list of dict: [{'path': ..., 'label': ..., 'combo_dir': ...}, ...]
    """
    entries = []
    for entry in sorted(os.listdir(final_dir)):
        subdir = os.path.join(final_dir, entry)
        if not os.path.isdir(subdir):
            continue

        # run_config.json으로 shap 여부 판단
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

        if not dr.startswith('shap_'):
            continue

        label = _strategy_label(dr, ft, rc)

        # 두 파일 경로 모두 추적
        sel_path = os.path.join(subdir, 'selected_features_by_year.csv')
        full_path = os.path.join(subdir, 'validation_shap_feature_importance.csv')

        sel_exists = os.path.exists(sel_path)
        full_exists = os.path.exists(full_path)

        if not sel_exists and not full_exists:
            print(f'[WARN] No SHAP file in {entry}')
            continue

        entries.append({
            'path': sel_path if sel_exists else full_path,
            'full_shap_path': full_path if full_exists else None,
            'label': label,
            'combo_dir': entry,
        })

    return entries


# ============================================================
# 2. GW (2024) Group Mapping
# ============================================================

def define_gw_groups():
    return {
        'Macroeconomic': ['infl', 'ik', 'cay', 'ogap', 'gpce'],
        'Market Risk & Comovement': ['avgcor', 'tail', 'svar'],
        'Technical': ['tchi'],
        'Valuation': ['dp', 'dy', 'ep', 'de', 'bm'],
        'Equity Issuance & Financing': ['ntis', 'eqis'],
        'Rate, Term Structure & Credit': ['tms', 'dfy', 'dfr', 'lty', 'ltr', 'tbl'],
    }


def expand_group_mapping(group_mapping, feature_names):
    """_pred, _std 파생변수를 원변수 그룹에 자동 매핑."""
    expanded = {}
    for group, base_vars in group_mapping.items():
        expanded_vars = []
        for var in base_vars:
            for suffix in ['', '_pred', '_std']:
                candidate = f'{var}{suffix}'
                if candidate in feature_names:
                    expanded_vars.append(candidate)
        if expanded_vars:
            expanded[group] = expanded_vars
    return expanded


def aggregate_shap_by_groups(shap_df, group_mapping):
    """변수별 SHAP → 그룹별 합산."""
    rows = []
    for year in shap_df.index:
        year_shap = shap_df.loc[year]
        group_sums = {}
        for group, variables in group_mapping.items():
            group_sums[group] = sum(
                year_shap.get(v, 0) for v in variables
            )
        rows.append(pd.Series(group_sums, name=year))
    return pd.DataFrame(rows).fillna(0)


# ============================================================
# 3. Predictor Bundle Aggregation & Normalization
# ============================================================

def _extract_base_name(feature_name):
    """
    Feature name에서 base predictor name 추출.
    예: 'dy_pred' -> 'dy', 'dy_std' -> 'dy', 'dy' -> 'dy'
    """
    for suffix in ('_pred', '_std'):
        if feature_name.endswith(suffix):
            return feature_name[:-len(suffix)]
    return feature_name


def aggregate_to_predictor_bundle(shap_df):
    """
    개별 feature를 base predictor 기준으로 합산.

        S_{k,t} = sum_{c in {level, pred, std}} |phi_{k,c,t}|

    Parameters:
        shap_df: wide format DataFrame (index=year, columns=feature)

    Returns:
        bundle_df: wide format DataFrame (index=year, columns=base predictor)
        mapping: dict {base_name: [feature1, feature2, ...]}
    """
    # base name 매핑 구축
    mapping = {}
    for col in shap_df.columns:
        base = _extract_base_name(col)
        mapping.setdefault(base, []).append(col)

    # base name별 합산
    bundle_data = {}
    for base, features in mapping.items():
        bundle_data[base] = shap_df[features].sum(axis=1)

    bundle_df = pd.DataFrame(bundle_data, index=shap_df.index)
    return bundle_df, mapping


def normalize_within_year(bundle_df):
    """
    Within-year normalization:

        S_tilde_{k,t} = S_{k,t} / sum_j S_{j,t}

    각 연도의 합이 1이 되도록 정규화.
    연도 합이 0인 경우 해당 행은 0으로 유지.

    Returns:
        normalized_df: same shape, row sums = 1
    """
    row_sums = bundle_df.sum(axis=1)
    # 0으로 나누기 방지
    row_sums = row_sums.replace(0, np.nan)
    normalized_df = bundle_df.div(row_sums, axis=0).fillna(0.0)
    return normalized_df


# ============================================================
# 4. Predictor Stability Chart (신규 — 코멘트 2+3)
# ============================================================

def create_predictor_stability_chart(shap_df, n_levels=20, figsize=(15, 10),
                                     sort_by_importance=True, topN_filter=None):
    """
    Predictor-level stability bubble chart.

    Step 1: predictor bundle 합산 (dy_pred + dy_std + dy -> dy)
    Step 2: within-year normalization (share, row sum = 1)
    Step 3: quantile binning on normalized share
    Step 4: bubble chart

    Parameters:
        shap_df: wide format DataFrame (개별 feature 단위)
        n_levels: bubble 크기 단계 수
        topN_filter: 평균 share 기준 상위 N개만 표시
    
    Returns:
        fig, ax, bundle_df (raw), normalized_df, mapping
    """
    # Step 1: predictor bundle 합산
    bundle_df, mapping = aggregate_to_predictor_bundle(shap_df)

    # Step 2: within-year normalization
    normalized_df = normalize_within_year(bundle_df)

    # all-zero predictor 제거
    non_zero = normalized_df.columns[normalized_df.sum(axis=0) > 0]
    df = normalized_df[non_zero]

    # topN 필터 (평균 share 기준)
    if topN_filter and topN_filter < len(df.columns):
        mean_share = df.mean(axis=0).nlargest(topN_filter)
        df = df[mean_share.index]

    # 정렬
    if sort_by_importance:
        order = df.mean(axis=0).sort_values(ascending=False).index
        df = df[order]

    # Step 3: quantile binning on normalized share
    nz_vals = df.values[df.values > 0]
    if len(nz_vals) > 0:
        quantiles = np.linspace(0, 1, n_levels + 1)
        thresholds = np.quantile(nz_vals, quantiles)
        thresholds[0] *= 0.99
    else:
        thresholds = np.linspace(0, 1, n_levels + 1)

    base_size = 120
    bubble_sizes = base_size * np.linspace(0.3, 8.0, n_levels)

    # 색상
    n_vars = len(df.columns)
    if n_vars <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_vars))
    else:
        c1 = plt.cm.tab20(np.linspace(0, 1, 20))
        c2 = plt.cm.Set3(np.linspace(0, 1, n_vars - 20))
        colors = np.vstack([c1, c2])

    # Step 4: bubble chart
    fig, ax = plt.subplots(figsize=figsize)

    for i, predictor in enumerate(df.columns):
        for j, year in enumerate(df.index):
            value = df.loc[year, predictor]
            if value == 0:
                ax.scatter(year, i, s=15, c=[colors[i]], alpha=0.25, edgecolors='none')
            else:
                level = np.clip(np.digitize(value, thresholds) - 1, 0, n_levels - 1)
                ax.scatter(year, i, s=bubble_sizes[level], c=[colors[i]],
                           alpha=0.8, edgecolors='white', linewidth=0.8)

    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns)
    ax.set_xlabel('Year', fontsize=17)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    years = df.index
    ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
    plt.tight_layout()

    return fig, ax, bundle_df, normalized_df, mapping


# ============================================================
# 5. Bubble Chart (개별 변수, 기존)
# ============================================================

def create_bubble_chart(shap_df, n_levels=20, figsize=(15, 10),
                        sort_by_importance=True, topN_filter=None):
    """
    SHAP importance bubble chart (변수 단위).

    Parameters:
        shap_df: wide format DataFrame
        n_levels: bubble 크기 단계 수
        topN_filter: int or None. 설정하면 평균 importance 기준 상위 N개만 표시
    """
    # all-zero 변수 제거
    non_zero = shap_df.columns[shap_df.sum(axis=0) > 0]
    df = shap_df[non_zero]

    # topN 필터
    if topN_filter and topN_filter < len(df.columns):
        mean_imp = df.mean(axis=0).nlargest(topN_filter)
        df = df[mean_imp.index]

    # 정렬
    if sort_by_importance:
        order = df.mean(axis=0).sort_values(ascending=False).index
        df = df[order]

    # 분위수 기반 크기 결정
    nz_vals = df.values[df.values > 0]
    if len(nz_vals) > 0:
        quantiles = np.linspace(0, 1, n_levels + 1)
        thresholds = np.quantile(nz_vals, quantiles)
        thresholds[0] *= 0.99
    else:
        thresholds = np.linspace(0, 1, n_levels + 1)

    base_size = 100
    bubble_sizes = base_size * np.linspace(0.2, 8.0, n_levels)

    # 색상
    n_vars = len(df.columns)
    if n_vars <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_vars))
    else:
        c1 = plt.cm.tab20(np.linspace(0, 1, 20))
        c2 = plt.cm.Set3(np.linspace(0, 1, n_vars - 20))
        colors = np.vstack([c1, c2])

    fig, ax = plt.subplots(figsize=figsize)

    for i, variable in enumerate(df.columns):
        for j, year in enumerate(df.index):
            value = df.loc[year, variable]
            if value == 0:
                ax.scatter(year, i, s=15, c=[colors[i]], alpha=0.25, edgecolors='none')
            else:
                level = np.clip(np.digitize(value, thresholds) - 1, 0, n_levels - 1)
                ax.scatter(year, i, s=bubble_sizes[level], c=[colors[i]],
                           alpha=0.8, edgecolors='white', linewidth=0.8)

    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns)
    ax.set_xlabel('Date', fontsize=17)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    plt.tight_layout()

    return fig, ax, df


# ============================================================
# 6. Group Bubble Chart (GW 분류)
# ============================================================

def create_group_bubble_chart(group_shap_df, n_levels=10, figsize=(16, 8),
                              sort_by_importance=True):
    """GW 그룹별 bubble chart."""
    non_zero = group_shap_df.columns[group_shap_df.sum(axis=0) > 0]
    df = group_shap_df[non_zero]

    if sort_by_importance:
        order = df.mean(axis=0).sort_values(ascending=False).index
        df = df[order]

    # within-year normalization (row sum = 1)
    df = normalize_within_year(df)

    nz_vals = df.values[df.values > 0]
    if len(nz_vals) > 0:
        quantiles = np.linspace(0, 1, n_levels + 1)
        thresholds = np.quantile(nz_vals, quantiles)
        thresholds[0] *= 0.99
    else:
        thresholds = np.linspace(0, 1, n_levels + 1)

    base_size = 150
    bubble_sizes = base_size * np.linspace(0.5, 9.0, n_levels)

    n_groups = len(df.columns)
    if n_groups <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_groups))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, n_groups))

    fig, ax = plt.subplots(figsize=figsize)

    for i, group in enumerate(df.columns):
        for j, year in enumerate(df.index):
            value = df.loc[year, group]
            if value == 0:
                ax.scatter(year, i, s=20, c=[colors[i]], alpha=0.2, edgecolors='none')
            else:
                level = np.clip(np.digitize(value, thresholds) - 1, 0, n_levels - 1)
                ax.scatter(year, i, s=bubble_sizes[level], c=[colors[i]],
                           alpha=0.85, edgecolors='white', linewidth=1.0)

    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns, fontsize=16)
    ax.set_xlabel('Year', fontsize=17)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    years = df.index
    ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
    plt.tight_layout()

    return fig, ax, df


# ============================================================
# 7. Bump Chart (그룹 순위 변화)
# ============================================================

def create_bump_chart(group_shap_df, figsize=(13, 5)):
    """그룹별 연도별 순위 변화 bump chart."""
    rank = group_shap_df.rank(axis=1, ascending=False)

    n_groups = len(group_shap_df.columns)
    if n_groups <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_groups))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, n_groups))

    fig, ax = plt.subplots(figsize=figsize)

    for i, g in enumerate(group_shap_df.columns):
        ax.plot(rank.index, rank[g], label=g, lw=2.5,
                color=colors[i], marker='o', markersize=6, alpha=0.8)

    ax.invert_yaxis()
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Rank', fontsize=14, fontweight='bold')
    ax.set_yticks(range(1, n_groups + 1))
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(ncol=min(3, n_groups), loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=11, frameon=True, shadow=True)
    plt.tight_layout()

    return fig, ax, rank


# ============================================================
# 8. Main
# ============================================================

def process_single(csv_path, output_dir, label, args, full_shap_path=None):
    """단일 SHAP 파일 처리."""
    print(f'\n{"=" * 60}')
    print(f'  {label}')
    print(f'  Input: {csv_path}')
    print(f'{"=" * 60}')

    shap_df, fmt = _detect_and_load(csv_path)
    print(f'  Format: {fmt} -> wide shape {shap_df.shape}')

    # full SHAP 파일 탐색 (GW group용)
    # 명시적으로 지정되지 않으면, 같은 디렉토리에서 자동 탐색
    if full_shap_path is None:
        candidate = os.path.join(os.path.dirname(csv_path),
                                 'validation_shap_feature_importance.csv')
        if os.path.exists(candidate):
            full_shap_path = candidate

    full_shap_df = None
    if full_shap_path and os.path.exists(full_shap_path):
        full_shap_df, _ = _detect_and_load(full_shap_path)
        print(f'  Full SHAP: {full_shap_path} -> shape {full_shap_df.shape}')

    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, label)

    # --- 개별 변수 bubble chart ---
    n_display = args.topN_filter or shap_df.shape[1]
    fig_h = max(6, min(n_display * 0.45, 20))
    fig1, ax1, sorted_df = create_bubble_chart(
        shap_df, n_levels=args.n_levels, topN_filter=args.topN_filter,
        figsize=(15, fig_h),
    )
    path1 = f'{prefix}_bubble.png'
    fig1.savefig(path1, dpi=300, bbox_inches='tight')
    print(f'  Saved: {path1}')
    plt.close(fig1)

    # --- GW 그룹 (full SHAP 우선 사용) ---
    if args.gw_group:
        if full_shap_df is not None:
            gw_source = full_shap_df
            print(f'  GW group: using full validation SHAP')
        else:
            gw_source = shap_df
            print(f'  [WARN] GW group: full SHAP not found, '
                  f'falling back to input (top-N only)')

        base_groups = define_gw_groups()
        expanded = expand_group_mapping(base_groups, gw_source.columns.tolist())
        group_df = aggregate_shap_by_groups(gw_source, expanded)

        fig2, ax2, _ = create_group_bubble_chart(group_df, n_levels=args.n_levels)
        path2 = f'{prefix}_gw_bubble.png'
        fig2.savefig(path2, dpi=100, bbox_inches='tight')
        print(f'  Saved: {path2}')
        plt.close(fig2)

        fig3, ax3, rank_df = create_bump_chart(group_df)
        path3 = f'{prefix}_gw_bump.png'
        fig3.savefig(path3, dpi=100, bbox_inches='tight')
        print(f'  Saved: {path3}')
        plt.close(fig3)

        group_df.to_csv(f'{prefix}_gw_importance.csv')
        rank_df.to_csv(f'{prefix}_gw_rank.csv')

    # --- Predictor bundle stability chart ---
    if args.predictor_bundle:
        n_predictors = len(set(_extract_base_name(c) for c in shap_df.columns))
        top_n = args.topN_filter or n_predictors
        fig_h = max(6, min(top_n * 0.45, 20))
        fig4, ax4, bundle_df, norm_df, mapping = create_predictor_stability_chart(
            shap_df, n_levels=args.n_levels, topN_filter=args.topN_filter,
            figsize=(15, fig_h),
        )
        path4 = f'{prefix}_predictor_stability.png'
        fig4.savefig(path4, dpi=100, bbox_inches='tight')
        print(f'  Saved: {path4}')
        plt.close(fig4)

        # bundle score (raw) + normalized share 저장
        bundle_df.to_csv(f'{prefix}_predictor_bundle.csv')
        norm_df.to_csv(f'{prefix}_predictor_bundle_normalized.csv')

        # mapping 기록 (어떤 feature가 어떤 bundle에 포함됐는지)
        map_rows = []
        for base, feats in mapping.items():
            for feat in feats:
                map_rows.append({'predictor_bundle': base, 'feature': feat})
        pd.DataFrame(map_rows).to_csv(f'{prefix}_predictor_bundle_mapping.csv', index=False)
        print(f'  Saved: {prefix}_predictor_bundle*.csv')


def main():
    parser = argparse.ArgumentParser(description='SHAP Bubble Chart (standalone)')
    parser.add_argument('--input', type=str, default=None,
                        help='단일 CSV 파일 경로 (long 또는 wide format 자동 감지)')
    parser.add_argument('--scan_dir', type=str, default=None,
                        help='final 디렉토리 전체 스캔 (shap 실험만)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='출력 디렉토리 (기본: 입력 파일과 같은 위치)')
    parser.add_argument('--n_levels', type=int, default=20,
                        help='bubble 크기 단계 수 (default: 20)')
    parser.add_argument('--topN_filter', type=int, default=None,
                        help='평균 importance 기준 상위 N개 변수만 표시')
    parser.add_argument('--gw_group', action='store_true',
                        help='GW (2024) 그룹 bubble + bump chart 추가 생성')
    parser.add_argument('--predictor_bundle', action='store_true',
                        help='Predictor bundle stability chart 추가 생성 '
                             '(bundle 합산 + within-year normalization)')

    args = parser.parse_args()

    if args.input is None and args.scan_dir is None:
        parser.error('--input 또는 --scan_dir 중 하나를 지정하세요.')

    # --- 단일 파일 모드 ---
    if args.input:
        out_dir = args.output_dir or os.path.dirname(args.input) or '.'
        label = os.path.splitext(os.path.basename(args.input))[0]
        process_single(args.input, out_dir, label, args)

    # --- 디렉토리 스캔 모드 ---
    if args.scan_dir:
        entries = scan_final_dir(args.scan_dir)
        if not entries:
            print(f'No SHAP experiments found in {args.scan_dir}')
            sys.exit(1)

        print(f'Found {len(entries)} SHAP experiments')
        out_dir = args.output_dir or os.path.join(args.scan_dir, 'shap_charts')

        for entry in entries:
            process_single(entry['path'], out_dir, entry['label'], args,
                           full_shap_path=entry.get('full_shap_path'))

    print('\nDone.')


if __name__ == '__main__':
    main()
