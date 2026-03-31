import numpy as np
import pandas as pd
from tqdm import tqdm
# NOTE:
#   The original implementation uses the `arch` package for GARCH volatility.
#   Some environments (e.g., lightweight CI) may not have it installed.
#   We therefore import it defensively and fall back to NaNs when unavailable.
try:
    from arch import arch_model  # type: ignore
except Exception:  # pragma: no cover
    arch_model = None
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm  #
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, Matern, ExpSineSquared
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import itertools
import os
import time
import glob
import json

import argparse
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Defaults
# ------------------------------------------------------------
DEFAULT_DATA_DIR = 'data'
DEFAULT_SAVE_DIR = 'results'
DEFAULT_START_YEAR = 1952
DEFAULT_END_YEAR = None
DEFAULT_INITIAL_TRAIN_YEARS = 20
DEFAULT_ORDER_FIXED = True
# NOTE: In case BLAS tries to oversubscribe threads when you parallelize targets,
# set these in your runner before spawning processes:
# OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1

n_cpu = max(1, multiprocessing.cpu_count() // 10)

save_dir = DEFAULT_SAVE_DIR
os.makedirs(save_dir, exist_ok=True)


def strip_slash(lst):
    return [col.replace('/', '') for col in lst]


def convert_yyyyq_to_datetime(x):
    year = x // 10
    quarter = x % 10
    month = {1: 1, 2: 4, 3: 7, 4: 10}[quarter]
    return pd.Timestamp(f"{year}-{month:02d}-01")


def r2_return(model_name, y_true, y_pred, data_type='OOS', verbose=True):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    if verbose:
        print(f"--- {model_name} [{data_type}] 모델 성능 ---")
        print(f"Mean Squared Error (MSE): {mse:.8f}")
        print(f"R-squared (R²): {r2:.4f}")
        print("\n")
    return mse, r2


def r2_return_oos(model_name, y_true, y_pred, y_benchmark_pred, data_type='OOS', verbose=True):
    """Campbell and Thompson (2008)의 OOS R² 정의."""
    y_true, y_pred = y_true.align(y_pred, join='inner')
    y_true, y_benchmark_pred = y_true.align(y_benchmark_pred, join='inner')

    sse_model = (y_true - y_pred).pow(2).sum()
    sse_benchmark = (y_true - y_benchmark_pred).pow(2).sum()
    r2 = 1 - (sse_model / sse_benchmark)

    mse = mean_squared_error(y_true, y_pred)

    if verbose:
        print(f"--- {model_name} [{data_type}] 모델 성능 (Campbell & Thompson 기준) ---")
        print(f"Mean Squared Error (MSE): {mse:.8f}")
        print(f"Out-of-Sample R-squared (R²_OS): {r2:.4f}")
        print("\n")

    return mse, r2


def r2_level(model_name, y_true, y_pred, data_type='OOS', verbose=True):
    # ① 정렬
    y_true, y_pred = y_true.align(y_pred, join='inner')
    # ② 첫 시점 제외: RW 벤치마크 정의상 t≥2에서만 비교
    e_model = (y_true - y_pred).iloc[1:] ** 2
    e_rw = (y_true.diff()).iloc[1:] ** 2
    r2 = 1 - e_model.sum() / e_rw.sum()
    mse = mean_squared_error(y_true.iloc[1:], y_pred.iloc[1:])
    if verbose:
        print(f"--- {model_name} [{data_type}] 모델 성능 ---")
        print(f"Mean Squared Error (MSE): {mse:.8f}")
        print(f"R-squared (R²): {r2:.4f}\n")
    return mse, r2

def r2_return_oos_ao(model_name, y_true, y_pred, y_train=None, window=12,
                     data_type='OOS', verbose=True):
    """AO-12 벤치마크(직전 12개월 평균) 대비 OOS R²."""
    y_full = pd.concat([y_train, y_true]) if y_train is not None else y_true.copy()
    y_bench_full = y_full.rolling(window=window, min_periods=window).mean().shift(1)
    y_benchmark_pred = y_bench_full.loc[y_true.index]

    y_true, y_pred = y_true.align(y_pred, join='inner')
    y_true, y_benchmark_pred = y_true.align(y_benchmark_pred, join='inner')

    sse_model = ((y_true - y_pred)**2).sum()
    sse_bench = ((y_true - y_benchmark_pred)**2).sum()

    r2 = np.nan
    if np.isfinite(sse_bench) and sse_bench > 0:
        r2 = 1 - (sse_model / sse_bench)

    mse = mean_squared_error(y_true, y_pred) if len(y_true) else np.nan

    if verbose:
        print(f"--- {model_name} [{data_type}] (AO-12 benchmark) ---")
        print(f"MSE: {mse:.8f}")
        print(f"R²_OS_AO12: {r2:.4f}\n")

    return mse, r2


m_vars_ = ['date', 'price', 'lty', 'ltr', 'tbl', 'd/p', 'd/y', 'e/p', 'd/e', 'tms', 'dfy', 'dfr', 'infl', 'svar', 'ntis', 'b/m', 'tchi', 'ogap', 'tail', 'avgcor']
q_vars_ = ['date', 'cay', 'i/k']
y_vars_ = ['date', 'eqis', 'gpce']

m_vars = strip_slash(m_vars_)
q_vars = strip_slash(q_vars_)
y_vars = strip_slash(y_vars_)

# 그룹 1: 랜덤워크(RW1) 벤치마크를 사용하는 레벨 변수 그룹
vars_level_rw1 = strip_slash([
    'lty', 'tbl', 'tms', 'dfy',  # 금리/스프레드
    'd/p', 'd/y', 'e/p', 'd/e', 'b/m',  # 밸류/배당
    'svar', 'tail', 'avgcor',  # 변동성/테일
    'eqis', 'ntis', 'tchi', 'cay', 'ogap', 'i/k'  # 발행/심리/부/갭
])
# 그룹 2: 12개월 롤링 평균(AO) 벤치마크를 사용하는 인플레이션
vars_return_ao = strip_slash(['infl'])

# 그룹 3: 확장 역사적 평균(GW) 벤치마크를 사용하는 성장률/수익률
vars_return_gw = strip_slash(['gpce', 'dfr', 'ltr'])

level_vars = vars_level_rw1
ret_vars = vars_return_ao + vars_return_gw

# Default target list (for runner)
DEFAULT_TARGETS = sorted(set(level_vars + ret_vars))


order_fixed = DEFAULT_ORDER_FIXED


def load_data(target, data_dir: str = DEFAULT_DATA_DIR):
    """Load and align mixed-frequency dataset, relative to 'target'.

    Args:
        target: target variable name (slash-stripped, e.g., 'dp', 'bm', 'ik').
        data_dir: directory containing monthly.csv / quarterly.csv / yearly.csv.

    Returns:
        df_all: aligned predictor panel with target column included.
    """
    df_m = pd.read_csv(os.path.join(data_dir, 'monthly.csv'))
    df_m['date'] = pd.to_datetime(df_m['date'], format='%Y%m')
    df_m = df_m[m_vars_]
    df_m.set_index('date', inplace=True)

    df_q = pd.read_csv(os.path.join(data_dir, 'quarterly.csv'))
    df_q['date'] = df_q['date'].apply(convert_yyyyq_to_datetime)
    df_q = df_q[q_vars_]
    df_q.set_index('date', inplace=True)

    df_y = pd.read_csv(os.path.join(data_dir, 'yearly.csv'))
    df_y['date'] = pd.to_datetime(df_y['date'], format='%Y')
    df_y = df_y[y_vars_]
    df_y.set_index('date', inplace=True)

    df_m.columns = strip_slash(df_m.columns)
    df_q.columns = strip_slash(df_q.columns)
    df_y.columns = strip_slash(df_y.columns)

    df_m_proc = df_m.copy()
    df_q_proc = df_q.copy()
    df_y_proc = df_y.copy()

    # === 집계 규칙(월→분기/연) ===
    # 마지막 값 (레벨형)
    vars_last_m = ['lty', 'tbl', 'tms', 'dfy', 'dp', 'dy', 'ep', 'de', 'bm', 'tchi', 'ogap']
    # 평균 (흐름/변동/위험상태)
    vars_mean_m = ['infl', 'svar', 'avgcor', 'tail']
    # 합계 (누계형)
    vars_sum_m = ['ntis']
    # 수익률형 (기하누적)
    vars_ret_m = ['dfr', 'ltr']

    # === 분기→연 규칙 분리 ===
    q_last_for_A = ['cay']  # 레벨
    q_mean_for_A = ['ik']   # 흐름/비율

    if target in m_vars:
        all_dates = pd.date_range(df_m.index.min(), df_m.index.max(), freq='MS')
        df_m = df_m.reindex(all_dates)
        df_q_ffill = df_q.reindex(all_dates).ffill()
        df_y_ffill = df_y.reindex(all_dates).ffill()
        df_all = pd.concat([df_m, df_q_ffill, df_y_ffill], axis=1)
        df_all.dropna(inplace=True)

    elif target in q_vars:
        all_q_dates = df_q.index

        # 월 → 분기
        cols = [v for v in vars_last_m if v in df_m_proc.columns]
        exo1_m = df_m_proc[cols].resample('Q').last()
        exo1_m.index = exo1_m.index.to_period('Q').to_timestamp(how='start')

        cols = [v for v in vars_mean_m if v in df_m_proc.columns]
        exo2_m = df_m_proc[cols].resample('Q').mean()
        exo2_m.index = exo2_m.index.to_period('Q').to_timestamp(how='start')

        cols = [v for v in vars_sum_m if v in df_m_proc.columns]
        exo3_m = df_m_proc[cols].resample('Q').sum()
        exo3_m.index = exo3_m.index.to_period('Q').to_timestamp(how='start')

        cols = [v for v in vars_ret_m if v in df_m_proc.columns]
        exo_ret_m = (1.0 + df_m_proc[cols]).resample('Q').prod() - 1.0 if cols else pd.DataFrame(index=exo1_m.index)
        if not exo_ret_m.empty:
            exo_ret_m.index = exo_ret_m.index.to_period('Q').to_timestamp(how='start')

        # 연 변수 → 분기 인덱스로 ffill
        df_y_ffill = df_y_proc.reindex(all_q_dates).ffill()

        df_all = pd.concat([df_q_proc, exo1_m, exo2_m, exo3_m, exo_ret_m, df_y_ffill], axis=1)
        df_all.dropna(inplace=True)

    elif target in y_vars:
        # 월 → 연
        cols = [v for v in vars_last_m if v in df_m_proc.columns]
        exo1_m = df_m_proc[cols].resample('A').last()
        exo1_m.index = exo1_m.index.to_period('A').to_timestamp(how='start')

        cols = [v for v in vars_mean_m if v in df_m_proc.columns]
        exo2_m = df_m_proc[cols].resample('A').mean()
        exo2_m.index = exo2_m.index.to_period('A').to_timestamp(how='start')

        cols = [v for v in vars_sum_m if v in df_m_proc.columns]
        exo3_m = df_m_proc[cols].resample('A').sum()
        exo3_m.index = exo3_m.index.to_period('A').to_timestamp(how='start')

        cols = [v for v in vars_ret_m if v in df_m_proc.columns]
        exo_ret_m = (1.0 + df_m_proc[cols]).resample('A').prod() - 1.0 if cols else pd.DataFrame(index=exo1_m.index)
        if not exo_ret_m.empty:
            exo_ret_m.index = exo_ret_m.index.to_period('A').to_timestamp(how='start')

        # 분기 → 연 (규칙 분리)
        cols = [v for v in q_last_for_A if v in df_q_proc.columns]
        exo_last_a_q = df_q_proc[cols].resample('A').last() if cols else pd.DataFrame()
        if not exo_last_a_q.empty:
            exo_last_a_q.index = exo_last_a_q.index.to_period('A').to_timestamp(how='start')

        cols = [v for v in q_mean_for_A if v in df_q_proc.columns]
        exo_mean_a_q = df_q_proc[cols].resample('A').mean() if cols else pd.DataFrame()
        if not exo_mean_a_q.empty:
            exo_mean_a_q.index = exo_mean_a_q.index.to_period('A').to_timestamp(how='start')

        df_all = pd.concat([df_y_proc, exo1_m, exo2_m, exo3_m, exo_ret_m, exo_last_a_q, exo_mean_a_q], axis=1)
        df_all.dropna(inplace=True)

    else:
        raise ValueError(f"Unknown target '{target}'.")

    return df_all



def select_best_arima_order(y, X=None, p_range=range(0, 3), d_range=range(0, 2), q_range=range(0, 3)):
    """Search ARIMA(p,d,q) with optional exogenous X via AIC."""
    best_aic = np.inf
    best_order = None
    best_model = None

    exog_data = X if (X is not None and not X.empty) else None

    for order in itertools.product(p_range, d_range, q_range):
        try:
            model = ARIMA(endog=y, exog=exog_data, order=order).fit(method_kwargs={'warn_convergence': False})
            if model.aic < best_aic:
                best_aic = model.aic
                best_order = order
                best_model = model
        except Exception:
            continue
    return best_model, best_order



def select_best_garch_order(res, vol='GARCH', p_range=range(1, 3), q_range=range(0, 3)):
    """Select a simple GARCH(p,q) order by AIC.

    If the optional `arch` dependency is not available, returns (None, None).
    """
    if arch_model is None:
        return None, None

    best_aic = np.inf
    best_order = None
    best_model = None

    for p in p_range:
        for q in q_range:
            try:
                model = arch_model(res, mean='zero', vol=vol, p=p, q=q).fit(disp='off', show_warning=False)
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, q)
                    best_model = model
            except Exception:
                continue

    return best_model, best_order



def make_freq_aware_lagged_exog(X: pd.DataFrame, exo_names, target, m_vars, q_vars, y_vars):
    """Create a real-time exogenous regressor panel by applying publication lags.

    This implements the lag scheme described in the paper (Section 4.2.1):
      - monthly series   : 1 month
      - quarterly series : 4 months
      - annual series    : 6 months

    Notes
    -----
    - `X` is already aligned to the target's periodicity by `load_data()`.
      We implement publication lags by shifting each regressor's *timestamp*
      forward by its release lag (delaying availability), then taking the most
      recently released value as of each target date via forward-fill.
    - This avoids look-ahead even when the target is quarterly/annual, because
      release timestamps need not coincide with the target grid.
    """
    if not exo_names:
        return None

    exo_names = list(exo_names)
    X0 = X[exo_names].copy()
    idx = X0.index

    def _lag_months(var: str) -> int:
        if var in m_vars:
            return 1
        if var in q_vars:
            return 3
        if var in y_vars:
            return 12
        # Fallback: treat as monthly
        return 1

    out = pd.DataFrame(index=idx)
    for c in exo_names:
        lag_m = _lag_months(c)
        s = X0[c]

        # Shift index forward by publication lag to represent release time.
        try:
            s_rel = s.copy()
            s_rel.index = s_rel.index + pd.DateOffset(months=lag_m)
        except Exception:
            # Fallback: period shift (best-effort)
            s_rel = s.shift(lag_m)

        # Align to the target index by taking the last released value available
        # at each target date.
        out[c] = s_rel.reindex(idx, method='ffill')

    return out



def _normalize_stage1_model(model: str | None) -> str:
    """Normalize stage-1 model name.

    We accept a few common aliases for convenience.
    """
    m = (model or 'arimax').strip().lower()
    alias = {
        'arima': 'arimax',
        'arimax_garch': 'arimax',
        'sarimax': 'arimax',
        'ets': 'ets',
        'exp_smoothing': 'ets',
        'exponentialsmoothing': 'ets',
        'holtwinters': 'ets',
        # Some reviewers write "GPR"; some people mistype as "GRF".
        'gpr': 'gpr',
        'grf': 'gpr',
        'gaussian_process': 'gpr',
    }
    return alias.get(m, m)


def _get_target_output_dir(
    save_dir: str,
    start_year: int,
    order_fixed: bool,
    target: str,
    stage1_model: str | None = None,
) -> str:
    """Output directory for a (stage1_model, target).

    Backward compatible: for stage1_model='arimax' we keep the original layout
    save_dir/<root>/<target>/...
    For alternative models we write to
    save_dir/<stage1_model>/<root>/<target>/...
    """
    model = _normalize_stage1_model(stage1_model)
    root = f"{start_year}(fixed order)" if order_fixed else f"{start_year}"
    if model == 'arimax':
        out_dir = os.path.join(save_dir, root, target)
    else:
        out_dir = os.path.join(save_dir, model, root, target)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir




def test_exo_var(
    target: str,
    exo_var: list,
    *,
    df_all: pd.DataFrame | None = None,
    data_dir: str = DEFAULT_DATA_DIR,
    save_dir: str = DEFAULT_SAVE_DIR,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int | None = DEFAULT_END_YEAR,
    initial_train_years: int = DEFAULT_INITIAL_TRAIN_YEARS,
    order_fixed: bool = DEFAULT_ORDER_FIXED,
    sigma_source: str | None = None,
    compute_garch: bool = True,
    garch_order_fixed: bool = False,
    save_outputs: bool = True,
    make_plots: bool = True,
    show_plots: bool = False,
    verbose: bool = True,
):
    """Run Stage-1 ARIMAX with alternative uncertainty proxies.

    Mean
    ----
    - Fit ARIMA/ARIMAX to obtain the one-step-ahead conditional mean forecast.

    Sigma / uncertainty
    -------------------
    - sigma_source='garch' (default): fit GARCH on one-step-ahead residuals and
      use the 1-step-ahead conditional variance as the uncertainty proxy.
    - sigma_source='resid_var': use the expanding residual variance as a constant
      variance benchmark.

    Notes
    -----
    - `compute_garch=False` is mainly for candidate sweeps where only the mean
      equation matters for model selection. In that case, GARCH-based diagnostics
      remain NaN, while resid_var diagnostics are still available.
    - Output filenames keep the legacy `OOS_GARCH_VAR(...)` naming for downstream
      compatibility, even when `sigma_source='resid_var'`.
    """

    sigma_src = _normalize_sigma_source(sigma_source, model='arimax')

    p_range = range(0, 3)
    d_range = range(0, 2)
    q_range = range(0, 3)

    metrics = []
    oos_preds = []
    oos_hist_mean_preds = []
    train_mses = []
    in_sample_r2s = []
    oos_garch_vars = []

    # Load data once if not provided
    if df_all is None:
        df_all = load_data(target, data_dir=data_dir)

    if target not in df_all.columns:
        raise ValueError(f"Target '{target}' not found in aligned dataset columns.")

    y = df_all[target]

    # Ensure output folders exist (even if save_outputs=False we may later write best.csv etc)
    _ = _get_target_output_dir(save_dir, start_year, order_fixed, target)

    if exo_var:
        X = df_all[exo_var]
        X_exog = make_freq_aware_lagged_exog(X, exo_var, target, m_vars, q_vars, y_vars)

        # 타깃과 동일 표본으로 정렬
        y = y.loc[X_exog.dropna(how='all').index]
        X_exog = X_exog.loc[y.index].dropna()

        data = pd.concat([y, X_exog], axis=1).dropna()
        feature_cols = exo_var

    else:
        data = y.dropna().to_frame()
        feature_cols = []

    # period filter
    data = data[data.index.year >= start_year]
    if end_year is not None:
        data = data[data.index.year <= end_year]

    exo_var_str = '_'.join(exo_var) if exo_var else 'None'

    # Guard: need enough history
    if data.empty:
        return {
            'variable': target,
            'model': 'arimax',
            'exo_vars': exo_var_str,
            'best_order(ARIMA)': None,
            'sigma_source': sigma_src,
            'IS_R2': np.nan,
            'IS_MSE': np.nan,
            'OOS_R2': np.nan,
            'OOS_MSE': np.nan,
            **_nan_uncertainty_metric_fields(),
            'error': 'No data after date filtering'
        }

    years = sorted(data.index.year.unique())
    if len(years) <= initial_train_years:
        return {
            'variable': target,
            'model': 'arimax',
            'exo_vars': exo_var_str,
            'best_order(ARIMA)': None,
            'sigma_source': sigma_src,
            'IS_R2': np.nan,
            'IS_MSE': np.nan,
            'OOS_R2': np.nan,
            'OOS_MSE': np.nan,
            **_nan_uncertainty_metric_fields(),
            'error': f'Not enough years for initial_train_years={initial_train_years} (have {len(years)})'
        }

    train_end_year = years[initial_train_years - 1]
    test_dates = data[data.index.year > train_end_year].index

    if len(test_dates) == 0:
        return {
            'variable': target,
            'model': 'arimax',
            'exo_vars': exo_var_str,
            'best_order(ARIMA)': None,
            'sigma_source': sigma_src,
            'IS_R2': np.nan,
            'IS_MSE': np.nan,
            'OOS_R2': np.nan,
            'OOS_MSE': np.nan,
            **_nan_uncertainty_metric_fields(),
            'error': 'No OOS test dates (check sample window)'
        }

    step = 0
    variable_start_time = time.time()
    best_order = None
    garch_order = None

    for dt in tqdm(test_dates, desc=f"{target} ARIMAX OOS [{exo_var_str}]", disable=not verbose):
        step += 1
        train_data = data[data.index < dt]
        y_tr = train_data[target]
        X_tr = train_data[feature_cols] if feature_cols else None
        X_te = data.loc[[dt], feature_cols] if feature_cols else None

        hist_mean_pred = y_tr.mean()
        oos_hist_mean_preds.append(hist_mean_pred)

        y_tr_copy = y_tr.copy()
        X_tr_copy = X_tr.copy() if X_tr is not None else None
        X_te_copy = X_te.copy() if X_te is not None else None

        # 초기 p, d, q 탐색 후 고정
        if order_fixed is True:
            if step == 1:
                model, best_order = select_best_arima_order(y_tr_copy, X_tr_copy, p_range, d_range, q_range)
            else:
                # best_order가 없으면 계속 진행 불가
                if best_order is None:
                    model = None
                else:
                    model = ARIMA(y_tr_copy, exog=X_tr_copy, order=best_order).fit()

        # 1년마다 재탐색
        else:
            if target in m_vars:
                if (step == 1) or (step % 12 == 1):
                    model, best_order = select_best_arima_order(y_tr_copy, X_tr_copy, p_range, d_range, q_range)
                else:
                    model = ARIMA(y_tr_copy, exog=X_tr_copy, order=best_order).fit()

            elif target in q_vars:
                if (step == 1) or (step % 4 == 1):
                    model, best_order = select_best_arima_order(y_tr_copy, X_tr_copy, p_range, d_range, q_range)
                else:
                    model = ARIMA(y_tr_copy, exog=X_tr_copy, order=best_order).fit()

            elif target in y_vars:
                model, best_order = select_best_arima_order(y_tr_copy, X_tr_copy, p_range, d_range, q_range)
                model = ARIMA(y_tr_copy, exog=X_tr_copy, order=best_order).fit()
            else:
                model = None

        if model is None:
            oos_preds.append(np.nan)
            train_mses.append(np.nan)
            in_sample_r2s.append(np.nan)
            oos_garch_vars.append(np.nan)
            continue

        pred = model.forecast(steps=1, exog=X_te_copy)[0]
        oos_preds.append(pred)

        in_sample_pred = model.fittedvalues
        y_tr_aligned = y_tr.loc[in_sample_pred.index]
        train_mses.append(mean_squared_error(y_tr_aligned, in_sample_pred))

        # Sigma / uncertainty proxy
        resid = (y_tr_aligned - in_sample_pred)

        if sigma_src == 'garch':
            if compute_garch and (arch_model is not None) and (len(resid) >= 20):
                try:
                    if (not garch_order_fixed) or (garch_order is None):
                        garch_fitted, garch_order = select_best_garch_order(resid, vol='GARCH')
                    else:
                        p, q = garch_order
                        garch_fitted = arch_model(resid, mean='zero', vol='GARCH', p=p, q=q).fit(disp='off', show_warning=False)

                    if garch_fitted is None:
                        oos_garch_vars.append(np.nan)
                    else:
                        garch_forecast = garch_fitted.forecast(horizon=1)
                        oos_var = float(garch_forecast.variance.iloc[-1, 0])
                        oos_garch_vars.append(oos_var)
                except Exception as e:
                    if verbose:
                        print(f"GARCH Fitting Error at {dt}: {e}")
                    oos_garch_vars.append(np.nan)
            else:
                oos_garch_vars.append(np.nan)
        else:
            oos_garch_vars.append(float(resid.var()) if len(resid) else np.nan)

        # Window-level in-sample R2 diagnostics (kept as in original)
        if target in level_vars:
            _, is_r2_window = r2_level("ARIMAX_Window", y_tr_aligned, in_sample_pred, data_type='IS_Window', verbose=False)
        else:
            _, is_r2_window = r2_return("ARIMAX_Window", y_tr_aligned, in_sample_pred, data_type='IS_Window', verbose=False)
        in_sample_r2s.append(is_r2_window)

    # If best_order never set, we cannot compute final IS metrics reliably
    if best_order is None:
        return {
            'variable': target,
            'model': 'arimax',
            'exo_vars': exo_var_str,
            'best_order(ARIMA)': None,
            'sigma_source': sigma_src,
            'IS_R2': np.nan,
            'IS_MSE': float(np.nanmean(train_mses)) if len(train_mses) else np.nan,
            'OOS_R2': np.nan,
            'OOS_MSE': np.nan,
            **_nan_uncertainty_metric_fields(),
            'error': 'ARIMA order selection failed (best_order=None)'
        }

    # IS prediction (full training up to last OOS point)
    last_dt = test_dates[-1]

    train_data_full = data[data.index < last_dt]
    y_tr_full = train_data_full[target]
    X_tr_full = train_data_full[feature_cols] if feature_cols else None

    y_tr_full_copy = y_tr_full.copy()
    X_tr_full_copy = X_tr_full.copy() if X_tr_full is not None else None

    try:
        final_model = ARIMA(y_tr_full_copy, exog=X_tr_full_copy, order=best_order).fit()
        is_pred_full = final_model.fittedvalues

        d = best_order[1]
        is_pred = is_pred_full.iloc[d:]
        is_actuals = y_tr_full.loc[is_pred.index]
    except Exception as e:
        if verbose:
            print(f"Final ARIMA fit failed for target={target}, exo={exo_var_str}: {e}")
        is_pred = pd.Series(dtype=float)
        is_actuals = pd.Series(dtype=float)

    # In-sample sigma proxy (for reporting only)
    is_garch_var = np.nan
    if len(is_actuals) > 0:
        resid_full = (is_actuals - is_pred).dropna()
        if sigma_src == 'garch':
            if compute_garch and (arch_model is not None) and (len(resid_full) >= 20):
                try:
                    if (not garch_order_fixed) or (garch_order is None):
                        garch_full_fitted, garch_order = select_best_garch_order(resid_full, vol='GARCH')
                    else:
                        p, q = garch_order
                        garch_full_fitted = arch_model(resid_full, mean='zero', vol='GARCH', p=p, q=q).fit(disp='off', show_warning=False)
                    if garch_full_fitted is not None:
                        is_garch_var = float(np.mean(garch_full_fitted.conditional_volatility ** 2))
                except Exception as e:
                    if verbose:
                        print(f"GARCH IS Fitting Error: {e}")
        else:
            is_garch_var = float(resid_full.var()) if len(resid_full) else np.nan

    variable_end_time = time.time()
    if verbose:
        print(f"'{target}' 변수 예측에 소요된 시간: {variable_end_time - variable_start_time:.2f}초")

    oos_series = pd.Series(oos_preds, index=test_dates, name='arimax_pred')
    oos_actuals = data[target].loc[test_dates]

    oos_hist_mean_series = pd.Series(oos_hist_mean_preds, index=test_dates, name='hist_mean_pred')

    avg_train_mse = np.nanmean(train_mses) if len(train_mses) else np.nan
    is_garch_std = np.sqrt(is_garch_var) if np.isfinite(is_garch_var) else np.nan
    oos_var_series = pd.Series(oos_garch_vars, index=test_dates, name='oos_var')
    uncertainty_metrics = _compute_uncertainty_diagnostics(
        oos_actuals,
        oos_series,
        oos_var_series,
        is_std=is_garch_std,
    )

    # Compute IS/OOS metrics (keep original benchmark rules)
    if target in vars_level_rw1:
        is_mse, is_r2 = r2_level("ARIMAX", is_actuals, is_pred, data_type='IS', verbose=verbose)
        oos_mse, oos_r2 = r2_level("ARIMAX", oos_actuals, oos_series, data_type='OOS', verbose=verbose)

    elif target in vars_return_ao:
        is_mse, is_r2 = r2_return("ARIMAX", is_actuals, is_pred, data_type='IS', verbose=verbose)
        oos_mse, oos_r2 = r2_return_oos_ao("ARIMAX", oos_actuals, oos_series, data_type='OOS', verbose=verbose)

    elif target in vars_return_gw:
        is_mse, is_r2 = r2_return("ARIMAX", is_actuals, is_pred, data_type='IS', verbose=verbose)
        oos_mse, oos_r2 = r2_return_oos("ARIMAX", oos_actuals, oos_series, oos_hist_mean_series, data_type='OOS', verbose=verbose)

    else:
        is_mse, is_r2, oos_mse, oos_r2 = np.nan, np.nan, np.nan, np.nan

    metrics_row = {
        'variable': target,
        'model': 'arimax',
        'exo_vars': exo_var_str,
        'best_order(ARIMA)': best_order,
        'sigma_source': sigma_src,
        'IS_R2': is_r2,
        'IS_MSE': avg_train_mse,
        'OOS_R2': oos_r2,
        'OOS_MSE': oos_mse,
        **uncertainty_metrics,
    }

    # Save outputs (optional)
    if save_outputs:
        out_dir = _get_target_output_dir(save_dir, start_year, order_fixed, target)

        pd.DataFrame({'actual': oos_actuals, 'pred': oos_preds}).to_csv(os.path.join(out_dir, f"OOS({exo_var_str}).csv"))
        pd.DataFrame({'actual': is_actuals, 'pred': is_pred}).to_csv(os.path.join(out_dir, f"IS({exo_var_str}).csv"))

        # Keep legacy filename for downstream compatibility, regardless of sigma_source.
        oos_var_series.to_csv(os.path.join(out_dir, f"OOS_GARCH_VAR({exo_var_str}).csv"))

        metrics_df = pd.DataFrame([metrics_row])
        metrics_df.to_csv(os.path.join(out_dir, f"metrics({exo_var_str}).csv"), index=False)

        if make_plots:
            plt.figure(figsize=(12, 4))
            plt.plot(oos_actuals, label='Real')
            plt.plot(oos_series, label='Pred')
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(out_dir, f"plot({exo_var_str}).png"), dpi=100)
            if show_plots:
                plt.show()
            plt.close()

    return metrics_row


# ------------------------------------------------------------
# Alternative Stage-1 models (robustness checks)
# ------------------------------------------------------------

def _freq_steps_per_year(target: str) -> int:
    """Return the approximate number of observations per year for target."""
    if target in m_vars:
        return 12
    if target in q_vars:
        return 4
    return 1


def _slice_tail(s: pd.Series, max_len: int | None) -> pd.Series:
    if (max_len is None) or (max_len <= 0):
        return s
    if len(s) <= max_len:
        return s
    return s.iloc[-max_len:]


def _safe_series(x) -> pd.Series:
    """Convert 1-step forecast outputs to a float, robustly."""
    if isinstance(x, (pd.Series, pd.Index)):
        return pd.Series(x)
    return pd.Series(x)


def _safe_corr(x, y) -> float:
    """Robust correlation helper returning NaN when ill-defined."""
    try:
        a = pd.Series(x, dtype=float)
        b = pd.Series(y, dtype=float)
        ab = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ab) < 2:
            return np.nan
        x0 = ab.iloc[:, 0]
        y0 = ab.iloc[:, 1]
        if (x0.nunique(dropna=True) < 2) or (y0.nunique(dropna=True) < 2):
            return np.nan
        return float(x0.corr(y0))
    except Exception:
        return np.nan


def _nan_uncertainty_metric_fields(is_std: float = np.nan) -> dict:
    """NaN-filled uncertainty/calibration metric block."""
    return {
        'IS_STD': is_std,
        'OOS_STD': np.nan,
        'OOS_RMSE': np.nan,
        'STD / RMSE': np.nan,
        'ABS_STD_ERR': np.nan,
        'QLIKE': np.nan,
        'CORR_SIGMA_ABSERR': np.nan,
        'CORR_VAR_SQERR': np.nan,
        # Backward-compatible aliases
        'IS_GARCH_STD': is_std,
        'OOS_GARCH_STD': np.nan,
    }


def _compute_uncertainty_diagnostics(
    y_true,
    y_pred,
    oos_var_series,
    *,
    is_std: float = np.nan,
) -> dict:
    """Compute scale-calibration and variance-forecast diagnostics.

    Definitions
    -----------
    e_t = y_t - ŷ_t
    OOS_STD = sqrt(mean(v̂_t))
    OOS_RMSE = sqrt(mean(e_t^2))
    QLIKE = mean(log(v̂_t) + e_t^2 / v̂_t)

    Notes
    -----
    - `OOS_MSE` elsewhere in the code follows the paper's benchmark-specific
      evaluation conventions (e.g., level variables drop the first observation).
      The diagnostics here intentionally use the raw 1-step forecast errors.
    - For numerical stability, QLIKE clips variances below machine epsilon.
    """
    out = _nan_uncertainty_metric_fields(is_std=is_std)

    try:
        y_true_s = pd.Series(y_true, dtype=float)
        y_pred_s = pd.Series(y_pred, dtype=float)
        var_s = pd.Series(oos_var_series, dtype=float)

        df = pd.concat(
            [
                y_true_s.rename('actual'),
                y_pred_s.rename('pred'),
                var_s.rename('var'),
            ],
            axis=1,
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if df.empty:
            return out

        err = df['actual'] - df['pred']
        sqerr = err.pow(2)
        var_pos = df['var'].clip(lower=np.finfo(float).eps)
        sigma = np.sqrt(var_pos)

        oos_std = float(np.sqrt(var_pos.mean())) if len(var_pos) else np.nan
        oos_rmse = float(np.sqrt(sqerr.mean())) if len(sqerr) else np.nan
        std_ratio = (oos_std / oos_rmse) if (np.isfinite(oos_std) and np.isfinite(oos_rmse) and oos_rmse > 0) else np.nan
        qlike = float(np.mean(np.log(var_pos) + (sqerr / var_pos))) if len(var_pos) else np.nan

        out.update({
            'OOS_STD': oos_std,
            'OOS_RMSE': oos_rmse,
            'STD / RMSE': std_ratio,
            'ABS_STD_ERR': abs(std_ratio - 1.0) if np.isfinite(std_ratio) else np.nan,
            'QLIKE': qlike,
            'CORR_SIGMA_ABSERR': _safe_corr(sigma, np.abs(err)),
            'CORR_VAR_SQERR': _safe_corr(var_pos, sqerr),
            'OOS_GARCH_STD': oos_std,  # backward-compatible alias
        })
        return out
    except Exception:
        return out

# -------------------------
# Helper: sigma/volatility source
# -------------------------

def _normalize_sigma_source(s: str | None, model: str) -> str:
    """Normalize sigma/volatility source strings.

    Supported:
      - 'garch'     : fit GARCH on one-step-ahead residuals (if arch is available)
      - 'native'    : use model-native predictive variance (GPR only)
      - 'resid_var' : use (expanding) residual variance as a simple proxy

    Notes
    -----
    - For ETS, the Holt-Winters API in statsmodels does not expose a reliable
      one-step predictive variance, so 'native' is mapped to 'resid_var'.
    - For ARIMAX, the meaningful options are 'garch' and 'resid_var'.
    """
    m = (model or '').strip().lower()
    if s is None:
        return 'native' if m == 'gpr' else 'garch'

    s0 = (s or '').strip().lower()
    aliases = {
        'arch': 'garch',
        'garch': 'garch',
        'native': 'native',
        'pred': 'native',
        'predictive': 'native',
        'model': 'native',
        'resid': 'resid_var',
        'residual': 'resid_var',
        'resid_var': 'resid_var',
        'rv': 'resid_var',
    }
    s1 = aliases.get(s0, s0)

    if m == 'ets' and s1 == 'native':
        return 'resid_var'
    if m == 'arimax' and s1 == 'native':
        raise ValueError("ARIMAX supports sigma_source='garch' or 'resid_var' (not 'native').")

    if s1 not in {'garch', 'native', 'resid_var'}:
        raise ValueError(f"Unknown sigma_source='{s}'. Use one of: garch|native|resid_var")

    return s1


# -------------------------
# Helper: ETS candidate configs (trend/seasonal/damped)
# -------------------------

def _ets_seasonal_periods(target: str) -> int | None:
    if target in m_vars:
        return 12
    if target in q_vars:
        return 4
    return None


def _default_ets_candidates(target: str, *, allow_seasonal: bool = True) -> list[tuple[str, dict]]:
    """A small, defensible ETS candidate set.

    We keep it additive to avoid sign restrictions (macro series can be negative).

    Returns:
        List of (name, config_dict)
    """
    sp = _ets_seasonal_periods(target)

    cands: list[tuple[str, dict]] = [
        ('ANN', dict(trend=None, damped_trend=False, seasonal=None, seasonal_periods=None)),
        ('AAN', dict(trend='add', damped_trend=False, seasonal=None, seasonal_periods=None)),
        ('AAdN', dict(trend='add', damped_trend=True, seasonal=None, seasonal_periods=None)),
    ]

    # Additive seasonality only when frequency supports it.
    if allow_seasonal and (sp is not None) and (sp >= 2):
        cands += [
            ('ANA', dict(trend=None, damped_trend=False, seasonal='add', seasonal_periods=sp)),
            ('AAA', dict(trend='add', damped_trend=False, seasonal='add', seasonal_periods=sp)),
            ('AAdA', dict(trend='add', damped_trend=True, seasonal='add', seasonal_periods=sp)),
        ]

    return cands


def _select_best_ets_config(
    y: pd.Series,
    target: str,
    *,
    candidates: list[tuple[str, dict]] | None = None,
    allow_seasonal: bool = True,
    criterion: str = 'aicc',
) -> tuple[str, dict]:
    """Select ETS config on the current training sample by information criterion."""
    if candidates is None:
        candidates = _default_ets_candidates(target, allow_seasonal=allow_seasonal)

    crit = (criterion or 'aicc').strip().lower()
    if crit not in {'aicc', 'aic', 'bic'}:
        raise ValueError("ETS criterion must be one of: aicc|aic|bic")

    best_name: str | None = None
    best_cfg: dict | None = None
    best_val = np.inf

    for name, cfg in candidates:
        try:
            # Seasonal candidates require enough history; statsmodels will raise if not.
            mdl = ExponentialSmoothing(
                y,
                trend=cfg.get('trend', None),
                damped_trend=cfg.get('damped_trend', False),
                seasonal=cfg.get('seasonal', None),
                seasonal_periods=cfg.get('seasonal_periods', None),
                initialization_method='estimated',
            )
            res = mdl.fit(optimized=True)

            if crit == 'aicc':
                val = float(getattr(res, 'aicc'))
            elif crit == 'aic':
                val = float(getattr(res, 'aic'))
            else:
                val = float(getattr(res, 'bic'))

            if np.isfinite(val) and (val < best_val):
                best_val = val
                best_name = name
                best_cfg = cfg
        except Exception:
            continue

    if best_name is None or best_cfg is None:
        # Fallback to simplest configuration
        return 'ANN', dict(trend=None, damped_trend=False, seasonal=None, seasonal_periods=None)

    return best_name, best_cfg


# -------------------------
# Helper: GPR kernels (candidates) + selection
# -------------------------

def _build_gpr_kernel(kernel_name: str, *, seasonal_period: float | None = None):
    """Build a sklearn GP kernel by name.

    Names:
      - rbf
      - matern32
      - matern52
      - periodic (uses ExpSineSquared; requires a time index feature to be meaningful)
      - rbf+periodic
    """
    name = (kernel_name or 'rbf').strip().lower()

    if name in {'rbf', 'se', 'squaredexp', 'squared_exponential'}:
        base = RBF(length_scale=1.0)
    elif name in {'matern32', 'matern_32', 'matern1.5', 'matern15'}:
        base = Matern(length_scale=1.0, nu=1.5)
    elif name in {'matern52', 'matern_52', 'matern2.5', 'matern25'}:
        base = Matern(length_scale=1.0, nu=2.5)
    elif name in {'periodic', 'exp_sine', 'expsinesquared'}:
        per = float(seasonal_period) if seasonal_period is not None else 12.0
        base = ExpSineSquared(length_scale=1.0, periodicity=per)
    elif name in {'rbf+periodic', 'periodic+rbf'}:
        per = float(seasonal_period) if seasonal_period is not None else 12.0
        base = RBF(length_scale=1.0) + ExpSineSquared(length_scale=1.0, periodicity=per)
    else:
        raise ValueError(f"Unknown gpr_kernel='{kernel_name}'.")

    # Wrap with amplitude + white noise.
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * base + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))
    return kernel


def _select_best_gpr_kernel(
    X_train_std: np.ndarray,
    y_train: np.ndarray,
    *,
    kernel_candidates: list[str],
    alpha: float,
    optimizer: str | None,
    selection: str = 'lml',
    seasonal_period: float | None = None,
) -> str:
    """Pick a kernel using the current training sample.

    By default we maximize log-marginal-likelihood ('lml'), which is standard for GP.
    """
    sel = (selection or 'lml').strip().lower()
    if sel not in {'lml', 'mse'}:
        raise ValueError("gpr_kernel_select must be one of: lml|mse")

    best_name: str | None = None
    best_score = -np.inf

    for kname in kernel_candidates:
        try:
            kernel = _build_gpr_kernel(kname, seasonal_period=seasonal_period)
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                optimizer=optimizer,
                normalize_y=True,
            )
            gpr.fit(X_train_std, y_train)

            if sel == 'lml':
                score = float(getattr(gpr, 'log_marginal_likelihood_value_', np.nan))
            else:
                y_fit = gpr.predict(X_train_std)
                score = -float(mean_squared_error(y_train, y_fit))  # higher is better

            if np.isfinite(score) and (best_name is None or score > best_score):
                best_score = score
                best_name = kname
        except Exception:
            continue

    return best_name or kernel_candidates[0]

def test_ets_var(
    target: str,
    exo_var: list,
    *,
    df_all: pd.DataFrame | None = None,
    data_dir: str = DEFAULT_DATA_DIR,
    save_dir: str = DEFAULT_SAVE_DIR,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int | None = DEFAULT_END_YEAR,
    initial_train_years: int = DEFAULT_INITIAL_TRAIN_YEARS,
    order_fixed: bool = DEFAULT_ORDER_FIXED,
    # ETS config selection (trend/seasonal/damped)
    ets_candidates: list[tuple[str, dict]] | None = None,
    ets_allow_seasonal: bool = True,
    ets_select_criterion: str = 'aicc',
    # Volatility / uncertainty proxy
    sigma_source: str | None = None,
    compute_garch: bool = True,
    garch_order_fixed: bool = False,
    # Artifacts
    save_outputs: bool = True,
    make_plots: bool = True,
    show_plots: bool = False,
    verbose: bool = True,
):
    """Stage-1 forecasting with ETS (exponential smoothing).

    IMPORTANT
    ---------
    For the major-revision robustness check, ETS is treated as a **univariate**
    alternative by design (no exogenous regressors). This keeps the comparison
    simple and avoids mixing a static OLS layer with ETS dynamics.

    What this implements
    --------------------
    Mean
      - Fit ETS directly on the target series y.

    Volatility / sigma (GARCH-like part)
      - `sigma_source='garch'` (default for ETS): fit GARCH( p, q ) to the one-step-ahead residuals
        and use the 1-step-ahead conditional variance as sigma^2.
      - `sigma_source='resid_var'`: use residual variance as a lightweight proxy.

    Notes
    -----
    - The Holt-Winters ETS API in statsmodels does not expose a robust predictive variance;
      therefore `sigma_source='native'` is mapped to `resid_var`.
    - This function emits an "OOS_GARCH_VAR" file for compatibility with downstream Stage-2,
      even when the variance is a proxy (resid_var).
    """

    sigma_src = _normalize_sigma_source(sigma_source, model='ets')

    # ETS robustness: univariate by design.
    # We keep `exo_var` in the signature for backward compatibility, but ignore it.
    if exo_var:
        if verbose:
            print(f"[ETS] Ignoring exogenous regressors (univariate ETS). exo_var={exo_var}")
        exo_var = []

    # Load data once if not provided
    if df_all is None:
        df_all = load_data(target, data_dir=data_dir)

    if target not in df_all.columns:
        raise ValueError(f"Target '{target}' not found in aligned dataset columns.")

    y = df_all[target]

    # Build dataset (univariate)
    data = y.dropna().to_frame(name=target)
    feature_cols: list[str] = []

    # Period filter
    data = data[data.index.year >= start_year]
    if end_year is not None:
        data = data[data.index.year <= end_year]

    exo_var_str = '_'.join(exo_var) if exo_var else 'None'

    # Ensure output folder exists
    _ = _get_target_output_dir(save_dir, start_year, order_fixed, target, stage1_model='ets')

    # Guards
    if data.empty:
        return {
            'variable': target,
            'model': 'ets',
            'exo_vars': exo_var_str,
            'ets_config': None,
            'sigma_source': sigma_src,
            'IS_R2': np.nan,
            'IS_MSE': np.nan,
            'OOS_R2': np.nan,
            'OOS_MSE': np.nan,
            'IS_GARCH_STD': np.nan,
            'OOS_GARCH_STD': np.nan,
            'STD / RMSE': np.nan,
            'error': 'No data after date filtering'
        }

    years = sorted(data.index.year.unique())
    if len(years) <= initial_train_years:
        return {
            'variable': target,
            'model': 'ets',
            'exo_vars': exo_var_str,
            'ets_config': None,
            'sigma_source': sigma_src,
            'IS_R2': np.nan,
            'IS_MSE': np.nan,
            'OOS_R2': np.nan,
            'OOS_MSE': np.nan,
            'IS_GARCH_STD': np.nan,
            'OOS_GARCH_STD': np.nan,
            'STD / RMSE': np.nan,
            'error': f'Not enough years for initial_train_years={initial_train_years} (have {len(years)})'
        }

    train_end_year = years[initial_train_years - 1]
    test_dates = data[data.index.year > train_end_year].index

    if len(test_dates) == 0:
        return {
            'variable': target,
            'model': 'ets',
            'exo_vars': exo_var_str,
            'ets_config': None,
            'sigma_source': sigma_src,
            'IS_R2': np.nan,
            'IS_MSE': np.nan,
            'OOS_R2': np.nan,
            'OOS_MSE': np.nan,
            'IS_GARCH_STD': np.nan,
            'OOS_GARCH_STD': np.nan,
            'STD / RMSE': np.nan,
            'error': 'No OOS test dates (check sample window)'
        }

    # Candidate set
    if ets_candidates is None:
        ets_candidates = _default_ets_candidates(target, allow_seasonal=ets_allow_seasonal)

    # Accumulators
    oos_preds: list[float] = []
    oos_hist_mean_preds: list[float] = []
    oos_vars: list[float] = []
    train_mses: list[float] = []
    in_sample_r2s: list[float] = []

    best_ets_name: str | None = None
    best_ets_cfg: dict | None = None

    # Optional: reuse a fixed GARCH order (saves a lot of time)
    garch_order: tuple[int, int] | None = None

    step = 0
    variable_start_time = time.time()

    steps_per_year = _freq_steps_per_year(target)

    def _need_reselect(step_: int) -> bool:
        if order_fixed:
            return step_ == 1
        # reselect once per year boundary (frequency-aware)
        if steps_per_year <= 1:
            return True
        return (step_ == 1) or (step_ % steps_per_year == 1)

    for dt in tqdm(test_dates, desc=f"{target} ETS OOS [{exo_var_str}]", disable=not verbose):
        step += 1

        train_data = data[data.index < dt]
        y_tr = train_data[target]
        X_tr = train_data[feature_cols] if feature_cols else None
        X_te = data.loc[[dt], feature_cols] if feature_cols else None

        # Historical mean benchmark (needed for some R2 definitions)
        oos_hist_mean_preds.append(float(y_tr.mean()))

        # 1) Regression part (optional)
        if X_tr is not None and (not X_tr.empty):
            try:
                X_tr_c = sm.add_constant(X_tr, has_constant='add')
                ols = sm.OLS(y_tr, X_tr_c).fit()
                reg_fit = ols.fittedvalues
                resid_tr = (y_tr - reg_fit)

                X_te_c = sm.add_constant(X_te, has_constant='add')
                reg_forecast = float(ols.predict(X_te_c).iloc[0])
            except Exception:
                # Fallback: ignore exog if regression fails
                reg_fit = pd.Series(0.0, index=y_tr.index)
                resid_tr = y_tr.copy()
                reg_forecast = 0.0
        else:
            reg_fit = pd.Series(0.0, index=y_tr.index)
            resid_tr = y_tr.copy()
            reg_forecast = 0.0

        # 2) ETS config selection on the residual component
        if (best_ets_cfg is None) or _need_reselect(step):
            best_ets_name, best_ets_cfg = _select_best_ets_config(
                resid_tr,
                target,
                candidates=ets_candidates,
                allow_seasonal=ets_allow_seasonal,
                criterion=ets_select_criterion,
            )

        # 3) ETS fit + 1-step forecast
        try:
            cfg = best_ets_cfg or {}
            ets_model = ExponentialSmoothing(
                resid_tr,
                trend=cfg.get('trend', None),
                damped_trend=cfg.get('damped_trend', False),
                seasonal=cfg.get('seasonal', None),
                seasonal_periods=cfg.get('seasonal_periods', None),
                initialization_method='estimated',
            )
            ets_fit = ets_model.fit(optimized=True)

            resid_fc = ets_fit.forecast(1)
            resid_pred = float(resid_fc.iloc[0] if hasattr(resid_fc, 'iloc') else resid_fc[0])

            resid_is_fit = ets_fit.fittedvalues
        except Exception:
            # If ETS fails at this dt, emit NaNs and continue
            oos_preds.append(np.nan)
            train_mses.append(np.nan)
            in_sample_r2s.append(np.nan)
            oos_vars.append(np.nan)
            continue

        y_pred = reg_forecast + resid_pred
        oos_preds.append(float(y_pred))

        # In-sample prediction for diagnostics + sigma estimation
        resid_is_fit = resid_is_fit.reindex(resid_tr.index)
        in_sample_pred = (reg_fit + resid_is_fit).reindex(y_tr.index)

        y_tr_aligned = y_tr.loc[in_sample_pred.index]
        train_mses.append(float(mean_squared_error(y_tr_aligned, in_sample_pred)))

        # 4) Sigma / variance proxy
        resid_total = (y_tr_aligned - in_sample_pred).dropna()

        var = np.nan
        if (sigma_src == 'garch') and compute_garch and (arch_model is not None) and (len(resid_total) >= 20):
            try:
                if (not garch_order_fixed) or (garch_order is None):
                    garch_fit, garch_order = select_best_garch_order(resid_total, vol='GARCH')
                else:
                    p, q = garch_order
                    garch_fit = arch_model(resid_total, mean='zero', vol='GARCH', p=p, q=q).fit(disp='off', show_warning=False)

                garch_fc = garch_fit.forecast(horizon=1)
                var = float(garch_fc.variance.iloc[-1, 0])
            except Exception:
                var = float(resid_total.var())
        else:
            # resid_var proxy (also used as fallback)
            var = float(resid_total.var()) if len(resid_total) else np.nan

        oos_vars.append(var)

        # Window-level in-sample R2 diagnostic
        if target in level_vars:
            _, is_r2_window = r2_level("ETS_Window", y_tr_aligned, in_sample_pred, data_type='IS_Window', verbose=False)
        else:
            _, is_r2_window = r2_return("ETS_Window", y_tr_aligned, in_sample_pred, data_type='IS_Window', verbose=False)
        in_sample_r2s.append(is_r2_window)

    variable_end_time = time.time()
    if verbose:
        print(f"'{target}' ETS forecast time: {variable_end_time - variable_start_time:.2f}s")

    # Build OOS series
    oos_series = pd.Series(oos_preds, index=test_dates, name='ets_pred')
    oos_actuals = data[target].loc[test_dates]

    oos_hist_mean_series = pd.Series(oos_hist_mean_preds, index=test_dates, name='hist_mean_pred')

    # IS fit on the final training window
    last_dt = test_dates[-1]
    train_data_full = data[data.index < last_dt]
    y_tr_full = train_data_full[target]
    X_tr_full = train_data_full[feature_cols] if feature_cols else None

    # Regression (optional)
    if X_tr_full is not None and (not X_tr_full.empty):
        try:
            X_tr_full_c = sm.add_constant(X_tr_full, has_constant='add')
            ols_full = sm.OLS(y_tr_full, X_tr_full_c).fit()
            reg_fit_full = ols_full.fittedvalues
            resid_full = (y_tr_full - reg_fit_full)
        except Exception:
            reg_fit_full = pd.Series(0.0, index=y_tr_full.index)
            resid_full = y_tr_full.copy()
    else:
        reg_fit_full = pd.Series(0.0, index=y_tr_full.index)
        resid_full = y_tr_full.copy()

    # Ensure we have a final ETS config
    if best_ets_cfg is None:
        best_ets_name, best_ets_cfg = _select_best_ets_config(
            resid_full,
            target,
            candidates=ets_candidates,
            allow_seasonal=ets_allow_seasonal,
            criterion=ets_select_criterion,
        )

    is_pred = pd.Series(dtype=float)
    is_actuals = pd.Series(dtype=float)
    try:
        cfg = best_ets_cfg or {}
        ets_model_full = ExponentialSmoothing(
            resid_full,
            trend=cfg.get('trend', None),
            damped_trend=cfg.get('damped_trend', False),
            seasonal=cfg.get('seasonal', None),
            seasonal_periods=cfg.get('seasonal_periods', None),
            initialization_method='estimated',
        )
        ets_fit_full = ets_model_full.fit(optimized=True)
        resid_is_fit_full = ets_fit_full.fittedvalues.reindex(resid_full.index)

        is_pred = (reg_fit_full + resid_is_fit_full).reindex(y_tr_full.index)
        is_actuals = y_tr_full.loc[is_pred.index]
    except Exception:
        pass

    # IS sigma proxy
    is_std = np.nan
    if len(is_actuals) and len(is_pred):
        resid_is_total = (is_actuals - is_pred).dropna()
        if (sigma_src == 'garch') and compute_garch and (arch_model is not None) and (len(resid_is_total) >= 20):
            try:
                if (not garch_order_fixed) or (garch_order is None):
                    garch_fit_is, garch_order = select_best_garch_order(resid_is_total, vol='GARCH')
                else:
                    p, q = garch_order
                    garch_fit_is = arch_model(resid_is_total, mean='zero', vol='GARCH', p=p, q=q).fit(disp='off', show_warning=False)
                is_var = float(np.mean(garch_fit_is.conditional_volatility ** 2))
                is_std = float(np.sqrt(is_var))
            except Exception:
                is_std = float(np.sqrt(resid_is_total.var()))
        else:
            is_std = float(np.sqrt(resid_is_total.var()))

    # Aggregate
    avg_train_mse = float(np.nanmean(train_mses)) if len(train_mses) else np.nan

    oos_var_mean = float(np.nanmean(oos_vars)) if len(oos_vars) else np.nan
    oos_std = float(np.sqrt(oos_var_mean)) if np.isfinite(oos_var_mean) else np.nan

    # Compute metrics with the original benchmark rules
    if target in vars_level_rw1:
        is_mse, is_r2 = r2_level('ETS', is_actuals, is_pred, data_type='IS', verbose=verbose)
        oos_mse, oos_r2 = r2_level('ETS', oos_actuals, oos_series, data_type='OOS', verbose=verbose)

    elif target in vars_return_ao:
        is_mse, is_r2 = r2_return("ARIMAX", is_actuals, is_pred, data_type='IS', verbose=verbose)
        oos_mse, oos_r2 = r2_return_oos_ao("ARIMAX", oos_actuals, oos_series, data_type='OOS', verbose=verbose)

    elif target in vars_return_gw:
        is_mse, is_r2 = r2_return('ETS', is_actuals, is_pred, data_type='IS', verbose=verbose)
        oos_mse, oos_r2 = r2_return_oos('ETS', oos_actuals, oos_series, oos_hist_mean_series, data_type='OOS', verbose=verbose)

    else:
        is_mse, is_r2, oos_mse, oos_r2 = np.nan, np.nan, np.nan, np.nan

    oos_var_series = pd.Series(oos_vars, index=test_dates, name='oos_var')
    uncertainty_metrics = _compute_uncertainty_diagnostics(
        oos_actuals,
        oos_series,
        oos_var_series,
        is_std=is_std,
    )

    metrics_row = {
        'variable': target,
        'model': 'ets',
        'exo_vars': exo_var_str,
        'ets_config': best_ets_name,
        'sigma_source': sigma_src,
        'IS_R2': is_r2,
        'IS_MSE': avg_train_mse,
        'OOS_R2': oos_r2,
        'OOS_MSE': oos_mse,
        **uncertainty_metrics,
    }

    # Save outputs (optional)
    if save_outputs:
        out_dir = _get_target_output_dir(save_dir, start_year, order_fixed, target, stage1_model='ets')

        pd.DataFrame({'actual': oos_actuals, 'pred': oos_series}).to_csv(os.path.join(out_dir, f"OOS({exo_var_str}).csv"))
        pd.DataFrame({'actual': is_actuals, 'pred': is_pred}).to_csv(os.path.join(out_dir, f"IS({exo_var_str}).csv"))

        # Always emit variance proxy (for downstream compatibility)
        oos_var_series.to_csv(os.path.join(out_dir, f"OOS_GARCH_VAR({exo_var_str}).csv"))

        pd.DataFrame([metrics_row]).to_csv(os.path.join(out_dir, f"metrics({exo_var_str}).csv"), index=False)

        if make_plots:
            plt.figure(figsize=(12, 4))
            plt.plot(oos_actuals, label='Real')
            plt.plot(oos_series, label='Pred')
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(out_dir, f"plot({exo_var_str}).png"), dpi=100)
            if show_plots:
                plt.show()
            plt.close()

    return metrics_row


def test_gpr_var(
    target: str,
    exo_var: list,
    *,
    df_all: pd.DataFrame | None = None,
    data_dir: str = DEFAULT_DATA_DIR,
    save_dir: str = DEFAULT_SAVE_DIR,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int | None = DEFAULT_END_YEAR,
    initial_train_years: int = DEFAULT_INITIAL_TRAIN_YEARS,
    order_fixed: bool = DEFAULT_ORDER_FIXED,
    # GPR knobs
    gpr_max_train_size: int | None = None,
    gpr_ar_lags: int = 2,
    gpr_target_transform: str = 'auto',
    gpr_add_time_index: bool = False,
    gpr_alpha: float = 1e-2,
    gpr_optimizer: str | None = None,
    # Kernel selection
    gpr_kernel: str = 'matern52',
    gpr_kernel_candidates: list[str] | None = None,
    gpr_kernel_select: str = 'lml',
    # Volatility / sigma source
    sigma_source: str | None = None,
    compute_garch: bool = False,
    garch_order_fixed: bool = False,
    # Artifacts
    save_outputs: bool = True,
    make_plots: bool = True,
    show_plots: bool = False,
    verbose: bool = True,
):
    """Stage-1 forecasting with Gaussian Process Regression.

    Mean
    ----
    We construct a supervised 1-step-ahead dataset:
        y_t  ~  f( y_{t-1}, y_{t-2}, ..., y_{t-L}, x_t, [t] )
    where x_t is publication-lag-adjusted via make_freq_aware_lagged_exog, and
    [t] is an optional time-index feature.

    Sigma / uncertainty
    -------------------
    - sigma_source='native' (default for GPR): use the GP predictive variance
      returned by sklearn (return_std=True).
    - sigma_source='garch': fit GARCH on one-step residuals of the GP mean
      (more comparable to ARIMAX+GARCH). Requires `arch`.
    - sigma_source='resid_var': use residual variance as a simple proxy.

    Notes
    -----
    - Training is capped by `gpr_max_train_size` for speed (default: initial window length).
    - `gpr_ar_lags` controls how many autoregressive lags (L) are included.
    - Kernel selection can be done by providing `gpr_kernel_candidates` (selected on the
      first training window by log-marginal-likelihood or train MSE).
    """

    sigma_src = _normalize_sigma_source(sigma_source, model='gpr')

    # Defensive: at least 1 lag.
    try:
        gpr_ar_lags = int(gpr_ar_lags)
    except Exception:
        gpr_ar_lags = 1
    if gpr_ar_lags < 1:
        gpr_ar_lags = 1

    # Target transform (A-plan)
    # - For level variables (RW benchmark), we optionally forecast Δy and reconstruct the level:
    #       Δy_{t} = y_{t} - y_{t-1},  ŷ_t = y_{t-1} + \widehat{Δy}_t
    # - For return/growth variables, we typically forecast the level directly.
    _tt = (gpr_target_transform or 'auto').strip().lower()
    if _tt not in {'auto', 'level', 'delta'}:
        _tt = 'auto'
    use_delta = False
    if _tt == 'delta':
        use_delta = True
    elif _tt == 'auto' and target in vars_level_rw1:
        use_delta = True
    transform_used = 'delta' if use_delta else 'level'

    # Load data once if not provided
    if df_all is None:
        df_all = load_data(target, data_dir=data_dir)

    if target not in df_all.columns:
        raise ValueError(f"Target '{target}' not found in aligned dataset columns.")

    y = df_all[target]

    # Build exog (optional)
    feature_cols: list[str] = []
    if exo_var:
        X = df_all[exo_var]
        X_exog = make_freq_aware_lagged_exog(X, exo_var, target, m_vars, q_vars, y_vars)

        # Align
        y = y.loc[X_exog.dropna(how='all').index]
        X_exog = X_exog.loc[y.index].dropna()
        data = pd.concat([y, X_exog], axis=1).dropna()
        feature_cols = list(exo_var)
    else:
        data = y.dropna().to_frame(name=target)

    # period filter
    data = data[data.index.year >= start_year]
    if end_year is not None:
        data = data[data.index.year <= end_year]

    exo_var_str = '_'.join(exo_var) if exo_var else 'None'

    # Ensure output folder exists
    _ = _get_target_output_dir(save_dir, start_year, order_fixed, target, stage1_model='gpr')

    if data.empty:
        return {
            'variable': target,
            'model': 'gpr',
            'exo_vars': exo_var_str,
            'gpr_kernel': None,
            'sigma_source': sigma_src,
            'IS_R2': np.nan,
            'IS_MSE': np.nan,
            'OOS_R2': np.nan,
            'OOS_MSE': np.nan,
            'IS_GARCH_STD': np.nan,
            'OOS_GARCH_STD': np.nan,
            'STD / RMSE': np.nan,
            'error': 'No data after date filtering'
        }

    years = sorted(data.index.year.unique())
    if len(years) <= initial_train_years:
        return {
            'variable': target,
            'model': 'gpr',
            'exo_vars': exo_var_str,
            'gpr_kernel': None,
            'sigma_source': sigma_src,
            'IS_R2': np.nan,
            'IS_MSE': np.nan,
            'OOS_R2': np.nan,
            'OOS_MSE': np.nan,
            'IS_GARCH_STD': np.nan,
            'OOS_GARCH_STD': np.nan,
            'STD / RMSE': np.nan,
            'error': f'Not enough years for initial_train_years={initial_train_years} (have {len(years)})'
        }

    train_end_year = years[initial_train_years - 1]
    test_dates = data[data.index.year > train_end_year].index
    if len(test_dates) == 0:
        return {
            'variable': target,
            'model': 'gpr',
            'exo_vars': exo_var_str,
            'gpr_kernel': None,
            'sigma_source': sigma_src,
            'IS_R2': np.nan,
            'IS_MSE': np.nan,
            'OOS_R2': np.nan,
            'OOS_MSE': np.nan,
            'IS_GARCH_STD': np.nan,
            'OOS_GARCH_STD': np.nan,
            'STD / RMSE': np.nan,
            'error': 'No OOS test dates (check sample window)'
        }

    # Default max train size: match initial training window length
    if gpr_max_train_size is None:
        gpr_max_train_size = initial_train_years * _freq_steps_per_year(target)

    # Kernel setup
    steps_per_year = _freq_steps_per_year(target)
    seasonal_period = float(steps_per_year)

    chosen_kernel_name = (gpr_kernel or 'rbf').strip().lower()
    if gpr_kernel_candidates is not None and len(gpr_kernel_candidates) > 0:
        # We'll pick the best kernel on the *first* window.
        kernel_candidates = [k.strip() for k in gpr_kernel_candidates if str(k).strip()]
    else:
        kernel_candidates = []

    # Accumulators
    oos_preds: list[float] = []
    oos_vars: list[float] = []
    oos_hist_mean_preds: list[float] = []
    train_mses: list[float] = []

    # Optional: reuse fixed GARCH order (saves time)
    garch_order: tuple[int, int] | None = None

    variable_start_time = time.time()

    # Convenience aliases
    y_all = data[target].astype(float)
    X_all = data[feature_cols].astype(float) if feature_cols else None

    # We select kernel on first window (if candidates provided)
    kernel_selected = False

    for dt in tqdm(test_dates, desc=f"{target} GPR OOS [{exo_var_str}]", disable=not verbose):
        # expanding train (capped)
        y_tr_full = y_all.loc[y_all.index < dt]
        y_tr = _slice_tail(y_tr_full, gpr_max_train_size)

        if len(y_tr) < 5:
            oos_preds.append(np.nan)
            oos_vars.append(np.nan)
            train_mses.append(np.nan)
            oos_hist_mean_preds.append(float(y_tr_full.mean()) if len(y_tr_full) else np.nan)
            continue

        # Benchmark mean (expanding, as in the ARIMAX code)
        oos_hist_mean_preds.append(float(y_tr_full.mean()))

        # Build supervised dataset: y_t ~ f(y_{t-1}, ..., y_{t-L}, x_t, [t])
        L = gpr_ar_lags
        if len(y_tr) <= L:
            oos_preds.append(np.nan)
            oos_vars.append(np.nan)
            train_mses.append(np.nan)
            continue

        idx_eff = y_tr.index[L:]

        if use_delta:
            y_train = y_tr.diff().loc[idx_eff].values.reshape(-1, 1)
        else:
            y_train = y_tr.loc[idx_eff].values.reshape(-1, 1)

        feat_list: list[np.ndarray] = []
        for lag in range(1, L + 1):
            feat_list.append(y_tr.shift(lag).loc[idx_eff].values.reshape(-1, 1))

        if X_all is not None:
            X_tr_full = X_all.loc[X_all.index < dt]
            X_tr = _slice_tail(X_tr_full, gpr_max_train_size)
            feat_list.append(X_tr.loc[idx_eff].values)

        if gpr_add_time_index:
            # Simple linear time index over the capped window
            t_feat = np.arange(len(y_tr), dtype=float).reshape(-1, 1)[L:]
            feat_list.append(t_feat)

        X_train = np.hstack(feat_list)

        # Standardize features (important for GP kernels)
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)

        # Kernel selection (once)
        if (not kernel_selected) and kernel_candidates:
            try:
                chosen_kernel_name = _select_best_gpr_kernel(
                    X_train_std,
                    y_train.ravel(),
                    kernel_candidates=kernel_candidates,
                    alpha=gpr_alpha,
                    optimizer=gpr_optimizer,
                    selection=gpr_kernel_select,
                    seasonal_period=seasonal_period,
                )
                kernel_selected = True
                if verbose:
                    print(f"[{target}] Selected GPR kernel: {chosen_kernel_name}")
            except Exception:
                # Fall back silently
                kernel_selected = True

        # Build kernel object
        try:
            kernel = _build_gpr_kernel(chosen_kernel_name, seasonal_period=seasonal_period)
        except Exception:
            kernel = _build_gpr_kernel('rbf', seasonal_period=seasonal_period)
            chosen_kernel_name = 'rbf'

        # Fit GPR
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=gpr_alpha,
            optimizer=gpr_optimizer,
            normalize_y=True,
            random_state=0,
        )

        try:
            gpr.fit(X_train_std, y_train.ravel())
        except Exception:
            oos_preds.append(np.nan)
            oos_vars.append(np.nan)
            train_mses.append(np.nan)
            continue

        # In-sample fit diagnostics
        try:
            y_fit = gpr.predict(X_train_std)
            train_mses.append(float(mean_squared_error(y_train.ravel(), y_fit)))
        except Exception:
            y_fit = None
            train_mses.append(np.nan)

        # Prepare test features for dt
        test_feat: list[np.ndarray] = []
        for lag in range(1, L + 1):
            try:
                y_prev_l = float(y_tr.iloc[-lag])
            except Exception:
                y_prev_l = np.nan
            test_feat.append(np.array([[y_prev_l]], dtype=float))

        if X_all is not None:
            x_dt = X_all.loc[dt, feature_cols].values.reshape(1, -1)
            test_feat.append(x_dt.astype(float))

        if gpr_add_time_index:
            t_next = np.array([[len(y_tr)]], dtype=float)
            test_feat.append(t_next)

        X_test = np.hstack(test_feat)
        X_test_std = scaler.transform(X_test)

        # Predict mean and (optionally) native std
        try:
            y_pred, y_std = gpr.predict(X_test_std, return_std=True)
            y_pred_val = float(y_pred.ravel()[0])
            native_var = float((y_std.ravel()[0]) ** 2)
        except Exception:
            y_pred_val = np.nan
            native_var = np.nan

        # If we forecast Δy, reconstruct the level forecast: ŷ_t = y_{t-1} + Δŷ_t
        if use_delta and np.isfinite(y_pred_val):
            try:
                y_pred_val = float(y_tr_full.iloc[-1]) + float(y_pred_val)
            except Exception:
                y_pred_val = np.nan

        oos_preds.append(y_pred_val)

        # Sigma / variance proxy
        var = np.nan
        if sigma_src == 'native':
            var = native_var

        else:
            # residual-based sources need residuals
            if y_fit is None:
                try:
                    y_fit = gpr.predict(X_train_std)
                except Exception:
                    y_fit = None

            if y_fit is None:
                var = np.nan
            else:
                resid = (y_train.ravel() - np.asarray(y_fit).ravel())
                resid_s = pd.Series(resid, index=idx_eff)

                if sigma_src == 'garch' and compute_garch and (arch_model is not None) and (len(resid_s.dropna()) >= 20):
                    try:
                        if (not garch_order_fixed) or (garch_order is None):
                            gfit, garch_order = select_best_garch_order(resid_s.dropna(), vol='GARCH')
                        else:
                            p, q = garch_order
                            gfit = arch_model(resid_s.dropna(), mean='zero', vol='GARCH', p=p, q=q).fit(disp='off', show_warning=False)
                        gfc = gfit.forecast(horizon=1)
                        var = float(gfc.variance.iloc[-1, 0])
                    except Exception:
                        var = float(resid_s.var())
                else:
                    var = float(resid_s.var())

        oos_vars.append(var)

    variable_end_time = time.time()
    if verbose:
        print(f"'{target}' (GPR) elapsed: {variable_end_time - variable_start_time:.2f}s")

    oos_series = pd.Series(oos_preds, index=test_dates, name='gpr_pred')
    oos_actuals = y_all.loc[test_dates]
    oos_hist_mean_series = pd.Series(oos_hist_mean_preds, index=test_dates, name='hist_mean_pred')

    # IS metrics: fit once on the last available training window and compute fitted values
    last_dt = test_dates[-1]
    y_tr_full = y_all.loc[y_all.index < last_dt]
    y_tr = _slice_tail(y_tr_full, gpr_max_train_size)

    if len(y_tr) < 5:
        is_pred = pd.Series(dtype=float)
        is_actuals = pd.Series(dtype=float)
    else:
        L = gpr_ar_lags
        if len(y_tr) <= L:
            is_pred = pd.Series(dtype=float)
            is_actuals = pd.Series(dtype=float)
        else:
            idx_eff = y_tr.index[L:]

            if use_delta:
                y_train = y_tr.diff().loc[idx_eff].values.reshape(-1, 1)
            else:
                y_train = y_tr.loc[idx_eff].values.reshape(-1, 1)

            feat_list: list[np.ndarray] = []
            for lag in range(1, L + 1):
                feat_list.append(y_tr.shift(lag).loc[idx_eff].values.reshape(-1, 1))

            if X_all is not None:
                X_tr_full = X_all.loc[X_all.index < last_dt]
                X_tr = _slice_tail(X_tr_full, gpr_max_train_size)
                feat_list.append(X_tr.loc[idx_eff].values)

            if gpr_add_time_index:
                feat_list.append(np.arange(len(y_tr), dtype=float).reshape(-1, 1)[L:])

            X_train = np.hstack(feat_list)
            scaler = StandardScaler()
            X_train_std = scaler.fit_transform(X_train)

            kernel = _build_gpr_kernel(chosen_kernel_name, seasonal_period=seasonal_period)

            gpr = GaussianProcessRegressor(
                kernel=kernel,
                alpha=gpr_alpha,
                optimizer=gpr_optimizer,
                normalize_y=True,
                random_state=0,
            )

            try:
                gpr.fit(X_train_std, y_train.ravel())
                y_fit = gpr.predict(X_train_std)
                if use_delta:
                    prev_level = y_tr.shift(1).loc[idx_eff].values
                    is_pred = pd.Series(prev_level + np.asarray(y_fit).ravel(), index=idx_eff)
                else:
                    is_pred = pd.Series(y_fit, index=idx_eff)
                is_actuals = y_tr.loc[idx_eff]
            except Exception:
                is_pred = pd.Series(dtype=float)
                is_actuals = pd.Series(dtype=float)

    avg_train_mse = float(np.nanmean(train_mses)) if len(train_mses) else np.nan

    # IS sigma proxy (for reporting only)
    is_std = np.nan
    if len(is_actuals) and len(is_pred):
        resid_is = (is_actuals - is_pred).dropna()
        if sigma_src == 'garch' and compute_garch and (arch_model is not None) and (len(resid_is) >= 20):
            try:
                if (not garch_order_fixed) or (garch_order is None):
                    gfit_is, garch_order = select_best_garch_order(resid_is, vol='GARCH')
                else:
                    p, q = garch_order
                    gfit_is = arch_model(resid_is, mean='zero', vol='GARCH', p=p, q=q).fit(disp='off', show_warning=False)
                is_var = float(np.mean(gfit_is.conditional_volatility ** 2))
                is_std = float(np.sqrt(is_var))
            except Exception:
                is_std = float(np.sqrt(resid_is.var()))
        else:
            is_std = float(np.sqrt(resid_is.var()))

    oos_var_series = pd.Series(oos_vars, index=test_dates, name='oos_var')
    uncertainty_metrics = _compute_uncertainty_diagnostics(
        oos_actuals,
        oos_series,
        oos_var_series,
        is_std=is_std,
    )

    # Compute IS/OOS metrics (reuse original benchmark rules)
    if target in vars_level_rw1:
        is_mse, is_r2 = r2_level("GPR", is_actuals, is_pred, data_type='IS', verbose=verbose)
        oos_mse, oos_r2 = r2_level("GPR", oos_actuals, oos_series, data_type='OOS', verbose=verbose)

    elif target in vars_return_ao:
        is_mse, is_r2 = r2_return("ARIMAX", is_actuals, is_pred, data_type='IS', verbose=verbose)
        oos_mse, oos_r2 = r2_return_oos_ao("ARIMAX", oos_actuals, oos_series, data_type='OOS', verbose=verbose)

    elif target in vars_return_gw:
        is_mse, is_r2 = r2_return("GPR", is_actuals, is_pred, data_type='IS', verbose=verbose)
        oos_mse, oos_r2 = r2_return_oos("GPR", oos_actuals, oos_series, oos_hist_mean_series, data_type='OOS', verbose=verbose)

    else:
        is_mse, is_r2, oos_mse, oos_r2 = np.nan, np.nan, np.nan, np.nan

    metrics_row = {
        'variable': target,
        'model': 'gpr',
        'exo_vars': exo_var_str,
        'gpr_ar_lags': gpr_ar_lags,
        'gpr_target_transform': transform_used,
        'gpr_kernel': chosen_kernel_name,
        'sigma_source': sigma_src,
        'IS_R2': is_r2,
        'IS_MSE': avg_train_mse,
        'OOS_R2': oos_r2,
        'OOS_MSE': oos_mse,
        **uncertainty_metrics,
    }

    if save_outputs:
        out_dir = _get_target_output_dir(save_dir, start_year, order_fixed, target, stage1_model='gpr')
        pd.DataFrame({'actual': oos_actuals, 'pred': oos_series}).to_csv(os.path.join(out_dir, f"OOS({exo_var_str}).csv"))
        pd.DataFrame({'actual': is_actuals, 'pred': is_pred}).to_csv(os.path.join(out_dir, f"IS({exo_var_str}).csv"))
        oos_var_series.to_csv(os.path.join(out_dir, f"OOS_GARCH_VAR({exo_var_str}).csv"))
        pd.DataFrame([metrics_row]).to_csv(os.path.join(out_dir, f"metrics({exo_var_str}).csv"), index=False)

        if make_plots:
            plt.figure(figsize=(12, 4))
            plt.plot(oos_actuals, label='Real')
            plt.plot(oos_series, label='Pred')
            plt.legend(); plt.grid()
            plt.savefig(os.path.join(out_dir, f"plot({exo_var_str}).png"), dpi=100)
            if show_plots:
                plt.show()
            plt.close()

    return metrics_row



# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------

def main(
    target_variable: str,
    exo_var: str = 'None',
    *,
    stage1_model: str = 'arimax',
    data_dir: str = DEFAULT_DATA_DIR,
    save_dir: str = DEFAULT_SAVE_DIR,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int | None = DEFAULT_END_YEAR,
    initial_train_years: int = DEFAULT_INITIAL_TRAIN_YEARS,
    order_fixed: bool = DEFAULT_ORDER_FIXED,
    # Sigma / volatility
    sigma_source: str | None = None,
    garch_order_fixed: bool = False,
    # ETS knobs
    ets_allow_seasonal: bool = True,
    ets_select_criterion: str = 'aicc',
    # GPR knobs
    gpr_max_train_size: int | None = None,
    gpr_ar_lags: int = 1,
    gpr_add_time_index: bool = False,
    gpr_alpha: float = 1e-6,
    gpr_optimizer: str | None = None,
    gpr_kernel: str = 'rbf',
    gpr_kernel_candidates: list[str] | None = None,
    gpr_kernel_select: str = 'lml',
    gpr_target_transform: str = 'auto',
    verbose: bool = True,
):
    """Run stage-1 for a single target with a specified exogenous variable.

    Args:
        target_variable: target variable name (slash-stripped).
        exo_var: 'None' for univariate, or a variable name (e.g. 'dfy').
    """
    model = _normalize_stage1_model(stage1_model)
    exo_var = (exo_var or 'None').strip()
    exo_list = [] if exo_var == 'None' else [exo_var]

    if verbose:
        print(f"[Stage1] model={model} / target={target_variable} / exo={exo_var}")

    df = load_data(target_variable, data_dir=data_dir)

    if model == 'ets':
        row = test_ets_var(
            target_variable, exo_list,
            df_all=df, data_dir=data_dir, save_dir=save_dir,
            start_year=start_year, end_year=end_year,
            initial_train_years=initial_train_years, order_fixed=order_fixed,
            ets_allow_seasonal=ets_allow_seasonal,
            ets_select_criterion=ets_select_criterion,
            sigma_source=sigma_source, compute_garch=True,
            garch_order_fixed=garch_order_fixed,
            save_outputs=True, make_plots=True, show_plots=False,
            verbose=verbose,
        )
    elif model == 'gpr':
        row = test_gpr_var(
            target_variable, exo_list,
            df_all=df, data_dir=data_dir, save_dir=save_dir,
            start_year=start_year, end_year=end_year,
            initial_train_years=initial_train_years, order_fixed=order_fixed,
            gpr_max_train_size=gpr_max_train_size,
            gpr_ar_lags=gpr_ar_lags,
            gpr_target_transform=gpr_target_transform,
            gpr_add_time_index=gpr_add_time_index,
            gpr_alpha=gpr_alpha,
            gpr_optimizer=gpr_optimizer,
            gpr_kernel=gpr_kernel,
            gpr_kernel_candidates=gpr_kernel_candidates,
            gpr_kernel_select=gpr_kernel_select,
            sigma_source=sigma_source, compute_garch=True,
            garch_order_fixed=garch_order_fixed,
            save_outputs=True, make_plots=True, show_plots=False,
            verbose=verbose,
        )
    else:
        # default: ARIMAX
        row = test_exo_var(
            target_variable, exo_list,
            df_all=df, data_dir=data_dir, save_dir=save_dir,
            start_year=start_year, end_year=end_year,
            initial_train_years=initial_train_years, order_fixed=order_fixed,
            sigma_source=sigma_source, compute_garch=True,
            garch_order_fixed=garch_order_fixed,
            save_outputs=True, make_plots=True, show_plots=False,
            verbose=verbose,
        )

    # Save best.csv / best.json
    out_dir = _get_target_output_dir(save_dir, start_year, order_fixed,
                                      target_variable, stage1_model=model)
    pd.DataFrame([row]).to_csv(os.path.join(out_dir, 'best.csv'), index=False)
    with open(os.path.join(out_dir, 'best.json'), 'w') as f:
        json.dump({
            'target': target_variable,
            'stage1_model': model,
            'best_exo': None if exo_var == 'None' else exo_var,
            'start_year': start_year,
            'end_year': end_year,
            'initial_train_years': initial_train_years,
            'order_fixed': order_fixed,
        }, f, indent=2)

    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage-1 forecasting (ARIMAX / ETS / GPR)")
    parser.add_argument('--target', type=str, required=True,
                        help='Target variable name (slash-stripped).')
    parser.add_argument('--exo_var', type=str, default='None',
                        help="'None' (default) or a variable name (e.g. 'dfy').")
    parser.add_argument('--model', type=str, default='arimax',
                        choices=['arimax', 'ets', 'gpr', 'grf'],
                        help="Stage-1 model: 'arimax' (default) | 'ets' | 'gpr'")

    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument('--start_year', type=int, default=DEFAULT_START_YEAR)
    parser.add_argument('--end_year', type=int, default=DEFAULT_END_YEAR)
    parser.add_argument('--initial_train_years', type=int, default=DEFAULT_INITIAL_TRAIN_YEARS)
    parser.add_argument('--order_fixed', action=argparse.BooleanOptionalAction,
                        default=DEFAULT_ORDER_FIXED)

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
    parser.add_argument('--gpr_ar_lags', type=int, default=1)
    parser.add_argument('--gpr_add_time_index', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--gpr_alpha', type=float, default=1e-6)
    parser.add_argument('--gpr_optimizer', type=str, default=None)
    parser.add_argument('--gpr_kernel', type=str, default='rbf')
    parser.add_argument('--gpr_kernel_candidates', type=str, default=None)
    parser.add_argument('--gpr_kernel_select', type=str, default='lml', choices=['lml', 'mse'])
    parser.add_argument('--gpr_target_transform', type=str, default='auto')

    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    gpr_kernel_candidates = None
    if args.gpr_kernel_candidates:
        gpr_kernel_candidates = [x.strip() for x in args.gpr_kernel_candidates.split(',') if x.strip()]

    _ = main(
        strip_slash([args.target])[0],
        args.exo_var,
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
        gpr_add_time_index=args.gpr_add_time_index,
        gpr_alpha=args.gpr_alpha,
        gpr_optimizer=args.gpr_optimizer,
        gpr_kernel=args.gpr_kernel,
        gpr_kernel_candidates=gpr_kernel_candidates,
        gpr_kernel_select=args.gpr_kernel_select,
        gpr_target_transform=args.gpr_target_transform,
        verbose=args.verbose,
    )
