"""
통합 RF 실험 시스템 - 평가 모듈

Campbell & Thompson (2008) OOS R² 계산, 벤치마크 생성, 시각화를 유지하면서,
final 최종 재현 run에 필요한 확장 지표와 검정을 제공합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm


# ============================================================
# Core forecast utilities
# ============================================================

def compute_oos_r2(y_true, y_pred, y_benchmark):
    """Campbell & Thompson (2008) OOS R-squared."""
    df = pd.concat({'y': y_true, 'yhat': y_pred, 'bench': y_benchmark}, axis=1).dropna()
    if df.empty:
        return np.nan, np.nan

    sse_model = (df['y'] - df['yhat']).pow(2).sum()
    sse_benchmark = (df['y'] - df['bench']).pow(2).sum()
    mse = float(mean_squared_error(df['y'], df['yhat']))
    if float(sse_benchmark) <= 0:
        return mse, np.nan
    r2 = 1.0 - float(sse_model / sse_benchmark)
    return mse, r2



def compute_expanding_mean_benchmark(data, test_years):
    """확장 평균 벤치마크를 계산합니다."""
    first_test_year = test_years[0]
    initial_window_end = f'{first_test_year - 1}-12-31'
    min_periods = len(data.loc[:initial_window_end, 'rp'])
    return data['rp'].shift(1).expanding(min_periods=min_periods).mean()



def compute_in_sample_r2(y_true, y_pred):
    """표준 in-sample R² = 1 - SSE / SST."""
    df = pd.concat({'y': y_true, 'yhat': y_pred}, axis=1).dropna()
    if df.empty:
        return np.nan
    sse = float(((df['y'] - df['yhat']) ** 2).sum())
    sst = float(((df['y'] - df['y'].mean()) ** 2).sum())
    if sst <= 0:
        return np.nan
    return float(1.0 - sse / sst)



def compute_hit_rate(y_true, y_pred):
    """방향성 hit rate."""
    df = pd.concat({'y': y_true, 'yhat': y_pred}, axis=1).dropna()
    if df.empty:
        return np.nan
    return float((np.sign(df['y']) == np.sign(df['yhat'])).mean())



def plot_oos_predictions(y_true, y_pred, r2, save_path, title_suffix=''):
    """OOS 예측 vs 실제 플롯 저장."""
    plt.figure(figsize=(12, 4))
    plt.plot(y_pred, label='Pred')
    plt.plot(y_true, label='Real')
    plt.title(f'OOS R2 ({r2:.4f}){title_suffix}')
    plt.grid()
    plt.legend()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


# ============================================================
# Forecast summary metrics
# ============================================================

def forecast_metrics_summary(y: pd.Series, yhat: pd.Series, bench: pd.Series):
    """핵심 예측 정확도 요약 지표."""
    df = pd.concat({'y': y, 'yhat': yhat, 'bench': bench}, axis=1).dropna()
    if df.empty:
        return {
            'n_oos': 0,
            'OOS_R2': np.nan,
            'MSE': np.nan,
            'RMSE': np.nan,
            'MAE': np.nan,
            'MSE_bench': np.nan,
            'RMSE_bench': np.nan,
            'MAE_bench': np.nan,
            'RRMSE': np.nan,
            'MASE': np.nan,
            'hit_rate': np.nan,
        }

    err_m = df['y'] - df['yhat']
    err_b = df['y'] - df['bench']

    mse_m = float(np.mean(err_m ** 2))
    mse_b = float(np.mean(err_b ** 2))
    rmse_m = float(np.sqrt(mse_m))
    rmse_b = float(np.sqrt(mse_b))
    mae_m = float(np.mean(err_m.abs()))
    mae_b = float(np.mean(err_b.abs()))
    r2_oos = 1.0 - (mse_m / mse_b) if mse_b > 0 else np.nan

    return {
        'n_oos': int(len(df)),
        'OOS_R2': float(r2_oos),
        'MSE': mse_m,
        'RMSE': rmse_m,
        'MAE': mae_m,
        'MSE_bench': mse_b,
        'RMSE_bench': rmse_b,
        'MAE_bench': mae_b,
        'RRMSE': float(rmse_m / rmse_b) if rmse_b > 0 else np.nan,
        'MASE': float(mae_m / mae_b) if mae_b > 0 else np.nan,
        'hit_rate': compute_hit_rate(df['y'], df['yhat']),
    }


# ============================================================
# DM / Clark-West tests
# ============================================================

def _hac_tstat_mean(x: pd.Series, maxlags: int = 0):
    x = pd.Series(x).dropna()
    if len(x) < 3:
        return np.nan, np.nan, np.nan, int(len(x))
    X = np.ones((len(x), 1))
    res = sm.OLS(x.values, X).fit(cov_type='HAC', cov_kwds={'maxlags': int(maxlags)})
    mean_hat = float(res.params[0])
    se_hat = float(res.bse[0])
    t = float(res.tvalues[0])
    return mean_hat, se_hat, t, int(len(x))



def _p_value_from_t(t: float, alternative: str = 'two-sided'):
    if np.isnan(t):
        return np.nan
    if alternative == 'two-sided':
        return 2.0 * (1.0 - norm.cdf(abs(t)))
    if alternative == 'greater':
        return 1.0 - norm.cdf(t)
    if alternative == 'less':
        return norm.cdf(t)
    raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")



def dm_test_hac(y: pd.Series, f1: pd.Series, f2: pd.Series, loss: str = 'mse', maxlags: int = 0):
    """Diebold-Mariano test (HAC). f1이 f2보다 좋은지까지 함께 반환."""
    df = pd.concat({'y': y, 'f1': f1, 'f2': f2}, axis=1).dropna()
    if df.empty:
        return {'n': 0, 'mean_d': np.nan, 't': np.nan, 'p_two': np.nan, 'p_f1_better': np.nan}

    if loss == 'mse':
        d = (df['y'] - df['f1']) ** 2 - (df['y'] - df['f2']) ** 2
    elif loss == 'mae':
        d = (df['y'] - df['f1']).abs() - (df['y'] - df['f2']).abs()
    else:
        raise ValueError("loss must be 'mse' or 'mae'.")

    mean_d, _, t, n = _hac_tstat_mean(d, maxlags=maxlags)
    return {
        'n': n,
        'mean_d': mean_d,
        't': t,
        'p_two': _p_value_from_t(t, alternative='two-sided'),
        'p_f1_better': _p_value_from_t(t, alternative='less'),
    }



def clark_west_hac(y: pd.Series, f_model: pd.Series, f_bench: pd.Series, maxlags: int = 0):
    """Clark-West test (nested models)."""
    df = pd.concat({'y': y, 'fm': f_model, 'fb': f_bench}, axis=1).dropna()
    if df.empty:
        return {'n': 0, 'mean_d': np.nan, 't': np.nan, 'p_one': np.nan}

    e_b = df['y'] - df['fb']
    e_m = df['y'] - df['fm']
    adj = (df['fm'] - df['fb']) ** 2
    d = e_b ** 2 - (e_m ** 2 - adj)

    mean_d, _, t, n = _hac_tstat_mean(d, maxlags=maxlags)
    return {
        'n': n,
        'mean_d': mean_d,
        't': t,
        'p_one': _p_value_from_t(t, alternative='greater'),
    }


# ============================================================
# Tail-conditional metrics
# ============================================================

def _conditional_metrics_block(y_true: pd.Series, y_pred: pd.Series, y_bench: pd.Series,
                               q: float = 0.10, side: str = 'down'):
    base = pd.concat({'y': y_true, 'yhat': y_pred, 'bench': y_bench}, axis=1).dropna()
    if base.empty:
        return {'q': q, 'side': side, 'threshold': np.nan, 'n': 0,
                'R2_cond': np.nan, 'MAE_pred': np.nan, 'MAE_bench': np.nan,
                'hit_rate': np.nan}

    if side == 'down':
        thr = float(base['y'].quantile(q))
        subset = base[base['y'] <= thr]
    elif side == 'up':
        thr = float(base['y'].quantile(1.0 - q))
        subset = base[base['y'] >= thr]
    else:
        raise ValueError("side must be 'down' or 'up'.")

    n = int(len(subset))
    if n == 0:
        return {'q': q, 'side': side, 'threshold': thr, 'n': 0,
                'R2_cond': np.nan, 'MAE_pred': np.nan, 'MAE_bench': np.nan,
                'hit_rate': np.nan}

    mse_pred = float(np.mean((subset['y'] - subset['yhat']) ** 2))
    mse_bench = float(np.mean((subset['y'] - subset['bench']) ** 2))
    r2 = 1.0 - (mse_pred / mse_bench) if mse_bench > 0 else np.nan
    mae_pred = float(np.mean((subset['y'] - subset['yhat']).abs()))
    mae_bench = float(np.mean((subset['y'] - subset['bench']).abs()))
    hr = compute_hit_rate(subset['y'], subset['yhat'])

    return {
        'q': float(q),
        'side': side,
        'threshold': thr,
        'n': n,
        'R2_cond': float(r2),
        'MAE_pred': mae_pred,
        'MAE_bench': mae_bench,
        'hit_rate': hr,
    }



def conditional_metrics_by_realized_quantiles(y_true: pd.Series, y_pred: pd.Series,
                                              y_bench: pd.Series, q_values):
    rows = []
    for q in q_values:
        rows.append(_conditional_metrics_block(y_true, y_pred, y_bench, q=float(q), side='down'))
        rows.append(_conditional_metrics_block(y_true, y_pred, y_bench, q=float(q), side='up'))
    return pd.DataFrame(rows)


# ============================================================
# Volatility evaluation
# ============================================================

def compute_qlike(rv, var_hat):
    """QLIKE loss."""
    rv = np.asarray(rv, dtype=float)
    var_hat = np.asarray(var_hat, dtype=float)
    rv = np.clip(rv, 1e-12, None)
    var_hat = np.clip(var_hat, 1e-12, None)
    ratio = rv / var_hat
    return ratio - np.log(ratio) - 1.0



def variance_error_metrics(rv_proxy: pd.Series, var_hat: pd.Series):
    df = pd.concat({'rv': rv_proxy, 'var_hat': var_hat}, axis=1).dropna()
    if df.empty:
        return {'var_MSE': np.nan, 'var_RMSE': np.nan, 'var_MAE': np.nan, 'avg_QLIKE': np.nan}
    err = df['rv'] - df['var_hat']
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    qlike = float(np.mean(compute_qlike(df['rv'].values, df['var_hat'].values)))
    return {
        'var_MSE': mse,
        'var_RMSE': float(np.sqrt(mse)),
        'var_MAE': mae,
        'avg_QLIKE': qlike,
    }



def variance_coverage(y_true: pd.Series, y_pred: pd.Series, var_hat: pd.Series, k: float = 1.96):
    df = pd.concat({'y': y_true, 'yhat': y_pred, 'var_hat': var_hat}, axis=1).dropna()
    if df.empty:
        return np.nan
    sigma = np.sqrt(np.clip(df['var_hat'].astype(float).values, 1e-12, None))
    upper = df['yhat'].astype(float).values + k * sigma
    lower = df['yhat'].astype(float).values - k * sigma
    outside = (df['y'].astype(float).values > upper) | (df['y'].astype(float).values < lower)
    return float(1.0 - outside.mean())
