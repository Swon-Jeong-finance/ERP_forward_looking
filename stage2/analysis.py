"""
analysis.py — Portfolio & Forecast Evaluation (v2)

v1 final 실행 결과 디렉토리에서 자동 로딩하여
forecast 평가 + 포트폴리오 평가 + transaction cost 분석을 수행합니다.

Expected directory structure:
    output/final_{index_type}/          (or output(non-sigma)/final_{index_type}/)
        {dr}__{ft}__{it}__tau{rc}/
            forecast_oos.csv
            metrics_summary.csv
            vol_forecasts.csv
            run_config.json
    benchmark/
        CAPM.csv  (or CAPM_{index_type}.csv)
        FF3.csv   (or FF3_{index_type}.csv)

Usage:
    python analysis.py --index_type sp500
    python analysis.py --index_type sp500 --no_sigma
    python analysis.py --index_type russell3000 --no_sigma --vol_spec rolling_var
"""

import argparse
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm


# =============================================================
# Default Configuration (CLI args override these)
# =============================================================
DEFAULT_INDEX_TYPE = 'sp500'
DEFAULT_USE_SIGMA = True
DEFAULT_VOL_SPEC = 'best_vol_var'

BENCHMARK_DIR = 'benchmark'
DATA_DIR = 'data'

TAIL_QUANTILES = (0.05, 0.10, 0.15, 0.20)
DM_HAC_MAXLAGS = 0
CW_HAC_MAXLAGS = 0
HAC_SENSITIVITY_LAGS = (0, 1, 3)
TX_COST_BPS_LIST = (0, 10, 25, 50)

PAPER_EXCLUDE = {'CAPM', 'FF3'}
RUN_PORTFOLIO_EVAL = True

MV_GAMMA = 3.0
MV_MAX_LEVERAGE = 1.5
MV_MIN_LEVERAGE = 0.0

BENCHMARK_FILES = {
    'CAPM': 'CAPM.csv',
    'FF3': 'FF3.csv',
}
IR_BENCHMARK = 'CAPM'


# =============================================================
# 1. Data Loading
# =============================================================

def _parse_combo_dir(dirname):
    """Parse v1 combo dir name '{dr}__{ft}__{it}__tau{rc}' -> tuple or None."""
    parts = dirname.split('__')
    if len(parts) != 4 or not parts[3].startswith('tau'):
        return None
    try:
        rc = float(parts[3][3:].replace('p', '.').replace('m', '-'))
    except ValueError:
        return None
    return parts[0], parts[1], parts[2], rc


def _strategy_label(dr, ft, rc):
    """(dim_reduction, feature_type, r2_cut) -> human-readable name."""
    suffix = {
        'none': '', 'pca': '_pca', 'pls': '_pls',
        'shap_pca': '_pca_shap', 'shap_pls': '_pls_shap',
    }.get(dr, f'_{dr}')
    if ft == 'past':
        return f'past{suffix}'
    return f'dual{suffix}_{rc:g}'


def load_final_results(final_dir, index_type, vol_spec='best_vol_var'):
    """v1 final 출력 디렉토리 스캔 -> 예측 시계열 + 메트릭스 로딩."""
    if not os.path.isdir(final_dir):
        raise FileNotFoundError(f'Final directory not found: {final_dir}')

    rp_parts, vol_parts, met_parts = [], [], []

    for entry in sorted(os.listdir(final_dir)):
        subdir = os.path.join(final_dir, entry)
        if not os.path.isdir(subdir):
            continue

        cfg_path = os.path.join(subdir, 'run_config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            dr = cfg.get('dim_reduction', '')
            ft = cfg.get('feature_type', '')
            it = cfg.get('index_type', '')
            rc = float(cfg.get('r2_cut', 0))
        else:
            parsed = _parse_combo_dir(entry)
            if parsed is None:
                continue
            dr, ft, it, rc = parsed

        if it != index_type:
            continue

        fpath = os.path.join(subdir, 'forecast_oos.csv')
        if not os.path.exists(fpath):
            print(f'[WARN] Missing forecast_oos.csv in {entry}')
            continue

        name = _strategy_label(dr, ft, rc)
        fdf = pd.read_csv(fpath, parse_dates=['date']).set_index('date')
        rp_parts.append(fdf['rp_pred'].rename(f'{name}_rp'))

        if vol_spec == 'best_vol_var' and 'best_vol_var' in fdf.columns:
            vol_parts.append(fdf['best_vol_var'].rename(f'{name}_vol'))
        elif vol_spec != 'rolling_var':
            vpath = os.path.join(subdir, 'vol_forecasts.csv')
            if os.path.exists(vpath):
                vdf = pd.read_csv(vpath, index_col=0, parse_dates=True)
                if vol_spec in vdf.columns:
                    vol_parts.append(vdf[vol_spec].rename(f'{name}_vol'))

        mpath = os.path.join(subdir, 'metrics_summary.csv')
        if os.path.exists(mpath):
            mdf = pd.read_csv(mpath)
            mdf['strategy'] = name
            mdf['combo_dir'] = entry
            met_parts.append(mdf)

    if not rp_parts:
        raise ValueError(f'No valid results in {final_dir}')

    pred_df = pd.concat(rp_parts, axis=1)
    if vol_parts:
        pred_df = pred_df.join(pd.concat(vol_parts, axis=1), how='left')
    met_df = pd.concat(met_parts, ignore_index=True) if met_parts else pd.DataFrame()
    return pred_df, met_df


def load_benchmarks(benchmark_dir, index_type, benchmark_files=None):
    """벤치마크 CSV 로딩."""
    if benchmark_files is None:
        benchmark_files = BENCHMARK_FILES

    parts = []
    for bname, fname in benchmark_files.items():
        stem = fname.replace('.csv', '')
        candidates = [
            os.path.join(benchmark_dir, f'{stem}_{index_type}.csv'),
            os.path.join(benchmark_dir, fname),
        ]
        path = next((c for c in candidates if os.path.exists(c)), None)
        if path is None:
            print(f'[WARN] Benchmark not found: {bname} (tried {candidates})')
            continue

        df = pd.read_csv(path)
        df.columns.values[0] = 'date'
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        rp_col = 'rp_pred' if 'rp_pred' in df.columns else df.columns[0]
        rename_map = {rp_col: f'{bname}_rp'}

        for try_col in ['best_vol_var', 'GARCH', 'GJR', 'EGARCH']:
            if try_col in df.columns:
                rename_map[try_col] = f'{bname}_vol'
                break

        parts.append(df.rename(columns=rename_map)[[v for v in rename_map.values()]])

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=1).sort_index()


def load_actual_returns(data_dir, index_type='sp500', start_year=1973):
    """실제 수익률 시계열 로딩 (index_type별 분기)."""
    tmp = pd.read_csv(os.path.join(data_dir, 'monthly.csv'))
    tmp['date'] = pd.to_datetime(tmp['date'], format='%Y%m')
    tmp.set_index('date', inplace=True)
    tmp.sort_index(inplace=True)

    if index_type in ('sp500', 'sp500_short'):
        tmp['actual_rp'] = tmp['ret'] - tmp['Rfree']
        return tmp.loc[f'{start_year}-01-01':]

    elif index_type == 'russell3000':
        rus = pd.read_csv(os.path.join(data_dir, 'Russell3000.csv'))
        rus['date'] = pd.to_datetime(rus['date'], format='%Y%m')
        rus.set_index('date', inplace=True)
        rus['ret'] = np.log(rus['close']).diff()
        rus.dropna(inplace=True)
        merged = rus[['ret']].join(tmp['Rfree'], how='inner')
        merged['actual_rp'] = merged['ret'] - merged['Rfree']
        return merged.loc[f'{start_year}-01-01':]

    elif index_type == 'crsp_index':
        crsp = pd.read_csv(os.path.join(data_dir, 'crsp_index.csv'))
        crsp.columns = crsp.columns.str.lower()
        crsp['date'] = pd.to_datetime(crsp['date']).dt.to_period('M').dt.to_timestamp()
        crsp.set_index('date', inplace=True)
        crsp = crsp.rename(columns={'vwretd': 'ret'})
        merged = crsp[['ret']].join(tmp['Rfree'], how='inner')
        merged['actual_rp'] = merged['ret'] - merged['Rfree']
        return merged.loc[f'{start_year}-01-01':]

    else:
        raise ValueError(f"Unknown index_type: '{index_type}'")


def load_all(final_dir, benchmark_dir, data_dir, index_type,
             vol_spec='best_vol_var', benchmark_files=None, start_year=1973):
    """Master loader."""
    pred_df, v1_met = load_final_results(final_dir, index_type, vol_spec)
    bench_df = load_benchmarks(benchmark_dir, index_type, benchmark_files)

    merge_df = bench_df.join(pred_df, how='inner') if not bench_df.empty else pred_df.copy()

    actual_df = load_actual_returns(data_dir, index_type=index_type, start_year=start_year)
    roll_var = actual_df['actual_rp'].rolling(60).var().shift(1)

    for c in [c for c in merge_df.columns if c.endswith('_rp')]:
        vcol = c.replace('_rp', '_vol')
        if vcol not in merge_df.columns or vol_spec == 'rolling_var':
            merge_df[vcol] = roll_var.reindex(merge_df.index)

    merge_df.sort_index(inplace=True)
    return merge_df, actual_df, v1_met


# =============================================================
# 2. Performance Metrics
# =============================================================

def performance_metrics(ret, rfree, signal=None, benchmark_ret=None,
                        gamma=3.0, result_freq='yearly'):
    """포트폴리오 성과 지표."""
    ret = pd.Series(ret).dropna()
    rfree = pd.Series(rfree, index=ret.index) if not isinstance(rfree, pd.Series) else rfree.reindex(ret.index)
    excess = (ret - rfree).dropna()
    ret = ret.reindex(excess.index)

    mu_m = excess.mean()
    sigma_m = excess.std(ddof=1)
    downside_dev_m = np.sqrt((np.minimum(excess, 0.0) ** 2).mean())

    if result_freq == 'yearly':
        k = np.sqrt(12.0)
        mu, sigma, downside_dev = mu_m * 12.0, sigma_m * k, downside_dev_m * k
    else:
        mu, sigma, downside_dev = mu_m, sigma_m, downside_dev_m

    sharpe = (mu / sigma) if sigma > 0 else np.nan
    sortino = (mu / downside_dev) if downside_dev > 0 else np.nan
    cer = mu - 0.5 * gamma * (sigma ** 2)

    wealth = (1.0 + ret).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    max_drawdown = -dd.min()

    if signal is not None:
        w = pd.Series(signal, index=excess.index).astype(float)
        dw = w.diff().abs().dropna()
        per_period_turnover = dw.mean()
        turnover = per_period_turnover * (12.0 if result_freq == 'yearly' else 1.0)
    else:
        turnover = np.nan

    info_ratio = np.nan
    if benchmark_ret is not None:
        bench = pd.Series(benchmark_ret).reindex(excess.index).dropna()
        act = (ret.reindex(bench.index) - bench).dropna()
        if len(act) >= 2:
            mu_a_m, sig_a_m = act.mean(), act.std(ddof=1)
            if sig_a_m > 0:
                if result_freq == 'yearly':
                    info_ratio = (mu_a_m * 12.0) / (sig_a_m * np.sqrt(12.0))
                else:
                    info_ratio = mu_a_m / sig_a_m

    return mu, sigma, sharpe, sortino, cer, max_drawdown, turnover, info_ratio


def rolling_sharpe(ret, rfree=0.0, min_periods=12, freq='monthly'):
    ret = ret.dropna()
    excess = ret - (rfree.reindex(ret.index) if isinstance(rfree, pd.Series) else rfree)
    sharpe_vals = []
    for i in range(min_periods, len(excess) + 1):
        window = excess.iloc[:i]
        m, s = window.mean(), window.std()
        sr = (m / s * np.sqrt(12)) if (s > 0 and freq == 'monthly') else (m / s if s > 0 else np.nan)
        sharpe_vals.append(sr)
    return pd.Series([np.nan] * (min_periods - 1) + sharpe_vals, index=ret.index)


# =============================================================
# 3. Portfolio Construction
# =============================================================

def signal_portfolio(merge_df, actual_df, init_wealth=1.0):
    """Binary signal portfolio."""
    ret_df = actual_df[['ret', 'Rfree']].dropna()
    port = ret_df.join(merge_df, how='inner')
    rp_cols = [c for c in merge_df.columns if c.endswith('_rp')]

    sig_df = pd.DataFrame(index=port.index)
    sig_df['ret'] = port['ret']
    sig_df['Rfree'] = port['Rfree']

    for col in rp_cols:
        signal = (port[col] > 0).astype(int)
        wealth = [init_wealth]
        for i in range(len(port)):
            r = port.iloc[i]['ret'] if signal.iloc[i] == 1 else port.iloc[i]['Rfree']
            wealth.append(wealth[-1] * (1 + r))
        strat = col.replace('_rp', '')
        sig_df[f'{strat}_signal'] = signal.values
        sig_df[f'{strat}_wealth'] = wealth[1:]

    for c in [c for c in sig_df.columns if c.endswith('_wealth')]:
        rc = c.replace('_wealth', '_ret')
        w = sig_df[c]
        r = w.pct_change()
        r.iloc[0] = w.iloc[0] / init_wealth - 1.0
        sig_df[rc] = r

    sig_df['BH_wealth'] = init_wealth * (1 + sig_df['ret']).cumprod()
    sig_df['BH_ret'] = sig_df['BH_wealth'].pct_change()
    sig_df.loc[sig_df.index[0], 'BH_ret'] = sig_df['BH_wealth'].iloc[0] / init_wealth - 1.0
    return sig_df


def mv_portfolio(merge_df, actual_df, gamma=3.0, max_leverage=1.5,
                 min_leverage=0.0, init_wealth=1.0):
    """Constrained mean-variance portfolio."""
    ret_df = actual_df[['ret', 'Rfree']].dropna()
    port = ret_df.join(merge_df, how='inner')
    rp_cols = [c for c in merge_df.columns if c.endswith('_rp')]

    mv_df = pd.DataFrame(index=port.index)
    mv_df['ret'] = port['ret']
    mv_df['Rfree'] = port['Rfree']

    for col in rp_cols:
        vol_col = col.replace('_rp', '_vol')
        if vol_col not in port.columns:
            continue
        pred_rp = port[col]
        pred_vol = port[vol_col]
        weights = np.clip(pred_rp / (gamma * pred_vol), min_leverage, max_leverage)
        portf_ret = weights * port['ret'] + (1 - weights) * port['Rfree']
        wealth = init_wealth * (1 + portf_ret).cumprod()
        strat = col.replace('_rp', '')
        mv_df[f'{strat}_weight'] = weights
        mv_df[f'{strat}_wealth'] = wealth

    for c in [c for c in mv_df.columns if c.endswith('_wealth')]:
        rc = c.replace('_wealth', '_ret')
        w = mv_df[c]
        r = w.pct_change()
        r.iloc[0] = w.iloc[0] / init_wealth - 1.0
        mv_df[rc] = r

    mv_df['BH_wealth'] = init_wealth * (1 + mv_df['ret']).cumprod()
    mv_df['BH_ret'] = mv_df['BH_wealth'].pct_change()
    mv_df.loc[mv_df.index[0], 'BH_ret'] = mv_df['BH_wealth'].iloc[0] / init_wealth - 1.0
    return mv_df


# =============================================================
# 4. Portfolio Evaluation
# =============================================================

def evaluation_portfolio(portfolio_df, port_type='mv', eval_freq='yearly',
                         gamma=3.0, ir_benchmark=None):
    """포트폴리오 성과 요약 테이블."""
    strategies = [c.replace('_ret', '') for c in portfolio_df.columns
                  if c.endswith('_ret') and c != 'BH_ret']

    ir_bench = None
    if ir_benchmark and f'{ir_benchmark}_ret' in portfolio_df.columns:
        ir_bench = portfolio_df[f'{ir_benchmark}_ret']

    summary = []
    for strat in strategies:
        ret = portfolio_df[f'{strat}_ret'].dropna()
        rfree = portfolio_df['Rfree'].reindex(ret.index)
        w_col = f'{strat}_weight' if port_type == 'mv' else f'{strat}_signal'
        weight = portfolio_df[w_col].reindex(ret.index) if w_col in portfolio_df.columns else None
        bench = ir_bench.reindex(ret.index) if (ir_bench is not None and strat != ir_benchmark) else None

        metrics = performance_metrics(ret, rfree, signal=weight,
                                      benchmark_ret=bench, gamma=gamma,
                                      result_freq=eval_freq)
        summary.append([strat, *metrics])

    cols = ['Strategy', 'Mean Excess Return', 'Volatility', 'Sharpe Ratio',
            'Sortino Ratio', f'CER (gamma={gamma})', 'Max Drawdown', 'Turnover',
            f'Information Ratio (vs {ir_benchmark or "N/A"})']
    summary_df = pd.DataFrame(summary, columns=cols)

    bh_metrics = performance_metrics(
        portfolio_df['BH_ret'].dropna(),
        portfolio_df['Rfree'].reindex(portfolio_df['BH_ret'].dropna().index),
        benchmark_ret=ir_bench.reindex(portfolio_df['BH_ret'].dropna().index) if ir_bench is not None else None,
        gamma=gamma, result_freq=eval_freq)
    summary_df = pd.concat([
        pd.DataFrame([['Buy and Hold', *bh_metrics]], columns=cols),
        summary_df
    ], ignore_index=True)
    return summary_df.round(6)


def _apply_tx_costs(ret, weight, cost_bps):
    w = pd.Series(weight).reindex(ret.index)
    cost = (float(cost_bps) / 10000.0) * w.diff().abs().fillna(0.0)
    return (ret - cost).astype(float)


def evaluation_portfolio_with_costs(portfolio_df, port_type='mv', eval_freq='yearly',
                                    gamma=3.0, cost_bps_list=TX_COST_BPS_LIST,
                                    ir_benchmark=None):
    """거래비용 시나리오별 포트폴리오 성과."""
    strategies = [c.replace('_ret', '') for c in portfolio_df.columns
                  if c.endswith('_ret') and c != 'BH_ret']

    out_rows = []
    for cost_bps in cost_bps_list:
        ir_bench_net = None
        if ir_benchmark and f'{ir_benchmark}_ret' in portfolio_df.columns:
            w_key = f'{ir_benchmark}_weight' if port_type == 'mv' else f'{ir_benchmark}_signal'
            if w_key in portfolio_df.columns:
                ir_bench_net = _apply_tx_costs(portfolio_df[f'{ir_benchmark}_ret'],
                                               portfolio_df[w_key], cost_bps)

        for strat in strategies:
            ret = portfolio_df[f'{strat}_ret'].dropna()
            rfree = portfolio_df['Rfree'].reindex(ret.index)
            w_col = f'{strat}_weight' if port_type == 'mv' else f'{strat}_signal'
            if w_col not in portfolio_df.columns:
                continue
            w = portfolio_df[w_col].reindex(ret.index)
            net_ret = _apply_tx_costs(ret, w, cost_bps)

            bench = None
            if ir_bench_net is not None and strat != ir_benchmark:
                bench = ir_bench_net.reindex(net_ret.index)

            metrics = performance_metrics(net_ret, rfree, signal=w,
                                          benchmark_ret=bench, gamma=gamma,
                                          result_freq=eval_freq)
            out_rows.append({
                'cost_bps': cost_bps, 'Strategy': strat,
                'Mean Excess Return': metrics[0], 'Volatility': metrics[1],
                'Sharpe Ratio': metrics[2], 'Sortino Ratio': metrics[3],
                f'CER (gamma={gamma})': metrics[4], 'Max Drawdown': metrics[5],
                'Turnover': metrics[6],
                f'Information Ratio (vs {ir_benchmark or "N/A"})': metrics[7],
            })

    return pd.DataFrame(out_rows)


# =============================================================
# 5. SR* Bound
# =============================================================

def sr_star(sr_per_period, r2_oos, eps=1e-12):
    sr = float(sr_per_period)
    r2 = pd.Series(r2_oos).astype(float).clip(upper=1 - eps)
    sr_star_vals = np.sqrt((sr ** 2 + r2) / (1.0 - r2))
    return pd.DataFrame({'SR_star': sr_star_vals, 'Delta_SR': sr_star_vals - sr},
                        index=r2.index)


# =============================================================
# 6. Forecast Evaluation Utilities
# =============================================================

def _hac_tstat_mean(x, maxlags=0):
    x = pd.Series(x).dropna()
    if len(x) < 3:
        return np.nan, np.nan, np.nan, int(len(x))
    X = np.ones((len(x), 1))
    res = sm.OLS(x.values, X).fit(cov_type='HAC', cov_kwds={'maxlags': int(maxlags)})
    return float(res.params[0]), float(res.bse[0]), float(res.tvalues[0]), int(len(x))


def _p_value(t, alternative='two-sided'):
    if np.isnan(t):
        return np.nan
    if alternative == 'two-sided':
        return 2.0 * (1.0 - norm.cdf(abs(t)))
    if alternative == 'greater':
        return 1.0 - norm.cdf(t)
    if alternative == 'less':
        return norm.cdf(t)
    raise ValueError(f"Unknown alternative: {alternative}")


def dm_test_hac(y, f1, f2, loss='mse', maxlags=0):
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
    return {'n': n, 'mean_d': mean_d, 't': t,
            'p_two': _p_value(t, 'two-sided'), 'p_f1_better': _p_value(t, 'less')}


def clark_west_hac(y, f_model, f_bench, maxlags=0):
    df = pd.concat({'y': y, 'fm': f_model, 'fb': f_bench}, axis=1).dropna()
    if df.empty:
        return {'n': 0, 'mean_d': np.nan, 't': np.nan, 'p_one': np.nan}
    e_b, e_m = df['y'] - df['fb'], df['y'] - df['fm']
    d = e_b ** 2 - (e_m ** 2 - (df['fm'] - df['fb']) ** 2)
    mean_d, _, t, n = _hac_tstat_mean(d, maxlags=maxlags)
    return {'n': n, 'mean_d': mean_d, 't': t, 'p_one': _p_value(t, 'greater')}


def forecast_metrics(y, yhat, bench):
    df = pd.concat({'y': y, 'yhat': yhat, 'bench': bench}, axis=1).dropna()
    if df.empty:
        return {k: np.nan for k in ['n', 'R2_oos', 'RMSE_pred', 'MAE_pred',
                                     'RRMSE', 'MASE', 'RMSE_bench', 'MAE_bench']}

    err_m = df['y'] - df['yhat']
    err_b = df['y'] - df['bench']
    mse_m, mse_b = float((err_m ** 2).mean()), float((err_b ** 2).mean())
    rmse_m, rmse_b = np.sqrt(mse_m), np.sqrt(mse_b)
    mae_m, mae_b = float(err_m.abs().mean()), float(err_b.abs().mean())

    r2_oos = 1.0 - (mse_m / mse_b) if mse_b > 0 else np.nan
    rrmse = float(rmse_m / rmse_b) if rmse_b > 0 else np.nan
    mase_scale = float(df['y'].diff().abs().dropna().mean())
    mase = float(mae_m / mase_scale) if mase_scale > 0 else np.nan

    return {
        'n': int(len(df)), 'R2_oos': float(r2_oos),
        'RMSE_pred': float(rmse_m), 'MAE_pred': float(mae_m),
        'RRMSE': rrmse, 'MASE': mase,
        'RMSE_bench': float(rmse_b), 'MAE_bench': float(mae_b),
    }


def oos_r2_timeseries(y, yhat, bench, roll_window=36, min_periods=6):
    df = pd.concat({'y': y, 'yhat': yhat, 'bench': bench}, axis=1).dropna()
    err_f = (df['y'] - df['yhat']) ** 2
    err_b = (df['y'] - df['bench']) ** 2
    r2_cum = 1.0 - err_f.cumsum() / err_b.cumsum()
    r2_roll = 1.0 - (err_f.rolling(roll_window, min_periods=min_periods).sum() /
                      err_b.rolling(roll_window, min_periods=min_periods).sum())
    return pd.DataFrame({'R2_oos_cum': r2_cum, 'R2_oos_roll': r2_roll})


def hit_rate(y, yhat):
    df = pd.concat({'y': y, 'yhat': yhat}, axis=1).dropna()
    return float((np.sign(df['y']) == np.sign(df['yhat'])).mean()) if not df.empty else np.nan


def conditional_metrics(y, yhat, bench, quants=TAIL_QUANTILES, side='down'):
    base = pd.concat({'y': y, 'yhat': yhat, 'bench': bench}, axis=1).dropna()
    out = []
    for q in quants:
        if side == 'down':
            thr = base['y'].quantile(q)
            sub = base[base['y'] <= thr]
        else:
            thr = base['y'].quantile(1 - q)
            sub = base[base['y'] >= thr]
        if sub.empty:
            continue

        err_p, err_b = sub['y'] - sub['yhat'], sub['y'] - sub['bench']
        mse_p, mse_b = float((err_p ** 2).mean()), float((err_b ** 2).mean())
        rmse_p, rmse_b = np.sqrt(mse_p), np.sqrt(mse_b)
        mae_p, mae_b = float(err_p.abs().mean()), float(err_b.abs().mean())
        r2_cond = 1.0 - (mse_p / mse_b) if mse_b > 0 else np.nan
        rrmse_cond = float(rmse_p / rmse_b) if rmse_b > 0 else np.nan
        mase_cond = float(mae_p / mae_b) if mae_b > 0 else np.nan
        hr = float((np.sign(sub['y']) == np.sign(sub['yhat'])).mean())

        out.append({
            'side': side, 'q': float(q), 'threshold': float(thr), 'n': int(len(sub)),
            'R2_cond': float(r2_cond), 'RRMSE_cond': rrmse_cond, 'MASE_cond': mase_cond,
            'RMSE_pred': float(rmse_p), 'RMSE_bench': float(rmse_b),
            'MAE_pred': float(mae_p), 'MAE_bench': float(mae_b),
            'hit_rate_cond': hr,
        })

    return pd.DataFrame(out) if out else pd.DataFrame()

# =============================================================
# 7. Paper Tables
# =============================================================

def _parse_strategy_name(name):
    """strategy name -> (method, spec, tau).

    Examples:
        'past'            -> ('none', 'Past', None)
        'dual_0.05'       -> ('none', 'Dual', 0.05)
        'past_pca'        -> ('pca',  'Past', None)
        'dual_pca_0.1'    -> ('pca',  'Dual', 0.1)
        'past_pca_shap'   -> ('shap_pca', 'Past', None)
        'dual_pca_shap_0' -> ('shap_pca', 'Dual', 0.0)
    """
    suffix_to_method = {
        '': 'none', '_pca': 'pca', '_pls': 'pls',
        '_pca_shap': 'shap_pca', '_pls_shap': 'shap_pls',
    }
    if name.startswith('past'):
        suffix = name[4:]
        method = suffix_to_method.get(suffix, suffix)
        return method, 'Past', None
    elif name.startswith('dual'):
        rest = name[4:]           # e.g. '_pca_shap_0.05'
        parts = rest.rsplit('_', 1)
        try:
            tau = float(parts[-1])
            suffix = parts[0] if len(parts) > 1 else ''
        except ValueError:
            return name, 'Dual', None
        method = suffix_to_method.get(suffix, suffix)
        return method, 'Dual', tau
    return name, '?', None


def build_table_vs_benchmark(forecast_df, tests_df):
    """Table X: Forecast significance against the historical mean benchmark.

    DM_t 부호 반전: 양수 = 모델이 benchmark보다 좋음 (CW_t와 방향 통일).
    Columns: Method, Spec, tau, n, R2_OS, DM_t, DM_p_one_sided, CW_t, CW_p_one_sided
    """
    joined = forecast_df[['R2_oos']].join(
        tests_df[['n', 'DM_t_mse', 'DM_p_model_better_mse', 'CW_t', 'CW_p_one']])

    rows = []
    for name in joined.index:
        method, spec, tau = _parse_strategy_name(name)
        rows.append({
            'Method': method,
            'Spec': spec,
            'tau': tau if tau is not None else np.nan,
            'n': joined.loc[name, 'n'],
            'R2_OS': joined.loc[name, 'R2_oos'],
            'DM_t': -joined.loc[name, 'DM_t_mse'],
            'DM_p_one_sided': joined.loc[name, 'DM_p_model_better_mse'],
            'CW_t': joined.loc[name, 'CW_t'],
            'CW_p_one_sided': joined.loc[name, 'CW_p_one'],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    method_order = ['none', 'pca', 'pls', 'shap_pca', 'shap_pls']
    spec_order = ['Past', 'Dual']
    df['_m'] = df['Method'].map({m: i for i, m in enumerate(method_order)})
    df['_s'] = df['Spec'].map({s: i for i, s in enumerate(spec_order)})
    df = df.sort_values(['_m', '_s', 'tau']).drop(columns=['_m', '_s']).reset_index(drop=True)
    return df


def build_table_dual_vs_past(port_df, strategy_names, maxlags=0):
    """Table Y: Incremental value of forward-looking signals — matched dual vs past.

    각 method 안에서 dual(tau) vs past를 DM test로 비교.
    R2_OS는 dual과 past의 공통 sample(matched index)에서 재계산.
    DM_t 부호 반전: 양수 = dual이 past보다 좋음.
    Columns: Method, tau, n, Past_R2_OS, Dual_R2_OS, Delta_R2_OS, DM_t, DM_p_one_sided
    """
    # Parse all strategies
    parsed = {}
    for name in strategy_names:
        method, spec, tau = _parse_strategy_name(name)
        parsed[name] = (method, spec, tau)

    # past strategy per method
    past_map = {}
    for name, (method, spec, tau) in parsed.items():
        if spec == 'Past':
            past_map[method] = name

    rows = []
    for name, (method, spec, tau) in parsed.items():
        if spec != 'Dual':
            continue
        past_name = past_map.get(method)
        if past_name is None:
            continue

        dual_col = f'{name}_rp'
        past_col = f'{past_name}_rp'
        if dual_col not in port_df.columns or past_col not in port_df.columns:
            continue

        idx = port_df[[dual_col, past_col, 'actual_rp', 'bench']].dropna().index
        if len(idx) < 12:
            continue

        y = port_df.loc[idx, 'actual_rp']
        b = port_df.loc[idx, 'bench']
        f_dual = port_df.loc[idx, dual_col]
        f_past = port_df.loc[idx, past_col]

        dm = dm_test_hac(y, f_dual, f_past, loss='mse', maxlags=maxlags)

        dual_r2 = forecast_metrics(y, f_dual, b)['R2_oos']
        past_r2 = forecast_metrics(y, f_past, b)['R2_oos']

        rows.append({
            'Method': method,
            'tau': tau,
            'n': len(idx),
            'Past_R2_OS': past_r2,
            'Dual_R2_OS': dual_r2,
            'Delta_R2_OS': (dual_r2 - past_r2) if (pd.notna(dual_r2) and pd.notna(past_r2)) else np.nan,
            'DM_t': -dm['t'],
            'DM_p_one_sided': dm['p_f1_better'],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    method_order = ['none', 'pca', 'pls', 'shap_pca', 'shap_pls']
    df['_m'] = df['Method'].map({m: i for i, m in enumerate(method_order)})
    df = df.sort_values(['_m', 'tau']).drop(columns=['_m']).reset_index(drop=True)
    return df


# =============================================================
# 8. Main Execution
# =============================================================

def main():
    parser = argparse.ArgumentParser(description='Portfolio & Forecast Evaluation (v2)')
    parser.add_argument('--index_type', type=str, default=DEFAULT_INDEX_TYPE,
                        choices=['sp500', 'sp500_short', 'russell3000', 'crsp_index'])
    parser.add_argument('--no_sigma', action='store_true',
                        help='output(non-sigma) 디렉토리에서 결과 로딩')
    parser.add_argument('--vol_spec', type=str, default=DEFAULT_VOL_SPEC,
                        help='변동성 스펙: best_vol_var | rolling_var | GARCH | GJR | ...')
    parser.add_argument('--no_portfolio', action='store_true',
                        help='포트폴리오 평가 스킵')
    args = parser.parse_args()

    index_type = args.index_type
    use_sigma = not args.no_sigma
    vol_spec = args.vol_spec
    run_portfolio = not args.no_portfolio and RUN_PORTFOLIO_EVAL

    base = 'output' if use_sigma else 'output(non-sigma)'
    if index_type in ('sp500', 'sp500_short'):
        output_root = base
    else:
        output_root = f'{base}({index_type})'
        
    final_dir = os.path.join(output_root, f'final_{index_type}')
    port_dir = os.path.join(output_root, f'portfolio_{index_type}')
    os.makedirs(port_dir, exist_ok=True)

    print(f'\n{"=" * 70}')
    print(f'  Analysis')
    print(f'  index_type  : {index_type}')
    print(f'  use_sigma   : {use_sigma}')
    print(f'  vol_spec    : {vol_spec}')
    print(f'  final_dir   : {final_dir}')
    print(f'  port_dir    : {port_dir}')
    print(f'  portfolio   : {run_portfolio}')
    print(f'{"=" * 70}\n')

    # ---- Load ----
    merge_df, actual_df, v1_metrics = load_all(
        final_dir, BENCHMARK_DIR, DATA_DIR, index_type,
        vol_spec=vol_spec, benchmark_files=BENCHMARK_FILES)

    strategy_rp_cols = [c for c in merge_df.columns if c.endswith('_rp')]
    strategy_names = [c.replace('_rp', '') for c in strategy_rp_cols]

    print(f'Loaded {len(strategy_names)} strategies: {strategy_names}')
    print(f'Date range: {merge_df.index.min()} ~ {merge_df.index.max()}')

    if not v1_metrics.empty:
        v1_metrics.to_csv(os.path.join(port_dir, 'v1_metrics_summary.csv'), index=False)

    # ---- Build evaluation dataset ----
    port_df = actual_df[['actual_rp']].join(merge_df, how='left')
    port_df['bench'] = port_df['actual_rp'].expanding().mean().shift(1)

    # ---- (1) Overall forecast metrics + DM/CW tests ----
    forecast_rows, test_rows = [], []
    for name in strategy_names:
        col = f'{name}_rp'
        idx = port_df[col].dropna().index
        if len(idx) < 12:
            continue
        y = port_df.loc[idx, 'actual_rp']
        b = port_df.loc[idx, 'bench']
        yh = port_df.loc[idx, col]

        m = forecast_metrics(y, yh, b)
        m['strategy'] = name
        forecast_rows.append(m)

        dm_mse = dm_test_hac(y, yh, b, loss='mse', maxlags=DM_HAC_MAXLAGS)
        dm_mae = dm_test_hac(y, yh, b, loss='mae', maxlags=DM_HAC_MAXLAGS)
        cw = clark_west_hac(y, yh, b, maxlags=CW_HAC_MAXLAGS)
        test_rows.append({
            'strategy': name, 'n': dm_mse['n'],
            'DM_t_mse': dm_mse['t'], 'DM_p_two_mse': dm_mse['p_two'],
            'DM_p_model_better_mse': dm_mse['p_f1_better'],
            'DM_t_mae': dm_mae['t'], 'DM_p_two_mae': dm_mae['p_two'],
            'DM_p_model_better_mae': dm_mae['p_f1_better'],
            'CW_t': cw['t'], 'CW_p_one': cw['p_one'],
        })

    forecast_df = pd.DataFrame(forecast_rows).set_index('strategy').sort_index()
    tests_df = pd.DataFrame(test_rows).set_index('strategy').sort_index()
    forecast_df.to_csv(os.path.join(port_dir, 'forecast_metrics_overall.csv'))
    tests_df.to_csv(os.path.join(port_dir, 'forecast_tests_vs_bench.csv'))

    # ---- Paper tables ----
    paper_strategy_names = [s for s in strategy_names if s not in PAPER_EXCLUDE]

    table_x = build_table_vs_benchmark(
        forecast_df.loc[paper_strategy_names],
        tests_df.loc[paper_strategy_names])
    table_x.to_csv(os.path.join(port_dir, 'table_dm_vs_benchmark.csv'), index=False)
    print(f'\n[Table X] DM/CW vs benchmark: {len(table_x)} rows -> table_dm_vs_benchmark.csv')

    table_y = build_table_dual_vs_past(port_df, paper_strategy_names,
                                        maxlags=DM_HAC_MAXLAGS)
    table_y.to_csv(os.path.join(port_dir, 'table_dm_dual_vs_past.csv'), index=False)
    print(f'[Table Y] Dual vs Past DM: {len(table_y)} rows -> table_dm_dual_vs_past.csv')

    # ---- HAC lag sensitivity (lag 0, 1, 3) ----
    sens_x_parts, sens_y_parts = [], []
    for lag in HAC_SENSITIVITY_LAGS:
        # Table X sensitivity: re-run DM/CW vs benchmark with different lag
        sens_test_rows = []
        for name in paper_strategy_names:
            col = f'{name}_rp'
            idx = port_df[col].dropna().index
            if len(idx) < 12:
                continue
            y = port_df.loc[idx, 'actual_rp']
            b = port_df.loc[idx, 'bench']
            yh = port_df.loc[idx, col]
            dm_s = dm_test_hac(y, yh, b, loss='mse', maxlags=lag)
            cw_s = clark_west_hac(y, yh, b, maxlags=lag)
            method, spec, tau = _parse_strategy_name(name)
            sens_x_parts.append({
                'lag': lag, 'Method': method, 'Spec': spec,
                'tau': tau if tau is not None else np.nan,
                'DM_t': -dm_s['t'], 'DM_p_one_sided': dm_s['p_f1_better'],
                'CW_t': cw_s['t'], 'CW_p_one_sided': cw_s['p_one'],
            })

        # Table Y sensitivity: re-run dual vs past DM with different lag
        ty = build_table_dual_vs_past(port_df, paper_strategy_names, maxlags=lag)
        if not ty.empty:
            ty.insert(0, 'lag', lag)
            sens_y_parts.append(ty)

    if sens_x_parts:
        sens_x_df = pd.DataFrame(sens_x_parts)
        sens_x_df.to_csv(os.path.join(port_dir, 'table_dm_vs_benchmark_lag_sensitivity.csv'), index=False)
        print(f'[Sensitivity X] HAC lags {list(HAC_SENSITIVITY_LAGS)}: {len(sens_x_df)} rows')

    if sens_y_parts:
        sens_y_df = pd.concat(sens_y_parts, ignore_index=True)
        sens_y_df.to_csv(os.path.join(port_dir, 'table_dm_dual_vs_past_lag_sensitivity.csv'), index=False)
        print(f'[Sensitivity Y] HAC lags {list(HAC_SENSITIVITY_LAGS)}: {len(sens_y_df)} rows')

    print('\n=== Overall Forecast Metrics (preview) ===')
    print(forecast_df[['R2_oos', 'RMSE_pred', 'MAE_pred', 'RRMSE', 'MASE']].round(6))

    # ---- SR* bound ----
    rp_full = actual_df['actual_rp'].dropna()
    mu_rp, sd_rp = rp_full.mean(), rp_full.std(ddof=1)
    SR_monthly = mu_rp / sd_rp if sd_rp > 0 else np.nan

    if 'R2_oos' in forecast_df.columns:
        sr_table = sr_star(SR_monthly, forecast_df['R2_oos'])
        sr_results = forecast_df[['R2_oos']].join(sr_table)
        sr_results.to_csv(os.path.join(port_dir, 'SR_bound.csv'))
        print(f'\nFull-sample SR (annualized): {np.sqrt(12) * SR_monthly:.4f}')

    # ---- (2) Cumulative / rolling OOS R2 ----
    r2_panels = {}
    for name in strategy_names:
        col = f'{name}_rp'
        idx = port_df[col].dropna().index
        if len(idx) < 12:
            continue
        r2_panels[name] = oos_r2_timeseries(
            port_df.loc[idx, 'actual_rp'], port_df.loc[idx, col],
            port_df.loc[idx, 'bench'], roll_window=36, min_periods=12)

    if r2_panels:
        r2_big = pd.concat(r2_panels, axis=1).sort_index()
        r2_big.to_csv(os.path.join(port_dir, 'oos_r2_timeseries.csv'))

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        for name in r2_panels:
            axes[0].plot(r2_panels[name].index, r2_panels[name]['R2_oos_cum'], label=name)
            axes[1].plot(r2_panels[name].index, r2_panels[name]['R2_oos_roll'], label=name)
        for ax, title in zip(axes, ['Cumulative OOS $R^2$', 'Rolling OOS $R^2$ (36m)']):
            ax.axhline(0, ls='--', c='grey')
            ax.set_title(title)
            ax.legend(loc='best', fontsize=6)
        plt.tight_layout()
        fig.savefig(os.path.join(port_dir, 'oos_r2_plots.png'), dpi=200)
        plt.close(fig)

    # ---- (3) Hit rates + conditional metrics ----
    hit_rows = []
    cond_frames = []
    for name in strategy_names:
        col = f'{name}_rp'
        idx = port_df[col].dropna().index
        if len(idx) < 12:
            continue
        y = port_df.loc[idx, 'actual_rp']
        b = port_df.loc[idx, 'bench']
        yh = port_df.loc[idx, col]

        row = {'strategy': name, 'hit_rate_all': hit_rate(y, yh)}
        for q in TAIL_QUANTILES:
            thr_dn, thr_up = y.quantile(q), y.quantile(1 - q)
            dn_idx = y[y <= thr_dn].index
            up_idx = y[y >= thr_up].index
            row[f'hit_dn_{int(q*100)}%'] = hit_rate(y.loc[dn_idx], yh.loc[dn_idx])
            row[f'hit_up_{int(q*100)}%'] = hit_rate(y.loc[up_idx], yh.loc[up_idx])
        hit_rows.append(row)

        dn = conditional_metrics(y, yh, b, TAIL_QUANTILES, 'down')
        up = conditional_metrics(y, yh, b, TAIL_QUANTILES, 'up')
        df_cond = pd.concat([dn, up], ignore_index=True)
        if not df_cond.empty:
            df_cond.insert(0, 'strategy', name)
            cond_frames.append(df_cond)

    if hit_rows:
        pd.DataFrame(hit_rows).set_index('strategy').to_csv(
            os.path.join(port_dir, 'hit_rates_summary.csv'))
    if cond_frames:
        pd.concat(cond_frames, ignore_index=True).to_csv(
            os.path.join(port_dir, 'conditional_metrics_down_up.csv'), index=False)

    # ---- (4) Portfolio evaluation ----
    if run_portfolio:
        print('\n=== Portfolio Evaluation ===')
        try:
            mv_df = mv_portfolio(merge_df, actual_df, gamma=MV_GAMMA,
                                 max_leverage=MV_MAX_LEVERAGE, min_leverage=MV_MIN_LEVERAGE)
            mv_summary = evaluation_portfolio(mv_df, port_type='mv', eval_freq='yearly',
                                              gamma=MV_GAMMA, ir_benchmark=IR_BENCHMARK)
            mv_summary.to_csv(os.path.join(port_dir, 'mv_portfolio_gross.csv'), index=False)
            print('\n[MV Portfolio - Gross]')
            print(mv_summary.to_string(index=False))

            mv_net = evaluation_portfolio_with_costs(mv_df, port_type='mv', eval_freq='yearly',
                                                     gamma=MV_GAMMA, cost_bps_list=TX_COST_BPS_LIST,
                                                     ir_benchmark=IR_BENCHMARK)
            mv_net.to_csv(os.path.join(port_dir, 'mv_portfolio_net_of_costs.csv'), index=False)

            sig_df = signal_portfolio(merge_df, actual_df)
            sig_summary = evaluation_portfolio(sig_df, port_type='signal', eval_freq='yearly',
                                               gamma=MV_GAMMA, ir_benchmark=IR_BENCHMARK)
            sig_summary.to_csv(os.path.join(port_dir, 'signal_portfolio_gross.csv'), index=False)
            print('\n[Signal Portfolio - Gross]')
            print(sig_summary.to_string(index=False))

            sig_net = evaluation_portfolio_with_costs(sig_df, port_type='signal', eval_freq='yearly',
                                                      gamma=MV_GAMMA, cost_bps_list=TX_COST_BPS_LIST,
                                                      ir_benchmark=IR_BENCHMARK)
            sig_net.to_csv(os.path.join(port_dir, 'signal_portfolio_net_of_costs.csv'), index=False)

            print(f'\nPortfolio outputs saved to {port_dir}/')

        except Exception as e:
            print(f'\n[Portfolio evaluation] Error: {e}')
            import traceback
            traceback.print_exc()

    # ---- Summary ----
    print(f'\n{"=" * 70}')
    print(f'  All outputs saved to: {port_dir}/')
    saved = [f for f in os.listdir(port_dir) if not f.startswith('.')]
    for f in sorted(saved):
        print(f'    {f}')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()