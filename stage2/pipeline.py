"""
통합 RF 실험 시스템 - 파이프라인 모듈 (v1 revised)

final 모드 전용: 고정 (param_grid, topN, n_components) 셀 코어로
OOS 예측 + GARCH/DM/CW/tail/시계열 저장.
"""

from __future__ import annotations

import json
import gc
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from config import DEFAULT_MODEL_PARAMS, INDEX_SETTINGS, norm_grid
from evaluation import (
    clark_west_hac,
    compute_expanding_mean_benchmark,
    compute_hit_rate,
    compute_in_sample_r2,
    compute_oos_r2,
    conditional_metrics_by_realized_quantiles,
    dm_test_hac,
    forecast_metrics_summary,
    plot_oos_predictions,
    variance_coverage,
    variance_error_metrics,
)


# ============================================================
# Lazy imports
# ============================================================
def _require_shap():
    try:
        import shap  # type: ignore
        return shap
    except Exception as e:
        raise ImportError(
            "SHAP is required for dim_reduction='shap_pca'/'shap_pls'."
        ) from e


# ============================================================
# RF kwargs helper
# ============================================================
def _rf_kwargs(model_params=None, best_params=None):
    kwargs = {}
    if model_params:
        kwargs.update(dict(model_params))
    if best_params:
        kwargs.update(dict(best_params))
    kwargs.pop('n_jobs', None)
    return kwargs


# ============================================================
# Preprocessing helpers
# ============================================================
def split_data(data, year, valid_window):
    train_end = f'{year - valid_window}-12-31'
    val_end = f'{year - 1}-12-31'
    test_start = f'{year}-01-01'
    test_end = f'{year}-12-31'

    X_train = data.loc[:train_end].drop(columns='rp')
    y_train = data.loc[:train_end, 'rp']
    X_val = data.loc[pd.Timestamp(train_end) + pd.Timedelta(days=1): val_end].drop(columns='rp')
    y_val = data.loc[pd.Timestamp(train_end) + pd.Timedelta(days=1): val_end, 'rp']
    X_test = data.loc[test_start:test_end].drop(columns='rp')
    return X_train, X_val, X_test, y_train, y_val


def standardize_and_ffill(X_train, X_val, X_test, all_vars):
    X_train_raw = X_train.copy()
    X_val_raw = X_val.copy()
    X_test_raw = X_test.copy()

    m1 = X_train[all_vars].mean()
    m2 = X_train[all_vars].std().replace(0.0, np.nan)
    for _X in (X_train, X_val, X_test):
        _X[all_vars] = (_X[all_vars] - m1) / m2
    for _X in (X_train, X_val, X_test, X_train_raw, X_val_raw, X_test_raw):
        _X.ffill(inplace=True)
    return X_train, X_val, X_test, X_train_raw, X_val_raw, X_test_raw


def apply_dim_reduction(method, X_std, X_raw, y_train, n_components):
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        col_names = [f'PC{i+1}' for i in range(n_components)]
        train_t = reducer.fit_transform(X_std['train'])
        val_t = reducer.transform(X_std['val'])
        test_t = reducer.transform(X_std['test'])
    elif method == 'pls':
        reducer = PLSRegression(n_components=n_components, scale=True)
        col_names = [f'PLS_C{i+1}' for i in range(n_components)]
        train_t = reducer.fit_transform(X_raw['train'], y_train)[0]
        val_t = reducer.transform(X_raw['val'])
        test_t = reducer.transform(X_raw['test'])
    else:
        raise ValueError(f"Unknown dim reduction method: '{method}'")
    return {
        'train': pd.DataFrame(train_t, index=X_std['train'].index, columns=col_names),
        'val':   pd.DataFrame(val_t,   index=X_std['val'].index,   columns=col_names),
        'test':  pd.DataFrame(test_t,  index=X_std['test'].index,  columns=col_names),
    }


def do_grid_search(X_train, X_val, y_train, y_val, param_grid, model_params, scoring):
    base_rf = RandomForestRegressor(n_jobs=1, **_rf_kwargs(model_params))
    X_all = pd.concat([X_train, X_val])
    y_all = pd.concat([y_train, y_val])
    ps = PredefinedSplit([-1] * len(X_train) + [0] * len(X_val))
    grid = GridSearchCV(base_rf, param_grid, cv=ps, scoring=scoring, n_jobs=1)
    grid.fit(X_all, y_all)
    best_params = grid.best_params_
    best_score = float(grid.best_score_)
    del grid
    gc.collect()
    return best_params, X_all, y_all, best_score


def get_shap_top_features(X_train, X_val, y_train, best_params, model_params, topN):
    model = RandomForestRegressor(n_jobs=1, **_rf_kwargs(model_params, best_params))
    model.fit(X_train, y_train)
    shap = _require_shap()
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_val)
    shap_importance = np.abs(shap_vals).mean(axis=0)
    top_feats = (
        pd.Series(shap_importance, index=X_train.columns)
        .nlargest(topN).index.tolist()
    )
    del model, explainer, shap_vals
    gc.collect()
    return top_feats, shap_importance


# ============================================================
# GARCH helpers
# ============================================================
def _default_volatility_specs() -> Dict[str, Dict[str, Any]]:
    return {
        'GARCH':    {'vol': 'GARCH',  'p': 1, 'o': 0, 'q': 1, 'power': 2.0},
        'GJR':      {'vol': 'GARCH',  'p': 1, 'o': 1, 'q': 1, 'power': 2.0},
        'EGARCH':   {'vol': 'EGARCH', 'p': 1, 'o': 1, 'q': 1},
        'APARCH12': {'vol': 'APARCH', 'p': 1, 'o': 1, 'q': 1, 'power': 1.2},
        'APARCH15': {'vol': 'APARCH', 'p': 1, 'o': 1, 'q': 1, 'power': 1.5},
        'APARCH18': {'vol': 'APARCH', 'p': 1, 'o': 1, 'q': 1, 'power': 1.8},
        'APARCH20': {'vol': 'APARCH', 'p': 1, 'o': 1, 'q': 1, 'power': 2.0},
    }


def _fit_forecast_var(epsilon: np.ndarray, spec: Dict[str, Any]) -> float:
    try:
        from arch import arch_model  # type: ignore
    except Exception as e:
        raise ImportError('arch package is required for volatility forecasting.') from e

    eps = pd.Series(epsilon).dropna().astype(float)
    if len(eps) < 24:
        return np.nan
    kwargs = {
        'y': eps, 'mean': 'Zero',
        'vol': spec.get('vol', 'GARCH'),
        'p': int(spec.get('p', 1)),
        'o': int(spec.get('o', 0)),
        'q': int(spec.get('q', 1)),
        'dist': 'normal', 'rescale': False,
    }
    if 'power' in spec:
        kwargs['power'] = float(spec['power'])
    am = arch_model(**kwargs)
    res = am.fit(disp='off', show_warning=False)
    fcst = res.forecast(horizon=1, reindex=False)
    return float(np.clip(fcst.variance.values[-1, 0], 1e-12, None))


# ============================================================
# Fixed-cell core
# ============================================================
def _run_fixed_cell_core(dim_reduction, data, feature_names, all_vars,
                         index_type, param_grid,
                         model_params=None, scoring='neg_mean_squared_error',
                         n_components=None, topN=None, verbose=True):
    """
    고정 (param_grid, topN, n_components)로 전 기간 학습/예측.

    Returns dict:
      all_oos_predictions: list of pd.Series
      all_is_panels: list of pd.DataFrame
      best_params_dict: {year: best_params}
      validation_scores: list of float
      feature_importance_dict: {year: importance}
      shap_val_importance_dict: {year: shap_imp array}
      shap_test_importance_dict: {year: shap_imp Series}
      selected_features_records: list of dict
      best_params_dict_sel: {year: best_params} (SHAP 2nd stage)
    """
    if model_params is None:
        model_params = DEFAULT_MODEL_PARAMS.copy()

    settings = INDEX_SETTINGS[index_type]
    test_years = settings['test_years']
    valid_window = settings['valid_window']

    is_shap = dim_reduction.startswith('shap_')
    dim_method = dim_reduction.replace('shap_', '') if is_shap else dim_reduction

    all_oos_predictions = []
    all_is_panels = []
    best_params_dict = {}
    best_params_dict_sel = {}
    validation_scores = []
    feature_importance_dict = {}
    shap_val_importance_dict = {}
    shap_test_importance_dict = {}
    selected_features_records = []

    for year in test_years:
        if verbose:
            print(f"  [{dim_reduction}] {year}년 학습 (topN={topN}, n_comp={n_components})")

        X_train, X_val, X_test, y_train, y_val = split_data(data, year, valid_window)
        (X_train, X_val, X_test,
         X_train_raw, X_val_raw, X_test_raw) = standardize_and_ffill(X_train, X_val, X_test, all_vars)

        if dim_reduction == 'none':
            best_params, X_all, y_all, best_score = do_grid_search(
                X_train, X_val, y_train, y_val, param_grid, model_params, scoring)
            validation_scores.append(best_score)
            best_params_dict[year] = best_params
            model = RandomForestRegressor(n_jobs=1, **_rf_kwargs(model_params, best_params))
            model.fit(X_all, y_all)
            preds = pd.Series(model.predict(X_test), index=X_test.index)
            is_pred = pd.Series(model.predict(X_all), index=X_all.index)
            feature_importance_dict[year] = model.feature_importances_

        elif dim_reduction in ('pca', 'pls'):
            X_std = {'train': X_train, 'val': X_val, 'test': X_test}
            X_raw = {'train': X_train_raw, 'val': X_val_raw, 'test': X_test_raw}
            t = apply_dim_reduction(dim_method, X_std, X_raw, y_train, n_components)
            best_params, X_all, y_all, best_score = do_grid_search(
                t['train'], t['val'], y_train, y_val, param_grid, model_params, scoring)
            validation_scores.append(best_score)
            best_params_dict[year] = best_params
            model = RandomForestRegressor(n_jobs=1, **_rf_kwargs(model_params, best_params))
            model.fit(X_all, y_all)
            preds = pd.Series(model.predict(t['test']), index=X_test.index)
            is_pred = pd.Series(model.predict(X_all), index=X_all.index)
            feature_importance_dict[year] = model.feature_importances_

        elif is_shap:
            # 1st stage: full feature grid search for SHAP
            best_1st, _, _, _ = do_grid_search(
                X_train, X_val, y_train, y_val, param_grid, model_params, scoring)
            best_params_dict[year] = best_1st

            top_feats, shap_imp = get_shap_top_features(
                X_train, X_val, y_train, best_1st, model_params, topN)
            shap_val_importance_dict[year] = shap_imp

            shap_series = pd.Series(shap_imp, index=feature_names)
            for rank, (feat, imp) in enumerate(shap_series.nlargest(topN).items(), 1):
                selected_features_records.append({
                    'year': year,
                    'rank': rank,
                    'feature': feat,
                    'shap_importance': imp,
                    'selected_topN': topN,
                    'selected_n_components': n_components,
                })
            if verbose:
                print(f"    상위 {topN} 특성: {top_feats}")

            # 2nd stage: reduced feature set
            X_std_sel = {'train': X_train[top_feats], 'val': X_val[top_feats], 'test': X_test[top_feats]}
            X_raw_sel = {'train': X_train_raw[top_feats], 'val': X_val_raw[top_feats], 'test': X_test_raw[top_feats]}

            if n_components is not None and 0 < n_components <= len(top_feats):
                t = apply_dim_reduction(dim_method, X_std_sel, X_raw_sel, y_train, n_components)
                pc_names = list(t['train'].columns)
            else:
                t = X_std_sel
                pc_names = top_feats

            best_2nd, X_all_sel, y_all_sel, best_score = do_grid_search(
                t['train'], t['val'], y_train, y_val, param_grid, model_params, scoring)
            validation_scores.append(best_score)
            best_params_dict_sel[year] = best_2nd

            model_sel = RandomForestRegressor(n_jobs=1, **_rf_kwargs(model_params, best_2nd))
            model_sel.fit(X_all_sel, y_all_sel)
            preds = pd.Series(model_sel.predict(t['test']), index=X_test.index)
            is_pred = pd.Series(model_sel.predict(X_all_sel), index=X_all_sel.index)
            feature_importance_dict[year] = pd.Series(model_sel.feature_importances_, index=pc_names)

            shap_mod = _require_shap()
            exp2 = shap_mod.TreeExplainer(model_sel)
            shap_t = np.abs(exp2.shap_values(t['test'])).mean(axis=0)
            shap_test_importance_dict[year] = pd.Series(shap_t, index=pc_names)
            del model_sel, exp2, shap_t
            gc.collect()

            # For IS panel, use X_all_sel
            X_all = X_all_sel
            y_all = y_all_sel
        else:
            raise ValueError(f"Unknown dim_reduction: '{dim_reduction}'")

        all_oos_predictions.append(preds)
        all_is_panels.append(pd.DataFrame({
            'date': X_all.index,
            'rp_true': y_all.values,
            'rp_pred': is_pred.values,
            'year': year,
        }))

    return {
        'all_oos_predictions': all_oos_predictions,
        'all_is_panels': all_is_panels,
        'best_params_dict': best_params_dict,
        'best_params_dict_sel': best_params_dict_sel,
        'validation_scores': validation_scores,
        'feature_importance_dict': feature_importance_dict,
        'shap_val_importance_dict': shap_val_importance_dict,
        'shap_test_importance_dict': shap_test_importance_dict,
        'selected_features_records': selected_features_records,
    }


# ============================================================
# Public API: final
# ============================================================
def run_final_protocol(dim_reduction, data, feature_names, all_vars,
                       feature_type, index_type, r2_cut, param_grid, output_dir,
                       model_params=None, scoring='neg_mean_squared_error',
                       n_components=None, topN=None,
                       tail_q_list=None, use_sigma=True, verbose=True):
    """
    Final 모드: 고정 셀 코어 + full 평가 + 파일 저장.
    GARCH, DM/CW, tail-conditional metrics 포함.
    """
    if model_params is None:
        model_params = DEFAULT_MODEL_PARAMS.copy()
    if tail_q_list is None:
        tail_q_list = [0.05, 0.10, 0.15, 0.20]

    os.makedirs(output_dir, exist_ok=True)

    # ---- Run core ----
    core = _run_fixed_cell_core(
        dim_reduction=dim_reduction, data=data,
        feature_names=feature_names, all_vars=all_vars,
        index_type=index_type, param_grid=param_grid,
        model_params=model_params, scoring=scoring,
        n_components=n_components, topN=topN, verbose=verbose,
    )

    settings = INDEX_SETTINGS[index_type]
    test_years = settings['test_years']

    final_oos = pd.concat(core['all_oos_predictions']).sort_index()
    oos_dates = final_oos.index
    rp_true = data.loc[oos_dates, 'rp']
    benchmark = compute_expanding_mean_benchmark(data, test_years).loc[oos_dates]

    # ---- IS R² ----
    is_panel = pd.concat(core['all_is_panels'], ignore_index=True)
    is_panel['date'] = pd.to_datetime(is_panel['date'])
    is_r2 = compute_in_sample_r2(is_panel['rp_true'], is_panel['rp_pred'])

    # ---- OOS metrics ----
    mse, r2 = compute_oos_r2(rp_true, final_oos, benchmark)
    overall = forecast_metrics_summary(rp_true, final_oos, benchmark)
    dm_mse = dm_test_hac(rp_true, final_oos, benchmark, loss='mse', maxlags=0)
    dm_mae = dm_test_hac(rp_true, final_oos, benchmark, loss='mae', maxlags=0)
    cw = clark_west_hac(rp_true, final_oos, benchmark, maxlags=0)
    tail_df = conditional_metrics_by_realized_quantiles(rp_true, final_oos, benchmark, tail_q_list)

    val_scores = core['validation_scores']

    # ---- GARCH volatility forecasting ----
    vol_specs = _default_volatility_specs()
    vol_predictions_map: Dict[str, List[pd.Series]] = {name: [] for name in vol_specs}
    vol_failures: Dict[str, Optional[str]] = {name: None for name in vol_specs}

    # Build full IS+OOS prediction series for residuals
    full_is_pred = pd.concat([
        pd.Series(p['rp_pred'].values, index=pd.to_datetime(p['date'].values))
        for p in [is_panel[is_panel['year'] == y] for y in test_years]
    ])
    # deduplicate (overlapping IS windows)
    full_is_pred = full_is_pred[~full_is_pred.index.duplicated(keep='last')]
    full_pred_series = pd.concat([full_is_pred, final_oos]).sort_index()
    full_pred_series = full_pred_series[~full_pred_series.index.duplicated(keep='last')]

    for test_month in oos_dates:
        fit_loc = full_pred_series.index.get_loc(test_month)
        train_idx = full_pred_series.index[:fit_loc]
        ret_hist = data['rp'].reindex(train_idx)
        pred_hist = full_pred_series.reindex(train_idx)
        epsilon = (ret_hist.values - pred_hist.values)
        for name, spec in vol_specs.items():
            if vol_failures.get(name) is not None:
                continue
            try:
                cond_var = _fit_forecast_var(epsilon, spec)
            except Exception as e:
                vol_failures[name] = str(e)
                continue
            vol_predictions_map[name].append(pd.Series(cond_var, index=[test_month]))

    # ---- GARCH evaluation ----
    vol_df = pd.DataFrame({
        name: pd.concat(vals) if vals else pd.Series(dtype=float)
        for name, vals in vol_predictions_map.items()
    }).sort_index()

    vol_summary_rows: List[Dict[str, Any]] = []
    best_vol_model = None
    best_vol_var = None
    rv_proxy = ((rp_true - final_oos).astype(float) ** 2).clip(lower=1e-12)

    for name, failure in vol_failures.items():
        if failure is not None:
            vol_summary_rows.append({
                'vol_model': name, 'avg_QLIKE': np.nan,
                'coverage_95': np.nan, 'var_MSE': np.nan,
                'var_RMSE': np.nan, 'var_MAE': np.nan,
                'is_best_model': False, 'status': f'skipped: {failure}',
            })
    for col in vol_df.columns:
        vm = variance_error_metrics(rv_proxy, vol_df[col].reindex(oos_dates))
        cov = variance_coverage(rp_true, final_oos, vol_df[col].reindex(oos_dates), k=1.96)
        vol_summary_rows.append({
            'vol_model': col, 'avg_QLIKE': vm['avg_QLIKE'],
            'coverage_95': cov, 'var_MSE': vm['var_MSE'],
            'var_RMSE': vm['var_RMSE'], 'var_MAE': vm['var_MAE'],
            'is_best_model': False, 'status': 'ok',
        })
    ok_rows = [r for r in vol_summary_rows if r.get('status') == 'ok' and pd.notna(r.get('avg_QLIKE'))]
    if ok_rows:
        tmp = pd.DataFrame(ok_rows).sort_values(['avg_QLIKE', 'vol_model'], ascending=[True, True])
        best_vol_model = tmp.iloc[0]['vol_model']
        best_vol_var = vol_df[best_vol_model].reindex(oos_dates)
        for row_item in vol_summary_rows:
            if row_item.get('vol_model') == best_vol_model:
                row_item['is_best_model'] = True

    # ---- Build output DataFrames ----
    forecast_oos = pd.DataFrame({
        'date': oos_dates,
        'rp_true': rp_true.values,
        'rp_pred': final_oos.values,
        'bench_pred': benchmark.values,
    })
    if best_vol_var is not None:
        forecast_oos['best_vol_var'] = best_vol_var.values

    metrics_summary = {
        'feature_type': feature_type,
        'index_type': index_type,
        'r2_cut': float(r2_cut),
        'dim_reduction': dim_reduction,
        'topN': topN,
        'n_components': n_components,
        'use_sigma': bool(use_sigma),
        'IS_R2': is_r2,
        'OOS_R2': r2,
        'MSE': mse,
        'RMSE': overall['RMSE'],
        'MAE': overall['MAE'],
        'RRMSE': overall['RRMSE'],
        'MASE': overall['MASE'],
        'hit_rate': overall['hit_rate'],
        'DM_t_mse': dm_mse['t'],
        'DM_p_mse': dm_mse['p_two'],
        'DM_t_mae': dm_mae['t'],
        'DM_p_mae': dm_mae['p_two'],
        'CW_t': cw['t'],
        'CW_p': cw['p_one'],
        'best_vol_model': best_vol_model,
        'best_vol_qlike': (next((r['avg_QLIKE'] for r in vol_summary_rows if r.get('is_best_model')), np.nan)),
        'best_vol_coverage_95': (next((r['coverage_95'] for r in vol_summary_rows if r.get('is_best_model')), np.nan)),
    }

    # ---- Save files ----
    best_params_rows = []
    for y in settings['test_years']:
        bp = core['best_params_dict'].get(y, {})
        bp_sel = core['best_params_dict_sel'].get(y, {})
        row = {'year': y}
        row.update(bp if not bp_sel else bp_sel)
        row['validation_score'] = core['validation_scores'][settings['test_years'].index(y)] if y in settings['test_years'] else np.nan
        best_params_rows.append(row)

    pd.DataFrame(best_params_rows).to_csv(os.path.join(output_dir, 'best_params_by_year.csv'), index=False)
    forecast_oos.to_csv(os.path.join(output_dir, 'forecast_oos.csv'), index=False)
    pd.DataFrame([metrics_summary]).to_csv(os.path.join(output_dir, 'metrics_summary.csv'), index=False)
    tail_df.to_csv(os.path.join(output_dir, 'tail_metrics.csv'), index=False)

    if vol_summary_rows:
        pd.DataFrame(vol_summary_rows).to_csv(os.path.join(output_dir, 'volatility_summary.csv'), index=False)
    if not vol_df.empty:
        vol_df.to_csv(os.path.join(output_dir, 'vol_forecasts.csv'))

    if core['shap_val_importance_dict']:
        pd.DataFrame(
            core['shap_val_importance_dict'], index=feature_names
        ).T.to_csv(os.path.join(output_dir, 'validation_shap_feature_importance.csv'))

    if core['selected_features_records']:
        sf = pd.DataFrame(core['selected_features_records'])
        sf.to_csv(os.path.join(output_dir, 'selected_features_by_year.csv'), index=False)
        freq = sf.groupby('feature').agg(
            selection_count=('year', 'count'),
            avg_rank=('rank', 'mean'),
            avg_shap=('shap_importance', 'mean'),
        ).sort_values(['selection_count', 'avg_rank'], ascending=[False, True])
        freq.to_csv(os.path.join(output_dir, 'selection_frequency.csv'))

        # ---- bundle-level stability summary ----
        sf['bundle'] = sf['feature'].str.replace(r'(_pred|_std)$', '', regex=True)
        sf['component'] = sf['feature'].apply(
            lambda x: 'pred' if x.endswith('_pred')
            else ('std' if x.endswith('_std') else 'level')
        )

        bundle_year = sf.groupby(['bundle', 'year'])['shap_importance'].sum().reset_index()
        bundle_year.rename(columns={'shap_importance': 'S_bt'}, inplace=True)

        bundle_year['R_bt'] = bundle_year.groupby('year')['S_bt'].rank(
            ascending=False, method='min'
        )

        year_totals = bundle_year.groupby('year')['S_bt'].transform('sum')
        bundle_year['P_bt'] = bundle_year['S_bt'] / year_totals.replace(0, np.nan)
        bundle_year['P_bt'] = bundle_year['P_bt'].fillna(0.0)

        n_years = sf['year'].nunique()
        bundle_agg = bundle_year.groupby('bundle').agg(
            years_selected=('year', 'nunique'),
            avg_bundle_rank=('R_bt', 'mean'),
            avg_bundle_share=('P_bt', 'mean'),
            component_hits=('S_bt', 'count'),
        )
        bundle_agg['avg_bundle_share_pct'] = (bundle_agg['avg_bundle_share'] * 100).round(1)
        bundle_agg['stability'] = bundle_agg['years_selected'] / n_years

        comp_total = sf.groupby(['bundle', 'component'])['shap_importance'].sum()
        dominant = comp_total.groupby(level='bundle').idxmax().apply(lambda x: x[1])
        bundle_agg['dominant_component'] = dominant

        bundle_agg = bundle_agg.sort_values(
            ['years_selected', 'avg_bundle_rank'], ascending=[False, True]
        )

        bundle_agg[['years_selected', 'avg_bundle_rank', 'avg_bundle_share_pct',
                     'dominant_component', 'component_hits', 'stability']].to_csv(
            os.path.join(output_dir, 'selection_frequency_bundle.csv')
        )

    if verbose:
        print(f"  [FINAL] OOS R² = {r2:.4f}  saved to {output_dir}")

    return {
        'output_dir': output_dir,
        'metrics_summary': metrics_summary,
        'best_params_by_year': best_params_rows,
        'tail_metrics': tail_df,
        'volatility_summary': vol_summary_rows,
    }
