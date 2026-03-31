"""
통합 RF 실험 시스템 - 데이터 로딩 모듈
ARIMA 예측값/실제값, 인덱스 수익률 데이터를 로딩하고 병합합니다.
"""

import numpy as np
import pandas as pd
import os
import sys
from pandas.tseries.offsets import MonthEnd

from config import (
    M_VARS, Q_VARS, Y_VARS,
    DATA_DIR, ARIMA_RESULT_DIR, INDEX_SETTINGS
)


def convert_yyyyq_to_datetime(x):
    year = x // 10
    quarter = x % 10
    month = {1: 1, 2: 4, 3: 7, 4: 10}[quarter]
    return pd.Timestamp(f"{year}-{month:02d}-01")


def load_features(r2_cut):
    """
    ARIMA 예측 결과를 로딩하여 피처 DataFrame을 생성합니다.

    - OOS R² >= r2_cut인 변수: 예측값(pred) + GARCH 예측 표준편차(std) 사용
    - OOS R² < r2_cut인 변수: 실제값(1기 시프트) 사용

    Args:
        r2_cut: float, R² 기준 컷오프

    Returns:
        pd.DataFrame, date index, 피처 컬럼들
    """
    best_df = pd.read_csv(os.path.join(ARIMA_RESULT_DIR, 'stage1_best_summary.csv'))

    var_list = [
        (var, exo if pd.notna(exo) else 'None', r2)
        for var, exo, r2 in zip(
            best_df['variable'], best_df['exo_vars'], best_df['OOS_R2']
        )
    ]

    start_date = '1972-01-01'
    series_to_merge = []

    for var, exo, r2 in var_list:
        oos_path = os.path.join(ARIMA_RESULT_DIR, f'{var}/OOS({exo}).csv')
        df = pd.read_csv(oos_path)
        df.rename(columns={df.columns[0]: 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.to_period('M').dt.to_timestamp()
        df.set_index('date', inplace=True)

        if r2 < r2_cut:
            # R² 미달 → 과거 실제값 사용 (정보누수 방지를 위해 1기 시프트)
            if var in M_VARS:
                actual_df = pd.read_csv(os.path.join(DATA_DIR, 'monthly.csv'))
                actual_df['date'] = pd.to_datetime(actual_df['date'], format='%Y%m')
            elif var in Q_VARS:
                actual_df = pd.read_csv(os.path.join(DATA_DIR, 'quarterly.csv'))
                actual_df['date'] = actual_df['date'].apply(convert_yyyyq_to_datetime)
            else:
                actual_df = pd.read_csv(os.path.join(DATA_DIR, 'yearly.csv'))
                actual_df['date'] = pd.to_datetime(actual_df['date'], format='%Y')

            actual_df.columns = [c.replace('/', '') for c in actual_df.columns]
            actual_df.set_index('date', inplace=True)
            s_actual = actual_df[var].shift(1)
            s_actual = s_actual[s_actual.index >= start_date]
            s_actual = s_actual.rename(var)
            series_to_merge.append(s_actual)

        else:
            # R² 충족 → ARIMA 예측값 사용
            s_pred = df['pred'] if 'pred' in df.columns else df[var]
            s_pred = s_pred.rename(f'{var}_pred')
            series_to_merge.append(s_pred)

            # GARCH 예측 표준편차
            try:
                garch_path = os.path.join(ARIMA_RESULT_DIR, f'{var}/OOS_GARCH_VAR({exo}).csv')
                garch_df = pd.read_csv(garch_path)
                garch_df.rename(columns={garch_df.columns[0]: 'date'}, inplace=True)
                garch_df['date'] = pd.to_datetime(garch_df['date'])
                garch_df['date'] = garch_df['date'].dt.to_period('M').dt.to_timestamp()
                garch_df.set_index('date', inplace=True)

                s_std = np.sqrt(garch_df['oos_var']).rename(f'{var}_std')
                series_to_merge.append(s_std)
            except Exception as e:
                print(f"[WARN] GARCH std 로딩 실패 ({var}, {exo}): {e}")

    merged_df = pd.concat(series_to_merge, axis=1)
    return merged_df


def prepare_features(feature_type, r2_cut):
    """
    feature_type에 따라 피처 DataFrame을 구성합니다.

    Args:
        feature_type: 'dual' (예측값+실제값) 또는 'past' (실제값만)
        r2_cut: float

    Returns:
        pd.DataFrame
    """
    if feature_type == 'dual':
        pred_set = load_features(r2_cut)
        past_set = load_features(1.0)  # r2_cut=1.0이면 모든 변수가 실제값
        merged_df = pd.merge(past_set, pred_set, left_index=True, right_index=True, how='inner')

        # 중복 컬럼 제거 (_x, _y 처리)
        cols = merged_df.columns
        to_drop = [c for c in cols if c.endswith("_y") and f"{c[:-2]}_x" in cols]
        merged_df = merged_df.drop(columns=to_drop)
        merged_df = merged_df.rename(columns=lambda x: x[:-2] if x.endswith("_x") else x)

    elif feature_type == 'past':
        merged_df = load_features(1.0)

    else:
        sys.exit(f"Error: Unknown feature_type '{feature_type}'.")

    return merged_df


def load_target(index_type):
    """
    인덱스별 리스크 프리미엄(rp) 시리즈를 로딩합니다.

    Args:
        index_type: 'sp500' or 'crsp_index'

    Returns:
        pd.Series, date index, name='rp'
    """
    if index_type == 'sp500':
        tmp = pd.read_csv(os.path.join(DATA_DIR, 'monthly.csv'))
        tmp['date'] = pd.to_datetime(tmp['date'], format='%Y%m')
        tmp['rp'] = tmp['ret'] - tmp['Rfree']
        tmp.set_index('date', inplace=True)
        rp = tmp['rp'].dropna()

    elif index_type == 'crsp_index':
        crsp = pd.read_csv(os.path.join(DATA_DIR, 'crsp_index.csv'))
        crsp.columns = crsp.columns.str.lower()
        crsp['date'] = pd.to_datetime(crsp['date']) + MonthEnd(0) + pd.Timedelta(days=1)
        crsp.set_index('date', inplace=True)
        crsp = crsp.rename(columns={'vwretd': 'ret'})

        tmp = pd.read_csv(os.path.join(DATA_DIR, 'monthly.csv'))
        tmp['date'] = pd.to_datetime(tmp['date'], format='%Y%m')
        tmp.set_index('date', inplace=True)

        rp_df = pd.merge(crsp['ret'], tmp['Rfree'], left_index=True, right_index=True, how='inner')
        rp_df['rp'] = rp_df['ret'] - rp_df['Rfree']
        rp = rp_df['rp'].dropna()

    else:
        sys.exit(f"Error: Unknown index_type '{index_type}'.")

    return rp


def prepare_data(feature_type, index_type, r2_cut):
    """
    피처와 타겟을 병합하여 최종 데이터셋을 생성합니다.

    Returns:
        data: pd.DataFrame (rp 컬럼 + 피처 컬럼들)
        feature_names: list, 전체 피처 이름
        all_vars: list, 표준화 대상 변수 이름 (_std 제외)
    """
    merged_df = prepare_features(feature_type, r2_cut)
    rp = load_target(index_type)

    if index_type == 'sp500':
        data = pd.merge(rp, merged_df, left_index=True, right_index=True, how='right')
    elif index_type == 'crsp_index':
        data = pd.merge(rp, merged_df, left_index=True, right_index=True, how='inner')
    else:
        sys.exit(f"Error: Unknown index_type '{index_type}'.")

    settings = INDEX_SETTINGS[index_type]
    data = data[data.index.year >= settings['start_year']]

    feature_names = list(merged_df.columns)
    all_vars = [col for col in feature_names if not col.endswith('_std')]

    print(f"[INFO] Data prepared: {len(data)} rows, {len(feature_names)} features "
          f"({len(all_vars)} standardized)")

    return data, feature_names, all_vars
