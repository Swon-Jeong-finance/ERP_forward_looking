"""
통합 RF 실험 시스템 - 설정 모듈 (v1 revised)

final 모드 전용: 선택한 specification을 실행 + full artifact 저장
"""

import json
import os
from typing import Optional

# ============================================================
# 변수 목록
# ============================================================
M_VARS_ = ['date','price','lty','ltr','tbl','d/p','d/y','e/p','d/e',
            'tms','dfy','dfr','infl','svar','ntis','b/m','tchi','ogap','tail','avgcor']
Q_VARS_ = ['date','cay','i/k']
Y_VARS_ = ['date','eqis','gpce']

LEVEL_VARS_ = ['lty','tbl','d/p','d/y','e/p','d/e','tms','dfy','infl',
               'svar','ntis','b/m','tchi','tail','avgcor','cay','i/k','eqis','ogap']
RET_VARS_ = ['dfr','ltr','gpce']


def strip_slash(lst):
    return [col.replace('/', '') for col in lst]


M_VARS = strip_slash(M_VARS_)
Q_VARS = strip_slash(Q_VARS_)
Y_VARS = strip_slash(Y_VARS_)
LEVEL_VARS = strip_slash(LEVEL_VARS_)
RET_VARS = strip_slash(RET_VARS_)

# ============================================================
# 인덱스별 설정
# ============================================================
INDEX_SETTINGS = {
    'sp500': {
        'start_year': 1973,
        'test_years': list(range(2000, 2025)),
        'valid_window': 9,
    },
    'crsp_index': {
        'start_year': 1973,
        'test_years': list(range(2000, 2025)),
        'valid_window': 9,
    },
}

# ============================================================
# 데이터 경로
# ============================================================
DATA_DIR = 'data'
ARIMA_RESULT_DIR = 'results(arima)/1952(fixed order)/'

# ============================================================
# 기본 모델 / 실험 설정
# ============================================================
DEFAULT_MODEL_PARAMS = {
    'random_state': 12,
}

DEFAULT_PARAM_SPACE = {
    "n_estimators":     [300],
    "max_depth":        [1],
    "max_features":     [0.02, 0.04],
    "min_samples_leaf": [0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.03, 0.05, 0.1],
}

DEFAULT_SCORING = 'neg_mean_squared_error'
DEFAULT_TAIL_Q_LIST = [0.05, 0.10, 0.15, 0.20]


# ============================================================
# 하이퍼파라미터 그리드 생성 (full expansion only)
# ============================================================
def generate_param_grid(param_space=None):
    """전체 param_space를 단일 param_grid dict로 반환 (final용)."""
    if param_space is None:
        param_space = DEFAULT_PARAM_SPACE
    return {k: list(v) for k, v in param_space.items()}


def norm_grid(grid: dict) -> str:
    """param_grid dict → 정규화된 JSON 문자열 (중복 체크 키)."""
    canon = {}
    for k in sorted(grid):
        v = grid[k]
        canon[k] = list(v) if isinstance(v, (list, tuple)) else v
    return json.dumps(canon, sort_keys=True, separators=(',', ':'))


# ============================================================
# 설정 로딩
# ============================================================
def load_config():
    """기본 설정 반환."""
    config = {
        'model': DEFAULT_MODEL_PARAMS.copy(),
        'param_space': {k: list(v) for k, v in DEFAULT_PARAM_SPACE.items()},
        'scoring': DEFAULT_SCORING,
        'tail_q_list': list(DEFAULT_TAIL_Q_LIST),
    }
    config['model'].pop('n_estimators', None)
    if 'n_estimators' not in config['param_space']:
        config['param_space']['n_estimators'] = [100]
    return config


# ============================================================
# 결과 디렉토리 이름 생성
# ============================================================
def get_output_root(use_sigma: bool = True, index_type: str = 'sp500') -> str:
    """final 모드 결과 루트."""
    base = 'output' if use_sigma else 'output(non-sigma)'
    if index_type == 'sp500':
        return base
    return f'{base}({index_type})'
