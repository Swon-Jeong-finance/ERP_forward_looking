# Equity Premium Forecasting with Reliability-Screened Forward-Looking Signals

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)


Replication code for the two-stage equity risk premium forecasting framework described in:

> Huh, J., Jeon, J., & Jeong, S. (2026). Equity Premium Forecasting with Reliability-Screened Forward-Looking Signals. *PLOS ONE* (under revision).

- **Stage 1:** Predictor-level forecasting (ARIMAX–GARCH) with mixed-frequency alignment and real-time publication lags. The main results use ARIMAX–GARCH.
- 
- **Stage 2:** Equity risk premium forecasting with Random Forest, optional SHAP screening and dimension reduction (PCA / PLS), and forecast evaluation. 

Third-party raw data files are **not distributed**; see [Data](#data) below.

## Repository structure

```text
.
├── stage1/
│   ├── data/                    # local, not distributed
│   ├── exo_arimax.json
│   ├── exo_gpr.json
│   ├── run_stage1.py
│   ├── stage1.py
│   └── results(arima)/1952(fixed order)/ ...
└── stage2/
    ├── data/                    # local, not distributed
    ├── benchmark/               # benchmark portfolio scripts
    ├── output/                  # summary tables (see Output Mapping)
    ├── analysis.py
    ├── collect_vol_qlike.py
    ├── config.py
    ├── data_loader.py
    ├── evaluation.py
    ├── final_config.json
    ├── pipeline.py
    ├── plot_shap_bubble.py
    └── run_experiments.py
```

## Data

Both stages expect a local `data/` directory. Stage 2 additionally reads Stage 1 outputs.

```text
stage1/data/                     stage2/data/
├── monthly.csv                  ├── monthly.csv
├── quarterly.csv                ├── quarterly.csv
└── yearly.csv                   ├── yearly.csv
                                 └── crsp_index.csv  # optional
```

**Sources:**

- `monthly.csv`, `quarterly.csv`, `yearly.csv` — constructed from the publicly available predictor dataset compiled by Welch and Goyal (2008) and updated by Goyal, Welch, and Zafirov (2024). The original data can be obtained from the authors' distribution page.
- `crsp_index.csv` — CRSP value-weighted market index returns. Restricted-access; obtain through an institutional WRDS/CRSP subscription (<https://wrds-www.wharton.upenn.edu/>).

The authors did not have any special access privileges that others would not have.

## Requirements

**Python 3.10+** (uses `X | Y` type union syntax).

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib tqdm threadpoolctl
# optional
```

## Quick start

### Stage 1

```bash
python stage1/run_stage1.py \
  --model arimax \
  --targets default \
  --exo_map stage1/exo_arimax.json \
  --data_dir stage1/data \
  --save_dir "stage1/results(arima)" \
  --start_year 1952 --end_year 2024 \
  --initial_train_years 20 \
  --order_fixed --n_jobs 4
```

### Stage 2

```bash
cd stage2
python run_experiments.py --config final_config.json
```

`final_config.json` specifies a shared hyperparameter grid and a list of experiment combinations (`dim_reduction`, `feature_type`, `index_type`, `r2_cut`, `topN`, `n_components`).

Single-combo and dry-run examples:

```bash
python run_experiments.py \
  --dim_reduction pca --feature_type dual \
  --index_type sp500 --r2_cut 0.05 --n_components 2

python run_experiments.py --config final_config.json --dry_run
```

## Summary outputs

The `output/` directories contain summary-level result files for verification of the main empirical findings reported in the paper. Users with access to the original source data can reconstruct the full workflow and all intermediate outputs using the provided code and configuration.

## References

- Welch, I., & Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. *The Review of Financial Studies*, 21(4), 1455–1508.
- Goyal, A., Welch, I., & Zafirov, A. (2024). A comprehensive 2022 look at the empirical performance of equity premium prediction. *The Review of Financial Studies*, 37(11), 3490–3557.
