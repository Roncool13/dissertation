# Multimodal Stock Movement Prediction (NSE)
*Dissertation Project – M.Tech (AI/ML)*

## Project Overview
This repository builds a **multimodal prediction system** for short-horizon stock direction on Indian equities (NSE). It combines multiple expert models:

- **OHLCV + technical indicators** (tabular baseline)
- **Candlestick pattern features** (rule-based pattern expert)
- **News sentiment signals** (FinBERT-based daily aggregates)
- **Late-fusion meta-model** that stacks expert probabilities

The pipeline is designed for **reproducibility** (Git + DVC), **traceability** (MLflow + DAGsHub), and **feature-store compatibility** (Feast offline tables).

---

## Repository Structure (Current)
```text
dissertation/
├── data/
│   └── features/                       # DVC pointers to feature-store artifacts
│       ├── ohlcv_features.parquet.dvc
│       ├── ohlcv_feature_metadata.json.dvc
│       ├── news_sentiment_features.parquet.dvc
│       ├── news_sentiment_feature_metadata.json.dvc
│       ├── pattern_features.parquet.dvc
│       └── pattern_feature_metadata.json.dvc
├── feast_repo/                         # Feast feature store config
│   └── feature_store.yaml
├── notebooks/                          # Training notebooks (baselines + fusion)
│   ├── ohlcv_training_tcs_2021_23_grid.ipynb
│   ├── ohlcv_training_grid_MULTISYM_2019_2023_v1_PERSYMNORM.ipynb
│   ├── pattern_training_grid_tcs_2021_23.ipynb
│   ├── pattern_training_grid_MULTISYM_2019_2023.ipynb
│   ├── news_sentiment_training_grid_tcs_2021_23_v3_min_features.ipynb
│   ├── news_sentiment_training_grid_2019_23_min_features_MULTISYM.ipynb
│   ├── fusion_model_training_tcs_2021_23_sent_1lag_gated.ipynb
│   └── fusion_model_training_MULTISYM_2019_23_gated.ipynb
├── src/
│   ├── core/                           # cleaning, transformations, patterns
│   ├── nlp/                            # FinBERT + relevance scoring
│   ├── pipelines/                      # ingestion + feature build pipelines
│   └── io/                             # S3 connectivity helpers
├── Data_Specification_Multimodal_Stocks.md
└── README.md
```

---

## End-to-End Workflow
### 1) Raw ingestion (Desiquant → S3 raw zone)
Mirrors candles, news, corporate announcements, and financial results into raw S3.

**Entry point:**
```
python src/pipelines/run_raw_desiquant_ingestion.py --symbols TCS INFY --s3-bucket <raw-bucket>
```

### 2) Processed ingestion (raw → cleaned parquet)
Normalizes OHLCV and news into canonical schemas and uploads to processed S3.

**Entry point:**
```
python src/pipelines/run_processed_data_ingestion.py --symbol TCS --start-year 2019 --end-year 2023 \
  --s3-raw-bucket <raw-bucket> --s3-processed-bucket <processed-bucket>
```

### 3) Feature build (processed → feature-store parquet)
Builds the model-ready features and metadata used by Feast and ML training.

- **OHLCV features + indicators + labels**
```
python src/pipelines/run_feature_build_ingestion.py --symbols TCS,INFY --start-year 2019 --end-year 2023 \
  --s3-processed-bucket <processed-bucket> --s3-features-bucket <features-bucket>
```

- **News sentiment features (FinBERT + lags)**
```
python src/pipelines/run_news_feature_build_ingestion.py --symbols TCS --start-year 2021 --end-year 2023 \
  --s3-bucket <processed-bucket> --score-mode finbert --lags 5
```

- **Pattern features (candlestick flags + rolling counts)**
```
python src/pipelines/run_pattern_feature_build_ingestion.py --symbols TCS,INFY --start-year 2019 --end-year 2023 \
  --s3-processed-bucket <processed-bucket> --s3-features-bucket <features-bucket>
```

### 4) Training + MLflow logging
Use notebooks in [notebooks/](notebooks/) for:
- Single-symbol (TCS) baselines
- Multi-symbol baselines
- Fusion (stacked) meta-models

Each notebook logs metrics, artifacts, and model versions to MLflow/DAGsHub and registers a model for reuse in stacking.

---

## Streamlit App (Interactive Predictor)
The Streamlit UI in [app/streamlit_app.py](app/streamlit_app.py) provides:
- **Single prediction** for a symbol/date (OHLCV-only or Fusion).
- **Backtest-lite** across a date range with accuracy, coverage, and probability diagnostics.
- **NO-TRADE band** controls around $p=0.5$ plus prediction history tracking.

### App dependencies
Use the pinned app dependencies from [app/requirements.txt](app/requirements.txt).

### Run locally
```
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

### Configuration
The app loads features via DVC (local or remote) and models from MLflow. Configure via environment variables or Streamlit secrets:

**Streamlit secrets (required for DVC S3 access)**
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `ASSUME_ROLE_ARN`

**Optional secrets**
- `AWS_REGION` (default: us-east-1)
- `ASSUME_ROLE_EXTERNAL_ID`

**Environment variables (optional overrides)**
- `MLFLOW_TRACKING_URI`
- `OHLCV_MODEL_URI`
- `FUSION_MODEL_URI`
- `PAT_MODEL_URI`
- `SENT_MODEL_URI`
- `OHLCV_FEATS_PATH`
- `SENT_FEATS_PATH`
- `PAT_FEATS_PATH`
- `LABEL_COL`

The sidebar lets you override model URIs, data paths, thresholds, and the NO-TRADE band in real time.

---

## Experiments & Tracking
- **MLflow** is configured in each notebook (autolog disabled for GridSearchCV).
- **DAGsHub** hosts MLflow experiments and registered models.
- **DVC pointers** in `data/features/` guarantee reproducibility.

---

## Notes on Targets & Splits
- Label column: `y_up_{horizon_days}d` (default horizon = 5 days).
- Metadata (`ohlcv_feature_metadata.json`) defines **global train/val/test splits**, typically by year:
  - Train: start_year → end_year-2
  - Val: end_year-1
  - Test: end_year

See [Data_Specification_Multimodal_Stocks.md](Data_Specification_Multimodal_Stocks.md) for full schema details.

---

## How to Reproduce a Baseline Quickly
1. Pull DVC features locally:
   ```
   dvc pull data/features/ohlcv_features.parquet
   dvc pull data/features/ohlcv_feature_metadata.json
   ```
2. Open a training notebook (e.g., `ohclv_training_tcs_2021_23_grid.ipynb`).
3. Run cells sequentially; MLflow will log metrics and register a model.

---

## License
MIT (see [LICENSE](LICENSE)).






