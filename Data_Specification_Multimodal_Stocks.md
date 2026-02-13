# Data Specification — Multimodal Stock Prediction (Current)
Updated: 2026-02-13

This document describes the **current** data assets used by the dissertation pipeline. All training notebooks pull from **DVC-tracked feature tables** in `data/features/`.

## 1) Universe & Time Range
- **Universe:** NSE symbols defined in [src/constants/symbols.py](src/constants/symbols.py). Typical experiments use TCS-only or multi-symbol subsets (2019–2023).
- **Date Range:** Determined by the feature-build pipeline (`start_year` → `end_year`).
- **Frequency:** Daily bars (`date` normalized to midnight, no timezone).

## 2) Storage Zones (S3 + DVC)
The pipeline uses three logical zones. Local files are stored under `data/` and tracked via DVC for reproducibility.

1. **Raw** (Desiquant mirror)
	- `raw/desiquant/ohlcv/<SYMBOL>/ohlcv_raw.parquet`
	- `raw/desiquant/news/<SYMBOL>/news_raw.parquet`
	- `raw/desiquant/corporate_announcements/<SYMBOL>/<SOURCE>/corporate_announcements_raw.parquet`
	- `raw/desiquant/financial_results/<SYMBOL>/financial_results_raw.parquet`

2. **Processed** (cleaned, canonical schemas)
	- `processed/ohlcv/<SYMBOL>/<YEAR>/ohlcv_processed.parquet`
	- `processed/news/<SYMBOL>/<YEAR>/news_processed.parquet`
	- `processed/corporate_announcements/.../corporate_announcements_processed.parquet`
	- `processed/financial_results/.../financial_results_processed.parquet`

3. **Features** (model-ready, DVC pointers in repo)
	- `data/features/ohlcv_features.parquet`
	- `data/features/ohlcv_feature_metadata.json`
	- `data/features/news_sentiment_features.parquet`
	- `data/features/news_sentiment_feature_metadata.json`
	- `data/features/pattern_features.parquet`
	- `data/features/pattern_feature_metadata.json`

## 3) Feature Tables

### 3.1 OHLCV Features (`ohlcv_features.parquet`)
**Entity keys:** `symbol`, `date`

**Base columns**
- `open`, `high`, `low`, `close`, `volume`

**Indicators (representative)**
- Returns: `ret_1d`, `log_ret_1d`
- Moving averages: `sma_5`, `sma_10`, `sma_20`, `ema_12`, `ema_26`
- Momentum: `macd`, `macd_signal`, `macd_hist`, `rsi_14`
- Volatility: `atr_14`, `volatility_20`
- Candle ranges: `hl_range`, `oc_change`, `oc_change_pct`

**Lagged features**
- `{feature}_lag1`, `{feature}_lag2`, `{feature}_lag3`, `{feature}_lag5` (configurable)

**Labels (forward-looking)**
- `close_fwd_{h}d`, `ret_fwd_{h}d`, `y_up_{h}d` where `h` is `horizon_days` (default 5).

**Metadata** (`ohlcv_feature_metadata.json`)
- `horizon_days`, `lags`, `symbols`, `start_year`, `end_year`
- `splits`: global train/val/test date ranges

### 3.2 News Sentiment Features (`news_sentiment_features.parquet`)
**Entity keys:** `symbol`, `date`

**Inputs**
- Cleaned Desiquant news with relevance scoring + optional FinBERT sentiment.

**Daily aggregates (representative)**
- Article counts, positive/negative/neutral ratios
- Polarity statistics (mean, weighted mean, std)
- Relevance-weighted metrics and `news_present` flag
- Lagged features (`*_lag_1`…`*_lag_5`) and rolling summaries

**Metadata** (`news_sentiment_feature_metadata.json`)
- `supervision.label_col_name` (usually `y_up_5d`)
- `lags`, `target_horizon_days`, optional split ranges

### 3.3 Pattern Features (`pattern_features.parquet`)
**Entity keys:** `symbol`, `date`

**Pattern anatomy**
- `body`, `range`, `upper_wick`, `lower_wick`
- Ratios: `body_to_range`, `upper_wick_to_range`, `lower_wick_to_range`

**Pattern flags (binary)**
- `pattern_doji`, `pattern_hammer`, `pattern_shooting_star`
- `pattern_bullish_engulfing`, `pattern_bearish_engulfing`

**Rolling counts**
- `{pattern}_cnt_{lookback}` per symbol

**Metadata** (`pattern_feature_metadata.json`)
- `lookback`, `symbols`, `start_year`, `end_year`

## 4) Labels & Splits
- **Label:** `y_up_{h}d = 1` if `close_{t+h} > close_t`, else 0.
- **Splits:** Stored in `ohlcv_feature_metadata.json` and applied globally across symbols.
  - Train: `start_year` → `end_year-2`
  - Val: `end_year-1`
  - Test: `end_year`

## 5) Leakage Guards
- Forward-looking columns (`close_fwd_*`, `ret_fwd_*`) are **never** used as features.
- Sentiment features are shifted/lagged to ensure **t+1** leakage is avoided.
- Fusion stacking uses **OOF probabilities** on train and strictly forward predictions on val/test.

## 6) Primary Outputs
- MLflow runs with metrics + artifacts (candidate ranking CSVs, model signatures)
- Registered models for each modality and fusion
- DVC pointers for all feature tables
