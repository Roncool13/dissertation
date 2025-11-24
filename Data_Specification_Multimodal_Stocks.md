# Data Specification — Multimodal Stock Prediction
Updated: 2025-11-10

- Universe: RELIANCE.NS, TCS.NS, HDFCBANK.NS, ICICIBANK.NS, INFY.NS, HINDUNILVR.NS
- Period: 2019-01-01 to 2024-12-31 (Daily)

## Files
- data/processed/ohlcv_indicators.parquet — OHLCV + indicators
- data/raw/news_newsapi.csv — Raw headlines (NewsAPI)
- data/processed/news_scored.parquet — Headlines with FinBERT probabilities
- data/processed/sentiment_daily.parquet — Daily sentiment per symbol
- data/processed/patterns.parquet — Candlestick flags + pattern_score
- data/processed/dataset_merged.parquet — Unified dataset

## Target
- ret_1d_fwd: next-day return
- y_up: 1 if ret_1d_fwd > 0 else 0

## Notes
- Use adjusted prices.
- Compute features using only t; target is t+1.
- Deduplicate news by URL/title.
