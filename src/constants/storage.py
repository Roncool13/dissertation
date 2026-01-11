# Default S3 Prefix for OHLCV Data Storage
OHLCV_S3_PREFIX = "ohlcv"
NEWS_S3_PREFIX = "news"
NEWS_CLEAN_S3_PREFIX = "news_clean"
NEWS_REL_S3_PREFIX = "news_clean_relevance"
RAW_S3_PREFIX = "raw"
PROCESSED_S3_PREFIX = "processed"

#  Raw-zone Desiquant data prefixes
RAW_DESIQUANT_PREFIX = f"{RAW_S3_PREFIX}/desiquant"
RAW_DESIQUANT_OHLCV_PREFIX = f"{RAW_DESIQUANT_PREFIX}/ohlcv"   # intraday parquet per symbol
RAW_DESIQUANT_NEWS_PREFIX = f"{RAW_DESIQUANT_PREFIX}/news"      # news parquet per symbol
RAW_DESIQUANT_FIN_RESULTS_PREFIX = f"{RAW_DESIQUANT_PREFIX}/financial_results"  # financial results per symbol
RAW_DESIQUANT_CORP_ANN_PREFIX = f"{RAW_DESIQUANT_PREFIX}/corporate_announcements"  # corp announcements per symbol

# Processed-zone data prefixes
PROCESSED_OHLCV_PREFIX = f"{PROCESSED_S3_PREFIX}/ohlcv"
PROCESSED_NEWS_PREFIX = f"{PROCESSED_S3_PREFIX}/news"
PROCESSED_FIN_RESULTS_PREFIX = f"{PROCESSED_S3_PREFIX}/financial_results"
PROCESSED_CORP_ANN_PREFIX = f"{PROCESSED_S3_PREFIX}/corporate_announcements"

# Filenames (keep consistent across all assets)
RAW_OHLCV_FILENAME = "ohlcv_raw.parquet"
RAW_NEWS_FILENAME = "news_raw.parquet"
RAW_CORP_ANN_FILENAME = "corporate_announcements_raw.parquet"
RAW_FIN_RESULTS_FILENAME = "financial_results_raw.parquet"

PROCESSED_OHLCV_FILENAME = "ohlcv_processed.parquet"
PROCESSED_NEWS_FILENAME = "news_processed.parquet"
PROCESSED_CORP_ANN_FILENAME = "corporate_announcements_processed.parquet"
PROCESSED_FIN_RESULTS_FILENAME = "financial_results_processed.parquet"

# Feature-store-zone data prefixes
FEATURE_STORE_OHLCV_PREFIX = f"{PROCESSED_S3_PREFIX}/features"
FEATURE_STORE_OHLCV_FILENAME = "ohlcv_features.parquet"
FEATURE_STORE_OHLCV_METADATA_FILENAME = "ohlcv_feature_metadata.json"