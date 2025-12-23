# Desiquant dataset access (R2 endpoint via S3-compatible API)
DESIQUANT_ENDPOINT_URL = "https://cbabd13f6c54798a9ec05df5b8070a6e.r2.cloudflarestorage.com"
DESIQUANT_ACCESS_KEY = "5c8ea9c516abfc78987bc98c70d2868a"
DESIQUANT_SECRET_KEY = "0cf64f9f0b64f6008cf5efe1529c6772daa7d7d0822f5db42a7c6a1e41b3cadf"

# Candles location pattern (symbol + segment)
# Example seen: s3://desiquant/data/candles/SBIN/EQ.parquet.gz  
DESIQUANT_CANDLES_BUCKET = "desiquant"
DESIQUANT_CANDLES_KEY_TEMPLATE = "data/candles/{symbol}/{segment}.parquet.gz"
DEFAULT_CANDLES_SEGMENT = "EQ"

# News location pattern (symbol)
# Example seen: s3://desiquant/data/news/TCS.parquet.gz
DESIQUANT_NEWS_BUCKET = "desiquant"
DESIQUANT_NEWS_KEY_TEMPLATE = "data/news/{symbol}.parquet.gz"

# Corporate Announcements location pattern (symbol)
# Example seen: s3://desiquant/data/announcements/bse/TCS.parquet.gz
DESIQUANT_CORP_ANNOUNCEMENTS_BUCKET = "desiquant"
DESIQUANT_CORP_ANNOUNCEMENTS_KEY_TEMPLATE = "data/announcements/{source}/{symbol}.parquet.gz"
DEFAULT_ANNOUNCEMENTS_SOURCE = "bse"
SUPPORTED_ANNOUNCEMENTS_SOURCES = ["bse", "nse"]

# Financial Results location pattern (symbol)
# Example seen: s3://desiquant/data/results/nse/TCS.parquet.gz
DESIQUANT_FINANCIAL_RESULTS_BUCKET = "desiquant"
DESIQUANT_FINANCIAL_RESULTS_KEY_TEMPLATE = "data/results/nse/{symbol}.parquet"