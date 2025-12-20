# Symbols Listed in NSE (National Stock Exchange of India)
NSE_SYMBOLS = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK",
]


# Mapping from NSE Symbols to Company Names
SYMBOL_TO_COMPANY = {
    "TCS": "Tata Consultancy Services",
    "RELIANCE": "Reliance Industries",
    "INFY": "Infosys",
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
    # add more as needed
}

# For relevance semantic anchor (required)
SYMBOL_TO_DESCRIPTION = {
    "TCS": (
        "Tata Consultancy Services is an Indian multinational "
        "information technology services and consulting company listed on NSE."
    ),
    "RELIANCE": (
        "Reliance Industries is an Indian conglomerate with businesses in "
        "energy, petrochemicals, retail, telecommunications and media."
    ),
    "INFY": (
        "Infosys is an Indian multinational information technology company "
        "providing consulting and digital services."
    ),
    "HDFCBANK": (
        "HDFC Bank is a major Indian private sector bank listed on NSE.",
    ),
    "ICICIBANK": (
        "ICICI Bank is a major Indian private sector bank listed on NSE.",
    ),
    # extend as needed
}

# Raw-zone bhavcopy (daily EOD dump for all symbols)
BHAVCOPY_RAW_S3_PREFIX = "bhavcopy/raw"
BHAVCOPY_LOCAL_DIR = "data/raw/bhavcopy"


# Default S3 Bucket and Prefix for OHLCV Data Storage
S3_BUCKET = "dissertation-databucket"
OHLCV_S3_PREFIX = "ohlcv"
NEWS_S3_PREFIX = "news"
NEWS_CLEAN_S3_PREFIX = "news_clean"
NEWS_REL_S3_PREFIX = "news_clean_relevance"
RAW_S3_PREFIX = "raw"
PROCESSED_OHLCV_PREFIX = "ohlcv"  # you already effectively use this pattern

RAW_DESIQUANT_PREFIX = f"{RAW_S3_PREFIX}/desiquant"
RAW_DESIQUANT_OHLCV_PREFIX = f"{RAW_DESIQUANT_PREFIX}/candles"   # intraday parquet per symbol
RAW_DESIQUANT_NEWS_PREFIX = f"{RAW_DESIQUANT_PREFIX}/news"      # news parquet per symbol
RAW_DESIQUANT_ANNOUNCEMENTS_PREFIX = f"{RAW_DESIQUANT_PREFIX}/announcements"  # corp announcements per symbol
RAW_DESIQUANT_FINANCIALS_PREFIX = f"{RAW_DESIQUANT_PREFIX}/results"  # financial results per symbol

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

# Financial Results location pattern (symbol)
# Example seen: s3://desiquant/data/results/nse/TCS.parquet.gz
DESIQUANT_FINANCIAL_RESULTS_BUCKET = "desiquant"
DESIQUANT_FINANCIAL_RESULTS_KEY_TEMPLATE = "data/results/nse/{symbol}.parquet.gz"