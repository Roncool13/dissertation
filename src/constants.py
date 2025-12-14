# Symbols Listed in NSE (National Stock Exchange of India)
NSE_SYMBOLS = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK",
]


# Default S3 Bucket and Prefix for OHLCV Data Storage
S3_BUCKET = "dissertation-databucket"
OHLCV_S3_PREFIX = "ohlcv"
NEWS_S3_PREFIX = "news"
NEWS_CLEAN_S3_PREFIX = "news_clean"
NEWS_REL_S3_PREFIX = "news_clean_relevance"


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