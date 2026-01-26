# Symbols Listed in NSE (National Stock Exchange of India)
NSE_SYMBOLS = [
    "ADANIPORTS",
    "AXISBANK",
    "BPCL",
    "BRITANNIA",
    "HCLTECH",
    "HDFCBANK",
    "HINDUNILVR",
    "ICICIBANK",
    "INFY",
    "IOC",
    "ITC",
    "KOTAKBANK",
    "LT",
    "NESTLEIND",
    "ONGC",
    "RELIANCE",
    "SBIN",
    "SUNPHARMA",
    "TATASTEEL",
    "TCS",
    "WIPRO",
    "INDUSINDBK",
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
