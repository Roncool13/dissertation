# src/core/data_clean.py

# Standard library imports
import datetime as dt
import logging

# Third-party imports
import pandas as pd

COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]

logger = logging.getLogger(__name__)


def clean_ohlcv_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Clean OHLCV data and return canonical minimal schema:
    date, symbol, open, high, low, close, volume
    """

    logger.debug("Starting OHLCV clean with %s rows and columns: %s", len(df), list(df.columns))
    df = df.copy()

    # Normalize volume naming if needed
    if "VOLUME" in df.columns:
        logger.info("Renaming VOLUME column to volume for OHLCV data")
        df.rename(columns={"VOLUME": "volume"}, inplace=True)

    # Keep strict canonical ordering
    logger.debug("Reordering OHLCV columns to canonical schema: %s", COLUMNS)
    df = df[COLUMNS]

    # Correct dtypes
    logger.debug("Converting OHLCV date column to datetime")
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        logger.debug("Converting column %s to numeric", col)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove bad rows
    before_dropna = len(df)
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    logger.debug("Dropped %s rows with NaNs in price columns", before_dropna - len(df))
    before_dupes = len(df)
    df.drop_duplicates(subset=["date"], inplace=True)
    logger.debug("Removed %s duplicate OHLCV rows", before_dupes - len(df))

    # filter out obviously bad rows
    before_filter = len(df)
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    df = df[df["high"] >= df["low"]]
    logger.debug("Filtered %s rows with invalid price data", before_filter - len(df))

    df.sort_values(["date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Insert symbol column
    df.insert(0, 'symbol', symbol)
    
    logger.info("Completed OHLCV clean with %s rows for %s symbols", len(df), df["symbol"].nunique())

    return df


def clean_news_data(
    df: pd.DataFrame,
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    """
    Clean raw news DataFrame and transform into canonical schema.
    Keeps summary/link/query_used if present (useful for relevance + debugging).
    """
    if df.empty:
        logger.info("Received empty news DataFrame for %s; skipping clean", symbol)
        return df

    logger.debug(
        "Starting news clean for %s with %s rows, columns: %s, date window: %s -> %s",
        symbol,
        len(df),
        list(df.columns),
        start_date,
        end_date,
    )
    df = df.copy()

    # Standardize column names depending on provider
    rename_map = {}
    if "title" in df.columns and "headline" not in df.columns:
        rename_map["title"] = "headline"
    if "time_published" in df.columns and "published_at" not in df.columns:
        rename_map["time_published"] = "published_at"
    df.rename(columns=rename_map, inplace=True)
    if rename_map:
        logger.debug("Applied news column renames: %s", rename_map)

    required_cols = ["headline", "source", "published_at"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error("News DataFrame missing required columns: %s", missing)
        raise ValueError(f"News DataFrame missing columns: {missing}")

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["date"] = df["published_at"].dt.date
    before_date_filter = len(df)
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    logger.debug("Filtered %s news rows outside date window", before_date_filter - len(df))
    df["symbol"] = symbol

    # Keep optional columns if present
    for optional in ["summary", "link", "query_used"]:
        if optional not in df.columns:
            df[optional] = pd.NA

    # Placeholder sentiment (FinBERT pipeline will fill later)
    df["sentiment_score"] = 0.0
    df["sentiment_label"] = "neutral"

    df = df[
        [
            "date",
            "symbol",
            "headline",
            "summary",
            "source",
            "published_at",
            "link",
            "query_used",
            "sentiment_score",
            "sentiment_label",
        ]
    ]

    before_dropna = len(df)
    df.dropna(subset=["headline", "published_at"], inplace=True)
    logger.debug("Dropped %s news rows missing headline/published_at", before_dropna - len(df))
    before_dupes = len(df)
    df.drop_duplicates(subset=["symbol", "headline", "published_at"], inplace=True)
    logger.debug("Removed %s duplicate news rows", before_dupes - len(df))
    df.sort_values(["symbol", "date", "published_at"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(
        "Completed news clean for %s with %s rows between %s and %s",
        symbol,
        len(df),
        df["date"].min(),
        df["date"].max(),
    )
    return df
