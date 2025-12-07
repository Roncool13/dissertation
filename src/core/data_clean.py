# src/data/ohlcv_clean.py

# Standard library imports
import datetime as dt

# Third-party imports
import pandas as pd

COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]

def clean_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean OHLCV data and return canonical minimal schema:
    date, symbol, open, high, low, close, volume
    """

    df = df.copy()

    # Normalize volume naming if needed
    if "VOLUME" in df.columns:
        df.rename(columns={"VOLUME": "volume"}, inplace=True)

    # Keep strict canonical ordering
    df = df[COLUMNS]

    # Correct dtypes
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove bad rows
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df.drop_duplicates(subset=["symbol", "date"], inplace=True)

    # filter out obviously bad rows
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    df = df[df["high"] >= df["low"]]

    df.sort_values(["symbol", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


def clean_news_data(
    df: pd.DataFrame,
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    """
    Clean raw news DataFrame and transform into canonical schema:
      date, symbol, headline, source, published_at, sentiment_score, sentiment_label
    """
    if df.empty:
        return df

    df = df.copy()
    # Standardize column names depending on provider
    rename_map = {}
    if "title" in df.columns and "headline" not in df.columns:
        rename_map["title"] = "headline"
    if "time_published" in df.columns and "published_at" not in df.columns:
        rename_map["time_published"] = "published_at"
    df.rename(columns=rename_map, inplace=True)

    # Ensure necessary columns exist
    required_cols = ["headline", "source", "published_at"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"News DataFrame missing columns: {missing}")

    # Parse timestamp
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")

    # Derive trading date (you can add custom logic for weekend/holiday mapping later)
    df["date"] = df["published_at"].dt.date

    # Filter by date range (defensive)
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    # Add symbol
    df["symbol"] = symbol

    # TODO: sentiment scoring
    # For now, set neutral sentiment; plug in VADER/TextBlob/HF model later.
    df["sentiment_score"] = 0.0
    df["sentiment_label"] = "neutral"

    # Canonical column order
    df = df[
        [
            "date",
            "symbol",
            "headline",
            "source",
            "published_at",
            "sentiment_score",
            "sentiment_label",
        ]
    ]

    df.dropna(subset=["headline", "published_at"], inplace=True)
    df.sort_values(["symbol", "date", "published_at"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df