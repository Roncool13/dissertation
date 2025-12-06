# src/data/ohlcv_clean.py

# Third-party imports
import pandas as pd


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
      - Ensure correct dtypes
      - Drop duplicates
      - Remove non-sensical rows
    """
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df.drop_duplicates(subset=["symbol", "date"], inplace=True)

    # filter out obviously bad rows
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    df = df[df["high"] >= df["low"]]

    df.sort_values(["symbol", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df