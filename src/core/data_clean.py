# src/data/ohlcv_clean.py

# Third-party imports
import pandas as pd

COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
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