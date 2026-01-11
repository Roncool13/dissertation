# src/core/data_transformations.py

# Standard library imports
import datetime as dt
import logging

# Third-party imports
import pandas as pd

logger = logging.getLogger(__name__)


def aggregate_intraday_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: datetime (or date+time), open/high/low/close, volume.
    Produces daily OHLCV.
    """
    logger.info("Aggregating intraday candles to daily OHLCV; input rows=%s, cols=%s", len(df), list(df.columns))

    # Normalize datetime column
    if "datetime" in df.columns:
        logger.debug("Using 'datetime' column as timestamp")
        ts = pd.to_datetime(df["datetime"])
    elif "date" in df.columns and "time" in df.columns:
        logger.debug("Using 'date' and 'time' columns as timestamp")
        ts = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    elif "timestamp" in df.columns:
        logger.debug("Using 'timestamp' column as timestamp")
        ts = pd.to_datetime(df["timestamp"])
    elif "date" in df.columns:
        logger.debug("Using 'date' column as timestamp")
        ts = pd.to_datetime(df["date"])
    else:
        raise ValueError("No recognizable timestamp column in raw candles data")

    ts = pd.to_datetime(ts, errors="coerce", utc=True).dt.tz_convert(None)  # parse robustly, drop timezone
    df = df.copy()
    df["ts"] = ts
    df["date"] = df["ts"].dt.date
    logger.debug("Parsed timestamps; example date=%s", df["date"].iloc[0] if not df.empty else None)

    # Column normalization (common candle column names)
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("o", "open"): colmap[c] = "open"
        if lc in ("h", "high"): colmap[c] = "high"
        if lc in ("l", "low"): colmap[c] = "low"
        if lc in ("c", "close"): colmap[c] = "close"
        if lc in ("v", "volume", "vol"): colmap[c] = "volume"
    df = df.rename(columns=colmap)
    logger.debug("After renaming, columns=%s", list(df.columns))

    required = ["open", "high", "low", "close"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing required column: {r}")

    if "volume" not in df.columns:
        df["volume"] = 0

    df = df.sort_values(["date", "ts"])
    out = (
        df.groupby("date", as_index=False)
          .agg(
              open=("open", "first"),
              high=("high", "max"),
              low=("low", "min"),
              close=("close", "last"),
              volume=("volume", "sum"),
          )
    )

    out["date"] = pd.to_datetime(out["date"])
    logger.info("Completed aggregation to daily; output rows=%s", len(out))
    return out
