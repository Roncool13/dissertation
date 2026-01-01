# src/feature_store/feature_build_ingestor.py
"""
Build engineered OHLCV features (technical indicators) from processed OHLCV parquet
and write a single consolidated parquet to S3 for Feast offline consumption.

Assumptions / conventions (aligned to your current layout):
- Input (processed OHLCV):  s3://{bucket}/processed/ohlcv/{SYMBOL}/{YEAR}/ohlcv_processed.parquet
  columns: ["symbol","date","open","high","low","close","volume"]
- Output (engineered features): s3://{bucket}/processed/features/ohlcv_features.parquet
  columns include base OHLCV + indicators, with:
    - entity key: symbol
    - event timestamp: date (timestamp)
"""

from __future__ import annotations

# Standard library imports
import argparse
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from src.config import setup_logging
from src.io.connections import S3Connection
from src.constants import storage as storage_constants

logger = logging.getLogger(__name__)


# -----------------------------
# Indicator utilities (pandas)
# -----------------------------
def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    # Wilder's smoothing via EWM(alpha=1/window)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a compact set of indicators commonly used for next-day direction models.
    Works per-symbol.
    """
    g = df.copy()
    g.sort_values(["symbol", "date"], inplace=True)

    # Returns
    g["ret_1d"] = g.groupby("symbol")["close"].pct_change(1)
    g["log_ret_1d"] = np.log(g.groupby("symbol")["close"].shift(0) / g.groupby("symbol")["close"].shift(1))

    # Moving averages
    for w in (5, 10, 20):
        g[f"sma_{w}"] = g.groupby("symbol")["close"].transform(lambda s: s.rolling(w, min_periods=w).mean())
    for span in (12, 26):
        g[f"ema_{span}"] = g.groupby("symbol")["close"].transform(lambda s: _ema(s, span))

    # MACD
    g["macd"] = g["ema_12"] - g["ema_26"]
    g["macd_signal"] = g.groupby("symbol")["macd"].transform(lambda s: _ema(s, 9))
    g["macd_hist"] = g["macd"] - g["macd_signal"]

    # RSI / ATR / Volatility
    g["rsi_14"] = g.groupby("symbol")["close"].transform(lambda s: _rsi(s, 14))
    g["atr_14"] = g.groupby("symbol").apply(lambda x: _atr(x["high"], x["low"], x["close"], 14)).reset_index(level=0, drop=True)
    g["volatility_20"] = g.groupby("symbol")["log_ret_1d"].transform(lambda s: s.rolling(20, min_periods=20).std())

    # Simple range / candle body features (often useful)
    g["hl_range"] = g["high"] - g["low"]
    g["oc_change"] = g["close"] - g["open"]
    g["oc_change_pct"] = (g["close"] - g["open"]) / g["open"].replace(0, np.nan)

    return g


# -----------------------------
# Ingestor
# -----------------------------
@dataclass(frozen=True)
class FeatureBuildConfig:
    bucket: str
    symbols: List[str]
    start_year: int
    end_year: int
    processed_prefix: str = "processed"
    ohlcv_relpath: str = "ohlcv"
    output_key: str = "processed/features/ohlcv_features.parquet"
    output_key_prefix: str = storage_constants.FEATURE_STORE_OHLCV_PREFIX


class OhlcvFeatureBuildIngestor:
    """
    Reads processed OHLCV from S3, builds indicators, and writes consolidated features parquet to S3.
    """

    def __init__(self, cfg: FeatureBuildConfig):
        self.cfg = cfg
        self.s3 = S3Connection(bucket=cfg.bucket)

    def _input_key(self, symbol: str, year: int) -> str:
        return f"{self.cfg.processed_prefix}/{self.cfg.ohlcv_relpath}/{symbol}/{year}/ohlcv_processed.parquet"
    
    def _output_key(self, symbol: str) -> str:
        return f"{self.cfg.output_key_prefix}/{symbol}/ohlcv_features.parquet"

    def _download_parquet(self, key: str, dst: Path) -> None:
        # Minimal download helper (S3Connection currently only uploads; use boto3 client directly)
        self.s3.s3_client.download_file(self.cfg.bucket, key, str(dst))

    def _load_year(self, symbol: str, year: int) -> pd.DataFrame:
        key = self._input_key(symbol, year)
        if not self.s3.object_exists(key):
            logger.warning("Missing processed OHLCV: s3://%s/%s (skipping)", self.cfg.bucket, key)
            return pd.DataFrame()

        with tempfile.TemporaryDirectory() as td:
            local = Path(td) / f"{symbol}_{year}.parquet"
            logger.info("Downloading %s -> %s", key, local)
            self._download_parquet(key, local)
            df = pd.read_parquet(local)

        # Defensive schema normalization
        expected = {"symbol", "date", "open", "high", "low", "close", "volume"}
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"Processed OHLCV missing columns {sorted(missing)} for {symbol} {year}: {list(df.columns)}")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = df["symbol"].astype(str)

        # Keep only requested year (in case file has extra)
        df = df[(df["date"].dt.year == year)]
        return df

    def build(self, symbol: str) -> pd.DataFrame:
        # for symbol in self.cfg.symbols:
        frames = []
        for year in range(self.cfg.start_year, self.cfg.end_year + 1):
            df_y = self._load_year(symbol, year)
            if not df_y.empty:
                frames.append(df_y)

        if not frames:
            raise RuntimeError("No processed OHLCV found for the requested symbols/years.")

        ohlcv = pd.concat(frames, ignore_index=True)
        # Drop duplicates on (symbol,date) just in case
        ohlcv.sort_values(["symbol", "date"], inplace=True)
        ohlcv.drop_duplicates(subset=["symbol", "date"], keep="last", inplace=True)

        feats = add_technical_indicators(ohlcv)

        # NOTE: Feast expects event_timestamp column present and timezone-naive timestamps are OK.
        # We keep "date" as timestamp (not date-only) to match typical Feast usage.
        feats["date"] = pd.to_datetime(feats["date"])

        return feats

    def write_to_s3(self, feats: pd.DataFrame, output_path: str) -> None:
        logger.info("Writing %s rows of features to s3://%s/%s", len(feats), self.cfg.bucket, output_path)
        with tempfile.TemporaryDirectory() as td:
            local = Path(td) / "ohlcv_features.parquet"
            feats.to_parquet(local, index=False)
            logger.info("Uploading features to s3://%s/%s", self.cfg.bucket, output_path)
            self.s3.upload_file(local, output_path)
        logger.info("Done. Wrote %s rows of features to s3://%s/%s", len(feats), self.cfg.bucket, output_path)

    def run(self) -> None:
        logger.info("Starting OHLCV feature build ingestion...")
        for symbol in self.cfg.symbols:
            logger.info("Building features for symbol: %s", symbol)
            feats = self.build(symbol)
            output_path = self._output_key(symbol)
            self.write_to_s3(feats, output_path)
            logger.info("Done building features for symbol: %s", symbol)
        logger.info("OHLCV feature build ingestion complete.")
            


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build OHLCV technical-indicator features for Feast.")
    p.add_argument("--symbols", required=True, help="Comma-separated list, e.g., TCS,INFY,RELIANCE")
    p.add_argument("--start-year", type=int, required=True)
    p.add_argument("--end-year", type=int, required=True)
    p.add_argument("--s3-bucket", required=True, default=storage_constants.S3_BUCKET,
                help="Your S3 bucket (e.g., dissertation-databucket)")
    p.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Logging level (e.g. DEBUG, INFO). Defaults to LOG_LEVEL env or INFO.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging(args.log_level)
    cfg = FeatureBuildConfig(
        bucket=args.s3_bucket,
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        start_year=args.start_year,
        end_year=args.end_year,
    )
    OhlcvFeatureBuildIngestor(cfg).run()


if __name__ == "__main__":
    import os  # local import to keep module clean
    main()
