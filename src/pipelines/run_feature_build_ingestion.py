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
import sys
import argparse
import logging
import tempfile
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

# Third-party imports
import numpy as np
import pandas as pd

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
    logger.debug("Adding technical indicators to %d rows", len(df))
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

    # --- ATR without groupby.apply ---
    g["prev_close"] = g.groupby("symbol")["close"].shift(1)

    tr1 = (g["high"] - g["low"]).abs()
    tr2 = (g["high"] - g["prev_close"]).abs()
    tr3 = (g["low"] - g["prev_close"]).abs()

    g["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    g["atr_14"] = g.groupby("symbol")["tr"].transform(
        lambda s: s.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    )

    g.drop(columns=["prev_close", "tr"], inplace=True)
    # -------------------------------
    
    g["volatility_20"] = g.groupby("symbol")["log_ret_1d"].transform(lambda s: s.rolling(20, min_periods=20).std())

    # Simple range / candle body features (often useful)
    g["hl_range"] = g["high"] - g["low"]
    g["oc_change"] = g["close"] - g["open"]
    g["oc_change_pct"] = (g["close"] - g["open"]) / g["open"].replace(0, np.nan)

    return g


def add_lagged_features(
    df: pd.DataFrame,
    base_cols: List[str],
    lags: List[int],
) -> pd.DataFrame:
    logger.debug("Adding lagged features for %d columns with lags %s", len(base_cols), lags)
    g = df.copy()
    g.sort_values(["symbol", "date"], inplace=True)

    for col in base_cols:
        if col not in g.columns:
            continue
        for lag in lags:
            g[f"{col}_lag{lag}"] = g.groupby("symbol")[col].shift(lag)

    return g


def add_forward_labels(
    df: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    logger.debug("Adding forward labels with horizon %d days", horizon_days)
    g = df.copy()
    g.sort_values(["symbol", "date"], inplace=True)

    fwd_col = f"close_fwd_{horizon_days}d"
    ret_col = f"ret_fwd_{horizon_days}d"
    y_col = f"y_up_{horizon_days}d"

    g[fwd_col] = g.groupby("symbol")["close"].shift(-horizon_days)
    g[ret_col] = (g[fwd_col] / g["close"]) - 1.0
    g[y_col] = (g[fwd_col] > g["close"]).astype(int)

    return g

# -----------------------------
# Ingestor
# -----------------------------
@dataclass(frozen=True)
class FeatureBuildConfig:
    processed_bucket: str
    features_bucket: str
    symbols: List[str]
    start_year: int
    end_year: int

    # NEW
    horizon_days: int = 5
    lags: List[int] = field(default_factory=lambda: [1, 2, 3, 5])
    include_labels: bool = True
    write_metadata: bool = True

    processed_ohlcv_prefix: str = storage_constants.PROCESSED_OHLCV_PREFIX
    processed_ohlcv_filename: str = storage_constants.PROCESSED_OHLCV_FILENAME
    output_key_prefix: str = storage_constants.FEATURE_STORE_OHLCV_PREFIX
    output_filename: str = storage_constants.FEATURE_STORE_OHLCV_FILENAME
    # NEW
    metadata_filename: str = getattr(
        storage_constants,
        "FEATURE_STORE_OHLCV_METADATA_FILENAME",
        "ohlcv_feature_metadata.json",
    )

class OhlcvFeatureBuildIngestor:
    """
    Reads processed OHLCV from S3, builds indicators, and writes consolidated features parquet to S3.
    """

    def __init__(self, cfg: FeatureBuildConfig):
        self.cfg = cfg
        self.s3_processed = S3Connection(bucket=cfg.processed_bucket)
        self.s3_features = S3Connection(bucket=cfg.features_bucket)

    def _input_key(self, symbol: str, year: int) -> str:
        return f"{self.cfg.processed_ohlcv_prefix}/{symbol}/{year}/{self.cfg.processed_ohlcv_filename}"
    
    def _output_key(self) -> str:
        return f"{self.cfg.output_key_prefix}/{self.cfg.output_filename}"

    def _write_local_and_s3(self, filename: str, s3_key: str, write_func: Callable[[Path], None]) -> None:
        """Write artifact locally (for DVC) and upload the same temp file to S3."""
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td) / filename
            logger.debug("Writing %s to temp path %s", filename, tmp_path)
            write_func(tmp_path)

            local_dir = Path("data/features")
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / filename
            logger.debug("Writing %s to local DVC path %s", filename, local_path)
            write_func(local_path)

            logger.info("Uploading %s to s3://%s/%s", filename, self.cfg.features_bucket, s3_key)
            self.s3_features.upload_file(tmp_path, s3_key)

    def _generate_metadata(self, feats: pd.DataFrame) -> dict:
        from datetime import datetime, timezone, date

        test_year = self.cfg.end_year
        val_year = test_year - 1

        train_start = date(self.cfg.start_year, 1, 1)
        train_end = date(val_year - 1, 12, 31)

        val_start = date(val_year, 1, 1)
        val_end = date(val_year, 12, 31)

        test_start = date(test_year, 1, 1)
        test_end = date(test_year, 12, 31)

        logger.debug(
            "Computed split windows -> train:%s-%s val:%s-%s test:%s-%s",
            train_start,
            train_end,
            val_start,
            val_end,
            test_start,
            test_end,
        )

        splits = {
            "scheme": "global_time_split_v1",
            "train": {
                "start": train_start.isoformat(),
                "end": train_end.isoformat(),
            },
            "val": {
                "start": val_start.isoformat(),
                "end": val_end.isoformat(),
            },
            "test": {
                "start": test_start.isoformat(),
                "end": test_end.isoformat(),
            },
        }

        meta = {
            "dataset": "ohlcv_features",
            "symbols": self.cfg.symbols,
            "start_year": self.cfg.start_year,
            "end_year": self.cfg.end_year,
            "horizon_days": self.cfg.horizon_days,
            "lags": self.cfg.lags,
            "row_count": int(len(feats)),
            "columns": list(feats.columns),
            "splits": splits,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        logger.debug(
            "Metadata snapshot -> rows:%d columns:%d include_labels:%s",
            meta["row_count"],
            len(meta["columns"]),
            self.cfg.include_labels,
        )
        return meta

    def _download_parquet(self, key: str, dst: Path) -> None:
        # Minimal download helper (S3Connection currently only uploads; use boto3 client directly)
        logger.debug("Downloading s3://%s/%s to %s", self.cfg.processed_bucket, key, dst)
        self.s3_processed.s3_client.download_file(self.cfg.processed_bucket, key, str(dst))
    def _load_year(self, symbol: str, year: int) -> pd.DataFrame:
        logger.debug("Loading processed OHLCV for %s %s", symbol, year)
        key = self._input_key(symbol, year)
        if not self.s3_processed.object_exists(key):
            logger.warning("Missing processed OHLCV: s3://%s/%s (skipping)", self.cfg.processed_bucket, key)
            return pd.DataFrame()

        with tempfile.TemporaryDirectory() as td:
            local = Path(td) / f"{symbol}_{year}.parquet"
            logger.info("Downloading %s -> %s", key, local)
            self._download_parquet(key, local)
            df = pd.read_parquet(local)
            logger.debug("Loaded %d rows for %s %s", len(df), symbol, year)

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
        logger.debug("Filtered to %d rows for %s %s after year trim", len(df), symbol, year)
        return df

    def build(self) -> pd.DataFrame:
        logger.info(
            "Building OHLCV features for symbols=%s across years %s-%s",
            self.cfg.symbols,
            self.cfg.start_year,
            self.cfg.end_year,
        )
        frames = []
        for symbol in self.cfg.symbols:
            for year in range(self.cfg.start_year, self.cfg.end_year + 1):
                df_y = self._load_year(symbol, year)
                if not df_y.empty:
                    logger.debug("Appending %d rows for %s %s", len(df_y), symbol, year)
                    frames.append(df_y)

        if not frames:
            raise RuntimeError("No processed OHLCV found for the requested symbols/years.")

        ohlcv = pd.concat(frames, ignore_index=True)
        logger.info("Concatenated %d total OHLCV rows before indicator build", len(ohlcv))
        # Drop duplicates on (symbol,date) just in case
        ohlcv.sort_values(["symbol", "date"], inplace=True)
        ohlcv.drop_duplicates(subset=["symbol", "date"], keep="last", inplace=True)
        logger.debug("Post-dedup OHLCV rows: %d", len(ohlcv))

        feats = add_technical_indicators(ohlcv)
        logger.debug("Feature frame after indicators: %s", feats.shape)

        # NOTE: Feast expects event_timestamp column present and timezone-naive timestamps are OK.
        # We keep "date" as timestamp (not date-only) to match typical Feast usage.
        feats["date"] = pd.to_datetime(feats["date"])

        # Lag base columns (only those present will be lagged)
        lag_base_cols = [
            "ret_1d", "log_ret_1d",
            "rsi_14",
            "macd", "macd_signal", "macd_hist",
            "atr_14",
            "volatility_20",
            "sma_5", "sma_10", "sma_20",
            "ema_12", "ema_26",
            "hl_range", "oc_change", "oc_change_pct",
        ]

        feats = add_lagged_features(feats, base_cols=lag_base_cols, lags=self.cfg.lags)
        logger.debug("Feature frame after lags: %s", feats.shape)

        if self.cfg.include_labels:
            feats = add_forward_labels(feats, horizon_days=self.cfg.horizon_days)
            logger.debug("Feature frame after labels: %s", feats.shape)

        # For model training, it's best to drop rows where rolling/lag/label created NaNs
        # (Feast offline can store NaNs too, but training will drop them anyway.)
        required = ["symbol", "date", "open", "high", "low", "close", "volume"]
        if self.cfg.include_labels:
            required += [f"y_up_{self.cfg.horizon_days}d", f"ret_fwd_{self.cfg.horizon_days}d"]
        feats = feats.dropna(subset=[c for c in required if c in feats.columns])
        logger.info("Final feature frame has %d rows and %d columns", *feats.shape)

        return feats

    def write_to_s3(self, feats: pd.DataFrame, output_path: str) -> None:
        logger.info("Writing %s rows of features to s3://%s/%s", len(feats), self.cfg.features_bucket, output_path)
        self._write_local_and_s3(
            filename=self.cfg.output_filename,
            s3_key=output_path,
            write_func=lambda path: feats.to_parquet(path, index=False),
        )
        logger.info("Done. Wrote %s rows of features to s3://%s/%s", len(feats), self.cfg.features_bucket, output_path)

    def write_metadata_to_s3(self, feats: pd.DataFrame) -> None:
        if not self.cfg.write_metadata:
            logger.info("Skipping metadata write (write_metadata disabled)")
            return

        meta = self._generate_metadata(feats)
        key = f"{self.cfg.output_key_prefix}/{self.cfg.metadata_filename}"

        logger.info("Writing metadata JSON to s3://%s/%s", self.cfg.features_bucket, key)
        payload = json.dumps(meta, indent=2)

        def _write_json(path: Path) -> None:
            path.write_text(payload, encoding="utf-8")

        self._write_local_and_s3(
            filename=self.cfg.metadata_filename,
            s3_key=key,
            write_func=_write_json,
        )
        logger.info("Done. Wrote metadata to s3://%s/%s", self.cfg.features_bucket, key)

    def run(self) -> None:
        logger.info("Starting OHLCV feature build ingestion...")
        feats = self.build()
        output_path = self._output_key()
        self.write_to_s3(feats, output_path)
        self.write_metadata_to_s3(feats)
        logger.info("OHLCV feature build ingestion complete.")
            


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build OHLCV technical-indicator features for Feast.")
    p.add_argument("--symbols", required=True, help="Comma-separated list, e.g., TCS,INFY,RELIANCE")
    p.add_argument("--start-year", type=int, required=True)
    p.add_argument("--end-year", type=int, required=True)
    p.add_argument("--s3-processed-bucket", required=True,
                help="Your S3 processed bucket (e.g., dissertation-databucket-processed)")
    p.add_argument("--s3-features-bucket", required=True,
                help="Your S3 features bucket (e.g., dissertation-databucket-dvcstore)")
    p.add_argument("--horizon-days", type=int, default=5)
    p.add_argument("--lags", type=str, default="1,2,3,5", help="Comma-separated lags")
    p.add_argument("--include-labels", action="store_true", default=True)
    p.add_argument("--no-labels", action="store_false", dest="include_labels")
    p.add_argument("--write-metadata", action="store_true", default=True)
    p.add_argument("--no-metadata", action="store_false", dest="write_metadata")
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
        processed_bucket=args.s3_processed_bucket,
        features_bucket=args.s3_features_bucket,
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        start_year=args.start_year,
        end_year=args.end_year,
        horizon_days=args.horizon_days,
        lags=[int(x) for x in args.lags.split(",") if x.strip()],
        include_labels=args.include_labels,
        write_metadata=args.write_metadata,
    )
    OhlcvFeatureBuildIngestor(cfg).run()


if __name__ == "__main__":
    import os  # local import to keep module clean
    main()
