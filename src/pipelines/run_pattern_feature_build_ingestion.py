"""Build pattern-based OHLCV features from processed OHLCV parquet and upload to S3."""

from __future__ import annotations

# Standard library imports
import argparse
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List

# Third-party imports
import pandas as pd

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Local imports
from src.config import setup_logging
from src.core.patterns import build_pattern_features
from src.io.connections import S3Connection
from src.constants import storage as storage_constants


logger = logging.getLogger(__name__)


def compute_year_splits(start_year: int, end_year: int) -> Dict[str, List[int]]:
    years = list(range(start_year, end_year + 1))
    if len(years) >= 3:
        return {"train_years": [years[0]], "val_years": [years[1]], "test_years": years[2:]}
    if len(years) == 2:
        return {"train_years": [years[0]], "val_years": [], "test_years": [years[1]]}
    return {"train_years": years, "val_years": [], "test_years": []}


class S3Client:
    """Lightweight wrapper to allow dependency injection and easier testing."""

    def __init__(self, bucket: str, conn_factory: Callable[[str], object] = S3Connection):
        self.bucket = bucket
        self._conn = conn_factory(bucket)

    def exists(self, key: str) -> bool:
        return self._conn.object_exists(key)

    def upload(self, local_path: Path, key: str) -> None:
        self._conn.upload_file(local_path, key)

    def download(self, key: str, local_path: Path) -> None:
        self._conn.download_file(key, local_path)


@dataclass(frozen=True)
class PatternFeatureBuildConfig:
    processed_bucket: str
    features_bucket: str
    symbols: List[str]
    start_year: int
    end_year: int
    lookback: int = 10
    local_output_dir: Path = Path("data/features")

    processed_ohlcv_prefix: str = storage_constants.PROCESSED_OHLCV_PREFIX
    processed_ohlcv_filename: str = storage_constants.PROCESSED_OHLCV_FILENAME

    output_key_prefix: str = storage_constants.FEATURE_STORE_PATTERN_PREFIX
    output_features_filename: str = storage_constants.FEATURE_STORE_PATTERN_FILENAME
    output_metadata_filename: str = storage_constants.FEATURE_STORE_PATTERN_METADATA_FILENAME


class PatternFeatureBuildIngestor:
    def __init__(
        self,
        cfg: PatternFeatureBuildConfig,
        s3_client_factory: Callable[[str], S3Client] = S3Client,
    ) -> None:
        self.cfg = cfg
        self.s3_processed = s3_client_factory(cfg.processed_bucket)
        self.s3_features = s3_client_factory(cfg.features_bucket)

    def _input_key(self, symbol: str, year: int) -> str:
        return f"{self.cfg.processed_ohlcv_prefix}/{symbol}/{year}/{self.cfg.processed_ohlcv_filename}"

    def _output_key(self, filename: str) -> str:
        return f"{self.cfg.output_key_prefix}/{filename}"

    def _load_ohlcv_year(self, symbol: str, year: int) -> pd.DataFrame:
        key = self._input_key(symbol, year)
        if not self.s3_processed.exists(key):
            logger.info("Processed OHLCV missing; skipping s3://%s/%s", self.cfg.processed_bucket, key)
            return pd.DataFrame()

        with tempfile.TemporaryDirectory() as td:
            local = Path(td) / f"{symbol}_{year}_ohlcv.parquet"
            logger.info("Downloading %s -> %s", key, local)
            self.s3_processed.download(key, local)
            df = pd.read_parquet(local)

        if df.empty:
            return df

        df = df.copy()
        if "symbol" not in df.columns:
            df.insert(0, "symbol", symbol)
        df["symbol"] = df["symbol"].astype(str)

        if "date" not in df.columns:
            raise ValueError(f"Processed OHLCV missing 'date' for {symbol} {year}. Columns={list(df.columns)}")
        df["date"] = pd.to_datetime(df["date"])

        return df

    def _load_all_ohlcv(self) -> pd.DataFrame:
        years = list(range(self.cfg.start_year, self.cfg.end_year + 1))
        frames: List[pd.DataFrame] = []

        for sym in self.cfg.symbols:
            for year in years:
                dfy = self._load_ohlcv_year(sym, year)
                if dfy.empty:
                    continue
                frames.append(dfy)

        if not frames:
            raise RuntimeError("No processed OHLCV found for given symbols/years.")

        ohlcv = pd.concat(frames, axis=0, ignore_index=True)
        return ohlcv

    def run(self) -> None:
        if self.cfg.start_year > self.cfg.end_year:
            raise ValueError("Start year cannot be after end year.")

        logger.info(
            "Building pattern features for %s (%s-%s)",
            ",".join(self.cfg.symbols),
            self.cfg.start_year,
            self.cfg.end_year,
        )

        ohlcv = self._load_all_ohlcv()
        feats = build_pattern_features(ohlcv, lookback=self.cfg.lookback)

        os.makedirs(self.cfg.local_output_dir, exist_ok=True)
        out_parquet = self.cfg.local_output_dir / self.cfg.output_features_filename
        out_meta = self.cfg.local_output_dir / self.cfg.output_metadata_filename

        feats.to_parquet(out_parquet, index=False)

        meta = {
            "feature_set": "candlestick_patterns_rule_based",
            "symbols": self.cfg.symbols,
            "start_year": self.cfg.start_year,
            "end_year": self.cfg.end_year,
            "lookback": self.cfg.lookback,
            "rows": int(len(feats)),
            "columns": list(feats.columns),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "year_splits_default": compute_year_splits(self.cfg.start_year, self.cfg.end_year),
            "source_processed_prefix": f"{self.cfg.processed_ohlcv_prefix}/",
        }
        with open(out_meta, "w") as f:
            json.dump(meta, f, indent=2)

        features_key = self._output_key(self.cfg.output_features_filename)
        metadata_key = self._output_key(self.cfg.output_metadata_filename)

        self.s3_features.upload(out_parquet, features_key)
        self.s3_features.upload(out_meta, metadata_key)

        logger.info("Wrote %s and %s", out_parquet, out_meta)
        logger.info("Uploaded to s3://%s/%s", self.cfg.features_bucket, self.cfg.output_key_prefix)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. TCS,INFY")
    ap.add_argument("--start-year", required=True, type=int, help="Start year YYYY")
    ap.add_argument("--end-year", required=True, type=int, help="End year YYYY")
    ap.add_argument("--lookback", type=int, default=10, help="Lookback period for rolling features")
    ap.add_argument("--s3-processed-bucket", required=True, help="S3 bucket for processed OHLCV data")
    ap.add_argument("--s3-features-bucket", required=True, help="S3 bucket for feature store data")
    ap.add_argument("--log-level", default="INFO", help="Logging level (e.g. INFO, DEBUG)")
    args = ap.parse_args()

    setup_logging(args.log_level)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    cfg = PatternFeatureBuildConfig(
        processed_bucket=args.s3_processed_bucket,
        features_bucket=args.s3_features_bucket,
        symbols=symbols,
        start_year=args.start_year,
        end_year=args.end_year,
        lookback=args.lookback,
    )

    PatternFeatureBuildIngestor(cfg).run()


if __name__ == "__main__":
    main()
