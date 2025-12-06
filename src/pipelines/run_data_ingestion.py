# src/pipelines/run_ohlcv_ingestion.py

# Standard library imports
import argparse
import datetime as dt
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Local imports
from src.config import setup_logging
from src.constants import NSE_SYMBOLS, S3_BUCKET, OHLCV_S3_PREFIX
from src.core.data_download import download_ohlcv_nsepy
from src.core.data_clean import clean_ohlcv
from src.io.connections import S3Connection


@dataclass
class IngestConfig:
    symbol: str
    start_year: int
    end_year: int
    local_output: Path
    s3_bucket: str
    s3_prefix: str = OHLCV_S3_PREFIX


class S3Client:
    """
    Lightweight wrapper to allow dependency injection and easier testing.
    Uses your existing S3Connection under the hood.
    """

    def __init__(self, bucket: str, conn_factory: Callable[[str], object] = S3Connection):
        self.bucket = bucket
        # conn_factory(bucket) should implement .object_exists(key) and .upload_file(local, key)
        self._conn = conn_factory(bucket)

    def exists(self, key: str) -> bool:
        return self._conn.object_exists(key)

    def upload(self, local_path: Path, key: str) -> None:
        self._conn.upload_file(local_path, key)


class OHLCVIngestor:
    """
    Encapsulates the OHLCV ingestion pipeline.
    """

    logger = logging.getLogger(__name__)
    YEAR_KEY_PATTERN = re.compile(r"/(?P<symbol>[^/]+)/(?P<year>\d{4})/ohlcv\.parquet$")

    def __init__(self, config: IngestConfig, s3_client_factory: Callable[[str], S3Client] = S3Client):
        self.config = config
        # create a local client bound to the configured bucket
        self.s3 = s3_client_factory(config.s3_bucket)

    # ---- Validation helpers ----
    @staticmethod
    def validate_year_arg(arg: str) -> int:
        if not isinstance(arg, str) or len(arg) != 4 or not arg.isdigit():
            raise ValueError(f"Invalid year format: {arg!r}. Expected 'YYYY'.")
        year = int(arg)
        if not (1 <= year <= 9999):
            raise ValueError(f"Invalid year value: {year}")
        return year

    @staticmethod
    def validate_symbol(symbol: str) -> None:
        if symbol not in NSE_SYMBOLS:
            raise ValueError(f"Symbol {symbol!r} is not a valid NSE symbol.")

    @staticmethod
    def _s3_object_key(prefix: str, symbol: str, year: int) -> str:
        # store per-year under prefix/{symbol}/{year}/ohlcv.parquet
        return f"{prefix}/symbol={symbol}/start={dt.date(year,1,1)}/end={dt.date(year,12,31)}/ohlcv.parquet"

    # ---- S3 checking / listing ----
    def prepare_year_prefixes(self) -> Tuple[List[int], List[str]]:
        """
        Build a list of candidate object keys for each year and return (year_list, prefixes_to_process)
        where year_list contains the years corresponding to prefixes_to_process.
        """
        years = list(range(self.config.start_year, self.config.end_year + 1))
        prefixes = [self._s3_object_key(self.config.s3_prefix, self.config.symbol, y) for y in years]

        remaining_years: List[int] = []
        remaining_prefixes: List[str] = []
        for y, key in zip(years, prefixes):
            if self.s3.exists(key):
                self.logger.info(
                    "S3 object already exists; skipping upload for year=%s at s3://%s/%s",
                    y,
                    self.config.s3_bucket,
                    key,
                )
            else:
                remaining_years.append(y)
                remaining_prefixes.append(key)

        return remaining_years, remaining_prefixes

    # ---- Ingestion ----
    def ingest_year(self, year: int, key: str) -> None:
        start = dt.date(year, 1, 1)
        end = dt.date(year, 12, 31)
        self.logger.info("Downloading OHLCV for %s from %s to %s...", self.config.symbol, start, end)
        df = download_ohlcv_nsepy(self.config.symbol, start, end)
        self.logger.info("Downloaded %s rows. Cleaning data...", len(df))
        df = clean_ohlcv(df)
        self.logger.info("Cleaned data has %s rows. Saving and uploading to S3...", len(df))

        # Ensure local dir exists
        self.config.local_output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.config.local_output, index=False)
        # save_to_parquet(df, self.config.local_output)

        self.s3.upload(self.config.local_output, key)
        self.logger.info("Uploaded to s3://%s/%s", self.config.s3_bucket, key)

    def run(self) -> None:
        # Validate inputs first
        self.validate_symbol(self.config.symbol)
        if self.config.start_year > self.config.end_year:
            raise ValueError("Start year cannot be after end year.")

        years_to_process, prefixes = self.prepare_year_prefixes()
        if not prefixes:
            self.logger.info("All OHLCV data already exists in S3. Nothing to ingest.")
            return

        self.logger.info("Need to ingest OHLCV data for %s year(s).", len(prefixes))
        for year, key in zip(years_to_process, prefixes):
            self.ingest_year(year, key)

        self.logger.info("OHLCV ingestion complete.")


# ---- CLI ----
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OHLCV ingestion pipeline.")
    parser.add_argument("--symbol", type=str, required=True, help="Single NSE symbol to ingest (e.g. TCS).")
    parser.add_argument("--start", type=str, required=True, help="Start year YYYY")
    parser.add_argument("--end", type=str, required=True, help="End year YYYY")
    parser.add_argument("--local-output", type=str, default="/tmp/ohlcv.parquet", help="Local temp parquet file")
    parser.add_argument("--s3-bucket", type=str, default=S3_BUCKET, help="S3 bucket name")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    setup_logging()
    args = parse_args(argv)

    symbol = args.symbol.strip().upper()
    start_year = OHLCVIngestor.validate_year_arg(args.start)
    end_year = OHLCVIngestor.validate_year_arg(args.end)

    config = IngestConfig(
        symbol=symbol,
        start_year=start_year,
        end_year=end_year,
        local_output=Path(args.local_output),
        s3_bucket=args.s3_bucket,
    )

    ingestor = OHLCVIngestor(config)
    ingestor.run()


if __name__ == "__main__":
    main()
