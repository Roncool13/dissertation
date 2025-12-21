# src/pipelines/run_ohlcv_ingestion.py

# Standard library imports
import argparse
import datetime as dt
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Callable, List, Optional, Sequence, Tuple

import pandas as pd
from pandas.tseries.offsets import BDay

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Local imports
import src.constants as constants
from src.config import setup_logging
from src.core.data_clean import clean_ohlcv_data, clean_news_data
from src.core.data_transformations import aggregate_intraday_to_daily
from src.io.connections import S3Connection
from src.nlp.relevance_scoring import RelevanceScorer


@dataclass
class IngestConfig:
    symbol: str
    start_year: int
    end_year: int
    local_output: Path
    s3_bucket: str

    # Prefixes
    ohlcv_prefix: str = constants.OHLCV_S3_PREFIX
    ohlcv_raw_prefix: str = constants.RAW_DESIQUANT_OHLCV_PREFIX
    news_raw_prefix: str = constants.RAW_DESIQUANT_NEWS_PREFIX
    news_clean_prefix: str = constants.NEWS_CLEAN_S3_PREFIX
    news_rel_prefix: str = constants.NEWS_REL_S3_PREFIX


def _year_end_date(year: int, latest: dt.date) -> dt.date:
    """
    Use T-2 business days for the current year, otherwise December 31st.
    """
    end_date = dt.date(year, 12, 31)
    if end_date > latest:
        logging.getLogger(__name__).debug(
            "Adjusted end date for year %s from %s to earliest allowed %s", year, end_date, latest
        )
        return latest
    logging.getLogger(__name__).debug("Computed year-end date for %s: %s", year, end_date)
    return end_date


def _year_start_date(year: int, earliest: dt.date) -> dt.date:
    """
    Use January 1st of the year, but not before the earliest allowed date.
    """
    start_date = dt.date(year, 1, 1)
    if start_date < earliest:
        logging.getLogger(__name__).debug(
            "Adjusted start date for year %s from %s to earliest allowed %s", year, start_date, earliest
        )
        return earliest
    logging.getLogger(__name__).debug("Computed year-start date for %s: %s", year, start_date)
    return start_date


class S3Client:
    """
    Lightweight wrapper to allow dependency injection and easier testing.
    Uses your existing S3Connection under the hood.
    """

    def __init__(self, bucket: str, conn_factory: Callable[[str], object] = S3Connection):
        self.bucket = bucket
        self._conn = conn_factory(bucket)

    def exists(self, key: str) -> bool:
        return self._conn.object_exists(key)

    def upload(self, local_path: Path, key: str) -> None:
        self._conn.upload_file(local_path, key)

    def download(self, key: str, local_path: Path) -> None:
        self._conn.download_file(key, local_path)



class OHLCVIngestor:
    """
    Encapsulates the OHLCV ingestion pipeline.
    """

    logger = logging.getLogger(__name__)

    def __init__(self, config: IngestConfig, s3_client_factory: Callable[[str], S3Client] = S3Client):
        self.config = config
        self.s3 = s3_client_factory(config.s3_bucket)

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
        if symbol not in constants.NSE_SYMBOLS:
            raise ValueError(f"Symbol {symbol!r} is not a valid NSE symbol.")

    @staticmethod
    def _s3_object_key(prefix: str, symbol: str, year: int) -> str:
        return f"{prefix}/{symbol}/{year}/ohlcv.parquet"
    
    @staticmethod
    def _raw_s3_object_key(prefix: str, symbol: str) -> str:
        return f"{prefix}/{symbol}/EQ.parquet"

    def prepare_year_prefixes(self) -> Tuple[List[int], List[str]]:
        years = list(range(self.config.start_year, self.config.end_year + 1))
        keys = [self._s3_object_key(self.config.ohlcv_prefix, self.config.symbol, y) for y in years]

        remaining_years: List[int] = []
        remaining_keys: List[str] = []
        for y, key in zip(years, keys):
            if self.s3.exists(key):
                self.logger.info("OHLCV exists; skipping year=%s at s3://%s/%s", y, self.config.s3_bucket, key)
            else:
                remaining_years.append(y)
                remaining_keys.append(key)

        return remaining_years, remaining_keys
    
    def fetch_raw_data(self) -> pd.DataFrame:
        raw_key = self._raw_s3_object_key(self.config.ohlcv_raw_prefix, self.config.symbol)
        self.logger.info("Reading RAW candles from s3://%s/%s", self.config.s3_bucket, raw_key)

        with tempfile.TemporaryDirectory() as td:
            raw_local = os.path.join(td, f"{self.config.symbol}_raw.parquet")
            if not self.s3.exists(raw_key):
                raise FileNotFoundError(f"RAW candles not found at s3://{self.config.s3_bucket}/{raw_key}")

            self.s3.download(raw_key, raw_local)
            raw_df = pd.read_parquet(raw_local)

            self.logger.info("Raw rows: %d", len(raw_df))

            daily_df = aggregate_intraday_to_daily(raw_df)
            self.logger.info("Daily rows: %d", len(daily_df))

            return daily_df

    def ingest_year(self, year: int, key: str, df: pd.DataFrame, daily_start: dt.date, daily_end: dt.date) -> None:
        start = _year_start_date(year, daily_start)
        end = _year_end_date(year, daily_end)

        self.logger.info("Ingesting OHLCV for year %s: %s to %s", year, start, end)

        self.logger.info("Filtering OHLCV for year %s: %s to %s...", year, start, end)
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        if df.empty:
            self.logger.warning("No OHLCV data for %s after filtering %s → %s", self.config.symbol, start, end)
            return

        df = clean_ohlcv_data(df)
        if df.empty:
            raise RuntimeError(f"[VALIDATION] OHLCV empty for {self.config.symbol} year={year} start={start} end={end} after cleaning.")

        self.logger.info("Cleaned OHLCV has %s rows. Saving and uploading to S3...", len(df))
        self.config.local_output.parent.mkdir(parents=True, exist_ok=True)

        local_path = self.config.local_output.with_name(
            f"{self.config.local_output.stem}_{self.config.symbol}_{year}_ohlcv.parquet"
        )
        df.to_parquet(local_path, index=False)
        self.s3.upload(local_path, key)
        self.logger.info("Uploaded OHLCV to s3://%s/%s", self.config.s3_bucket, key)

    def run(self) -> None:
        self.validate_symbol(self.config.symbol)
        if self.config.start_year > self.config.end_year:
            raise ValueError("Start year cannot be after end year.")

        years_to_process, keys = self.prepare_year_prefixes()
        if not keys:
            self.logger.info("All OHLCV data already exists in S3. Nothing to ingest.")
            return

        self.logger.info("Need to ingest OHLCV data for %s year(s).", len(keys))

        daily_df = self.fetch_raw_data()
        earliest = daily_df["date"].min()
        latest = daily_df["date"].max()
        self.logger.info("Available daily OHLCV date range: %s to %s", earliest, latest)

        for year, key in zip(years_to_process, keys):
            self.ingest_year(year, key, daily_df, earliest, latest)

        self.logger.info("OHLCV ingestion complete.")


class NewsIngestor:
    """
    News ingestion pipeline:
      - Fetch Google News RSS
      - Clean
      - Semantic relevance scoring
      - Upload clean + clean+relevance to S3
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        config: IngestConfig,
        s3_client_factory: Callable[[str], S3Client] = S3Client,
        scorer: Optional[RelevanceScorer] = None,
    ):
        self.config = config
        self.s3 = s3_client_factory(config.s3_bucket)
        self.scorer = scorer or RelevanceScorer()

    @staticmethod
    def _s3_clean_key(prefix: str, symbol: str, year: int) -> str:
        return f"{prefix}/{symbol}/{year}/news_clean.parquet"

    @staticmethod
    def _s3_rel_key(prefix: str, symbol: str, year: int) -> str:
        return f"{prefix}/{symbol}/{year}/news_rel.parquet"
    
    @staticmethod
    def _raw_s3_object_key(prefix: str, symbol: str) -> str:
        return f"{prefix}/{symbol}/news.parquet"

    def prepare_year_prefixes(self) -> Tuple[List[int], List[str], List[str]]:
        years = list(range(self.config.start_year, self.config.end_year + 1))
        clean_keys = [self._s3_clean_key(self.config.news_clean_prefix, self.config.symbol, y) for y in years]
        rel_keys = [self._s3_rel_key(self.config.news_rel_prefix, self.config.symbol, y) for y in years]

        remaining_years: List[int] = []
        remaining_clean: List[str] = []
        remaining_rel: List[str] = []

        for y, ck, rk in zip(years, clean_keys, rel_keys):
            # If relevance layer exists, consider year done
            if self.s3.exists(rk):
                self.logger.info("NEWS+REL exists; skipping year=%s at s3://%s/%s", y, self.config.s3_bucket, rk)
            else:
                remaining_years.append(y)
                remaining_clean.append(ck)
                remaining_rel.append(rk)

        return remaining_years, remaining_clean, remaining_rel
    
    def fetch_raw_data(self) -> pd.DataFrame:
        raw_key = self._raw_s3_object_key(self.config.news_raw_prefix, self.config.symbol)
        self.logger.info("Reading RAW news from s3://%s/%s", self.config.s3_bucket, raw_key)

        with tempfile.TemporaryDirectory() as td:
            raw_local = os.path.join(td, f"{self.config.symbol}_raw.parquet")
            if not self.s3.exists(raw_key):
                raise FileNotFoundError(f"RAW news not found at s3://{self.config.s3_bucket}/{raw_key}")

            self.s3.download(raw_key, raw_local)
            raw_df = pd.read_parquet(raw_local)

            self.logger.info("Raw rows: %d", len(raw_df))

            return raw_df

    def ingest_year(self, year: int, clean_key: str, rel_key: str, df: pd.DataFrame, earliest: dt.date, latest: dt.date) -> None:
        start = _year_start_date(year, earliest)
        end = _year_end_date(year, latest)

        self.logger.info("Ingesting NEWS for %s from %s to %s...", self.config.symbol, start, end)
        # raw = fetch_news_from_provider(self.config.symbol, start, end)

        self.logger.info("Filtering NEWS for year %s: %s to %s...", year, start, end)
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        if df.empty:
            self.logger.warning("No NEWS data for %s after filtering %s → %s", self.config.symbol, start, end)
            return

        self.logger.info("Fetched %s raw news rows. Cleaning...", len(df))
        news_clean = clean_news_data(df, self.config.symbol, start, end)
        if news_clean.empty:
            raise RuntimeError(f"[VALIDATION] NEWS empty for {self.config.symbol} year={year}, {self.config.symbol}, start, end")

        self.logger.info("Cleaned NEWS has %s rows. Saving and uploading to S3...", len(news_clean))

        self.config.local_output.parent.mkdir(parents=True, exist_ok=True)

        local_clean = self.config.local_output.with_name(
            f"{self.config.local_output.stem}_{self.config.symbol}_{year}_news_clean.parquet"
        )
        news_clean.to_parquet(local_clean, index=False)
        self.s3.upload(local_clean, clean_key)
        self.logger.info("Uploaded NEWS clean to s3://%s/%s", self.config.s3_bucket, clean_key)

        # relevance scoring
        if news_clean.empty:
            news_rel = news_clean.copy()
            news_rel["relevance_score"] = pd.Series(dtype="float32")
        else:
            scores = self.scorer.score(
                symbol=self.config.symbol,
                headlines=news_clean["headline"].astype(str).tolist(),
                summaries=news_clean["summary"].astype(str).tolist()
                if "summary" in news_clean.columns
                else None,
            )
            news_rel = news_clean.copy()
            news_rel["relevance_score"] = scores.astype("float32")

        local_rel = self.config.local_output.with_name(
            f"{self.config.local_output.stem}_{self.config.symbol}_{year}_news_rel.parquet"
        )
        news_rel.to_parquet(local_rel, index=False)
        self.s3.upload(local_rel, rel_key)
        self.logger.info("Uploaded NEWS+REL to s3://%s/%s", self.config.s3_bucket, rel_key)

    def run(self) -> None:
        if self.config.start_year > self.config.end_year:
            raise ValueError("Start year cannot be after end year.")

        years, clean_keys, rel_keys = self.prepare_year_prefixes()
        if not rel_keys:
            self.logger.info("All NEWS+REL data already exists in S3. Nothing to ingest.")
            return

        self.logger.info("Need to ingest NEWS+REL data for %s year(s).", len(rel_keys))

        raw_df = self.fetch_raw_data()
        earliest = raw_df["date"].min()
        latest = raw_df["date"].max()
        self.logger.info("Available raw NEWS date range: %s to %s", earliest, latest)

        for y, ck, rk in zip(years, clean_keys, rel_keys):
            self.ingest_year(y, ck, rk, raw_df,earliest, latest)

        self.logger.info("News ingestion complete.")


class MultiModalIngestor:
    """
    Orchestrates running OHLCV ingestion and News ingestion sequentially
    with the same CLI inputs (symbol, start, end).
    """

    logger = logging.getLogger(__name__)

    def __init__(self, config: IngestConfig):
        self.config = config

    def run(self, run_ohlcv: bool = True, run_news: bool = True, log_level: Optional[str] = None) -> None:
        setup_logging(log_level)
        self.logger.info(
            "Starting multi-modal ingestion for %s (years %s-%s) with log level %s",
            self.config.symbol,
            self.config.start_year,
            self.config.end_year,
            (log_level or "ENV/DEFAULT").upper(),
        )
        # Validate inputs once here
        OHLCVIngestor.validate_symbol(self.config.symbol)

        if run_ohlcv:
            OHLCVIngestor(self.config).run()

        if run_news:
            NewsIngestor(self.config).run()

        self.logger.info("Multi-modal ingestion finished. (OHLCV=%s, NEWS=%s)", run_ohlcv, run_news)


# ---- CLI ----
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-modal ingestion pipeline (OHLCV + News + Relevance).")
    parser.add_argument("--symbol", type=str, required=True, help="Single NSE symbol to ingest (e.g. TCS).")
    parser.add_argument("--start", type=str, required=True, help="Start year YYYY")
    parser.add_argument("--end", type=str, required=True, help="End year YYYY")
    parser.add_argument("--local-output", type=str, default="/tmp/ingest.parquet", help="Local temp parquet stem")
    parser.add_argument("--s3-bucket", type=str, default=constants.S3_BUCKET, help="S3 bucket name")
    parser.add_argument("--ohlcv-only", action="store_true", help="Only ingest OHLCV.")
    parser.add_argument("--news-only", action="store_true", help="Only ingest news + relevance.")
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Logging level (e.g. DEBUG, INFO). Defaults to LOG_LEVEL env or INFO.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)

    symbol = args.symbol.strip().upper()
    start_year = OHLCVIngestor.validate_year_arg(args.start)
    end_year = OHLCVIngestor.validate_year_arg(args.end)

    run_ohlcv = True
    run_news = True
    if args.ohlcv_only:
        run_news = False
    if args.news_only:
        run_ohlcv = False

    config = IngestConfig(
        symbol=symbol,
        start_year=start_year,
        end_year=end_year,
        local_output=Path(args.local_output),
        s3_bucket=args.s3_bucket,
    )

    MultiModalIngestor(config).run(run_ohlcv=run_ohlcv, run_news=run_news, log_level=args.log_level)


if __name__ == "__main__":
    main()
