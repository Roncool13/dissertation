# src/pipelines/run_ohlcv_ingestion.py

# Standard library imports
import os
import sys
import argparse
import datetime as dt
import logging
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Callable, List, Optional, Sequence, Tuple

# Third-party imports
import pandas as pd
from pandas.tseries.offsets import BDay

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Local imports
import src.constants.symbols as symbols_constants
import src.constants.storage as storage_constants
import src.constants.desiquant as desiquant_constants
from src.config import setup_logging
from src.core.data_transformations import aggregate_intraday_to_daily
from src.io.connections import S3Connection
# from src.nlp.relevance_scoring import RelevanceScorer
from src.core.data_clean import (
    clean_ohlcv_data, 
    clean_news_data,
    clean_desiq_news_data,
    clean_desiq_corporate_announcements,
    clean_desiq_financial_results,
)


# Logger Config
logger = logging.getLogger(__name__)

@dataclass
class IngestConfig:
    symbol: str
    start_year: int
    end_year: int
    local_output: Path
    s3_bucket: str

    # Raw Prefixes
    ohlcv_raw_prefix: str = storage_constants.RAW_DESIQUANT_OHLCV_PREFIX
    news_raw_prefix: str = storage_constants.RAW_DESIQUANT_NEWS_PREFIX
    results_raw_prefix: str = storage_constants.RAW_DESIQUANT_FIN_RESULTS_PREFIX
    announcements_raw_prefix: str = storage_constants.RAW_DESIQUANT_CORP_ANN_PREFIX

    # Processed Prefixes
    ohlcv_processed_prefix: str = storage_constants.PROCESSED_OHLCV_PREFIX
    news_processed_prefix: str = storage_constants.PROCESSED_NEWS_PREFIX
    results_processed_prefix: str = storage_constants.PROCESSED_FIN_RESULTS_PREFIX
    announcements_processed_prefix: str = storage_constants.PROCESSED_CORP_ANN_PREFIX

    # Raw Filenames
    ohlcv_raw_filename: str = storage_constants.RAW_OHLCV_FILENAME
    news_raw_filename: str = storage_constants.RAW_NEWS_FILENAME
    results_raw_filename: str = storage_constants.RAW_FIN_RESULTS_FILENAME
    announcements_raw_filename: str = storage_constants.RAW_CORP_ANN_FILENAME

    # Processed Filenames
    ohlcv_processed_filename: str = storage_constants.PROCESSED_OHLCV_FILENAME
    news_processed_filename: str = storage_constants.PROCESSED_NEWS_FILENAME
    results_processed_filename: str = storage_constants.PROCESSED_FIN_RESULTS_FILENAME
    announcements_processed_filename: str = storage_constants.PROCESSED_CORP_ANN_FILENAME

def _year_end_date(year: int, latest: dt.date) -> dt.date:
    """
    Use T-2 business days for the current year, otherwise December 31st.
    """
    end_date = dt.date(year, 12, 31)
    if end_date > latest:
        logger.debug(
            "Adjusted end date for year %s from %s to earliest allowed %s", year, end_date, latest
        )
        return latest
    logger.debug("Computed year-end date for %s: %s", year, end_date)
    return end_date


def _year_start_date(year: int, earliest: dt.date) -> dt.date:
    """
    Use January 1st of the year, but not before the earliest allowed date.
    """
    start_date = dt.date(year, 1, 1)
    if start_date < earliest:
        logger.debug(
            "Adjusted start date for year %s from %s to earliest allowed %s", year, start_date, earliest
        )
        return earliest
    logger.debug("Computed year-start date for %s: %s", year, start_date)
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
    OHLCV ingestion pipeline:
    1. Check which years need processing (based on processed S3 keys)
    2. Download raw OHLCV data from S3
    3. For each year to process:
       a. Filter raw OHLCV by date range
       b. Clean OHLCV data
       c. Upload processed OHLCV to S3
    """

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
        if symbol not in symbols_constants.NSE_SYMBOLS:
            raise ValueError(f"Symbol {symbol!r} is not a valid NSE symbol.")

    @staticmethod
    def _processed_s3_object_key(prefix: str, symbol: str, year: int, filename: str) -> str:
        # e.g., processesd/ohlcv/TCS/2023/ohlcv_processed.parquet
        return f"{prefix}/{symbol}/{year}/{filename}"
    
    @staticmethod
    def _raw_s3_object_key(prefix: str, symbol: str, raw_filename: str) -> str:
        # e.g., raw/ohlcv/TCS/ohlcv_raw.parquet
        return f"{prefix}/{symbol}/{raw_filename}"

    def prepare_year_prefixes(self) -> Tuple[List[int], List[str]]:
        years = list(range(self.config.start_year, self.config.end_year + 1))
        processed_keys = [self._processed_s3_object_key(self.config.ohlcv_processed_prefix, self.config.symbol, y, self.config.ohlcv_processed_filename) for y in years]

        remaining_years: List[int] = []
        remaining_processed: List[str] = []
        for y, key in zip(years, processed_keys):
            if self.s3.exists(key):
                logger.info("OHLCV exists; skipping year=%s at s3://%s/%s", y, self.config.s3_bucket, key)
            else:
                remaining_years.append(y)
                remaining_processed.append(key)

        logger.debug("Years pending OHLCV ingestion for %s: %s", self.config.symbol, remaining_years)
        return remaining_years, remaining_processed
    
    def fetch_raw_data(self) -> pd.DataFrame:
        raw_key = self._raw_s3_object_key(self.config.ohlcv_raw_prefix, self.config.symbol, self.config.ohlcv_raw_filename)
        logger.info("Reading RAW candles from s3://%s/%s", self.config.s3_bucket, raw_key)

        with tempfile.TemporaryDirectory() as td:
            raw_local = os.path.join(td, f"{self.config.symbol}_raw.parquet")
            if not self.s3.exists(raw_key):
                raise FileNotFoundError(f"RAW candles not found at s3://{self.config.s3_bucket}/{raw_key}")

            self.s3.download(raw_key, raw_local)
            raw_df = pd.read_parquet(raw_local)

            logger.info("Raw rows: %d", len(raw_df))
            logger.debug("Raw OHLCV columns for %s: %s", self.config.symbol, raw_df.columns.tolist())

            daily_df = aggregate_intraday_to_daily(raw_df)
            logger.info("Daily rows: %d", len(daily_df))

            return daily_df

    def ingest_year(self, year: int, processed_key: str, raw_df: pd.DataFrame, daily_start: dt.date, daily_end: dt.date) -> None:
        start_date = _year_start_date(year, daily_start)
        end_date = _year_end_date(year, daily_end)

        logger.info("Ingesting OHLCV for year %s: %s to %s", year, start_date, end_date)

        logger.info("Filtering OHLCV for year %s: %s to %s...", year, start_date, end_date)
        df = raw_df[(raw_df["date"] >= start_date) & (raw_df["date"] <= end_date)]
        logger.debug("Filtered %s OHLCV rows for %s year %s", len(df), self.config.symbol, year)
        if df.empty:
            logger.warning("No OHLCV data for %s after filtering %s → %s", self.config.symbol, start_date, end_date)
            return

        df = clean_ohlcv_data(df, self.config.symbol)
        if df.empty:
            raise RuntimeError(f"[VALIDATION] OHLCV empty for {self.config.symbol} year={year} start={start_date} end={end_date} after cleaning.")

        logger.info("Cleaned OHLCV has %s rows. Saving and uploading to S3...", len(df))
        self.config.local_output.parent.mkdir(parents=True, exist_ok=True)

        local_path = self.config.local_output.with_name(
            f"{self.config.local_output.stem}_{self.config.symbol}_{year}_ohlcv.parquet"
        )
        logger.debug("Writing OHLCV parquet to %s", local_path)
        df.to_parquet(local_path, index=False)
        self.s3.upload(local_path, processed_key)
        logger.info("Uploaded OHLCV to s3://%s/%s", self.config.s3_bucket, processed_key)

    def run(self) -> None:
        logger.info("Starting OHLCV ingestion for %s (years %s-%s)...", self.config.symbol, self.config.start_year, self.config.end_year)

        if self.config.start_year > self.config.end_year:
            raise ValueError("Start year cannot be after end year.")

        years_to_process, processed_keys = self.prepare_year_prefixes()
        if not processed_keys:
            logger.info("All OHLCV data already exists in S3. Nothing to ingest.")
            return

        logger.info("Need to ingest OHLCV data for %s year(s).", len(processed_keys))

        daily_df = self.fetch_raw_data()
        earliest = daily_df["date"].min().date()
        latest = daily_df["date"].max().date()
        logger.info("Available daily OHLCV date range: %s to %s", earliest, latest)

        for year, key in zip(years_to_process, processed_keys):
            self.ingest_year(year, key, daily_df, earliest, latest)

        logger.info("Raw to processed ingestion pipeline for OHLCV completed.")


class NewsIngestor:
    """
    News ingestion pipeline:
    1. Check which years need processing (based on processed S3 keys)
    2. Download raw news data from S3
    3. For each year to process:
       a. Filter raw news by date range
       b. Clean news data
       c. Upload processed news to S3
    4. (Optional) Apply relevance scoring layer
    """

    def __init__(
        self,
        config: IngestConfig,
        s3_client_factory: Callable[[str], S3Client] = S3Client,
        # scorer: Optional[RelevanceScorer] = None,
    ):
        self.config = config
        self.s3 = s3_client_factory(config.s3_bucket)
        # self.scorer = scorer or RelevanceScorer()

    @staticmethod
    def _processed_s3_object_key(prefix: str, symbol: str, year: int, processed_filename: str) -> str:
        # e.g., processesd/ohlcv/TCS/2023/news_processed.parquet
        return f"{prefix}/{symbol}/{year}/{processed_filename}"
    
    @staticmethod
    def _raw_s3_object_key(prefix: str, symbol: str, raw_filename: str) -> str:
        # e.g., raw/ohlcv/TCS/news_raw.parquet
        return f"{prefix}/{symbol}/{raw_filename}"

    def prepare_year_prefixes(self) -> Tuple[List[int], List[str]]:
        years = list(range(self.config.start_year, self.config.end_year + 1))
        processed_keys = [self._processed_s3_object_key(self.config.news_processed_prefix, self.config.symbol, y, self.config.news_processed_filename) for y in years]

        remaining_years: List[int] = []
        remaining_processed: List[str] = []

        for y, ck in zip(years, processed_keys):
            # If relevance layer exists, consider year done
            if self.s3.exists(ck):
                logger.info("NEWS exists; skipping year=%s at s3://%s/%s", y, self.config.s3_bucket, ck)
            else:
                remaining_years.append(y)
                remaining_processed.append(ck)

        logger.debug("Years pending NEWS ingestion for %s: %s", self.config.symbol, remaining_years)
        return remaining_years, remaining_processed
    
    def fetch_raw_data(self) -> pd.DataFrame:
        raw_key = self._raw_s3_object_key(self.config.news_raw_prefix, self.config.symbol, self.config.news_raw_filename)
        logger.info("Reading RAW news from s3://%s/%s", self.config.s3_bucket, raw_key)

        with tempfile.TemporaryDirectory() as td:
            raw_local = os.path.join(td, f"{self.config.symbol}_raw.parquet")
            if not self.s3.exists(raw_key):
                raise FileNotFoundError(f"RAW news not found at s3://{self.config.s3_bucket}/{raw_key}")

            self.s3.download(raw_key, raw_local)
            raw_df = pd.read_parquet(raw_local)

            logger.info("Raw rows: %d", len(raw_df))
            logger.debug("Raw NEWS columns for %s: %s", self.config.symbol, raw_df.columns.tolist())

            return raw_df

    def ingest_year(self, year: int, processed_key: str, raw_df: pd.DataFrame, earliest: dt.date, latest: dt.date) -> None:
        start_date = _year_start_date(year, earliest)
        end_date = _year_end_date(year, latest)

        logger.info("Ingesting NEWS for %s from %s to %s...", self.config.symbol, start_date, end_date)

        logger.info("Filtering NEWS for year %s: %s to %s...", year, start_date, end_date)

        raw_df = raw_df[(raw_df["_dt"].dt.date >= start_date) & (raw_df["_dt"].dt.date <= end_date)].drop(columns=["_dt"])
        logger.debug("Filtered %d NEWS rows for %s year %s", len(raw_df), self.config.symbol, year)

        news_clean = clean_desiq_news_data(raw_df, symbol=self.config.symbol)
        logger.info("Cleaned NEWS has %s rows for %s year %s", len(news_clean), self.config.symbol, year)

        # write local -> upload (keep same pattern you use elsewhere)
        out_dir = self.config.local_output / "processed" / "news" / self.config.symbol / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / self.config.news_processed_filename
        logger.debug("Writing cleaned NEWS parquet to %s", out_path)
        news_clean.to_parquet(out_path, index=False)

        self.s3.upload(out_path, processed_key)
        logger.info("Uploaded processed news to s3://%s/%s", self.config.s3_bucket, processed_key)

    def run(self) -> None:
        logger.info("Starting News ingestion for %s (years %s-%s)...", self.config.symbol, self.config.start_year, self.config.end_year)

        if self.config.start_year > self.config.end_year:
            raise ValueError("Start year cannot be after end year.")

        years, processed_keys = self.prepare_year_prefixes()
        if not processed_keys:
            logger.info("All NEWS data already exists in S3. Nothing to ingest.")
            return

        logger.info("Need to ingest NEWS data for %s year(s).", len(processed_keys))

        raw_df = self.fetch_raw_data()

        # Filter by date window (year)
        if "date" in raw_df.columns:
            tmp_dt = pd.to_datetime(raw_df["date"], errors="coerce", utc=True)
        elif "published_at" in raw_df.columns:
            tmp_dt = pd.to_datetime(raw_df["published_at"], errors="coerce", utc=True)
        else:
            # desiquant news uses 'date' string column in raw dumps; if parquet kept it as 'date', above handles it.
            tmp_dt = pd.to_datetime(raw_df.get("date"), errors="coerce", utc=True)

        raw_df = raw_df.assign(_dt=tmp_dt)
        invalid_dates = raw_df["_dt"].isna().sum()
        raw_df = raw_df.dropna(subset=["_dt"])
        if invalid_dates:
            logger.debug("Dropped %s raw NEWS rows with invalid dates for %s", invalid_dates, self.config.symbol)
        
        earliest = raw_df["_dt"].min().date()
        latest = raw_df["_dt"].max().date()
        logger.info("Available raw NEWS date range: %s to %s", earliest, latest)

        for y, pk in zip(years, processed_keys):
            self.ingest_year(y, pk, raw_df, earliest, latest)

        logger.info("Raw to processed ingestion pipeline for News completed.")


class CorporateAnnouncementsIngestor:
    """
    Corporate Announcements ingestion pipeline:
    1. Check which years need processing (based on processed S3 keys)
    2. Download raw Corporate Announcements data from S3
    3. For each year to process:
       a. Filter raw Corporate Announcements by date range
       b. Clean Corporate Announcements data
       c. Upload processed Corporate Announcements to S3
    """
    def __init__(self, config: IngestConfig, s3_client_factory: Callable[[str], S3Client] = S3Client):
        self.config = config
        self.s3 = s3_client_factory(config.s3_bucket)
        self.annoucements_sources = desiquant_constants.SUPPORTED_ANNOUNCEMENTS_SOURCES
        self.dtcol_name_mapping = {
            "bse": "news_dt",
            "nse": "sort_date"
        }

    @staticmethod
    def _processed_s3_object_key(prefix: str, symbol: str, year: int, source:str, processed_filename: str) -> str:
        # e.g., processed/corporate_announcements/TCS/2023/bse/corporate_announcements_processed.parquet
        return f"{prefix}/{symbol}/{year}/{source}/{processed_filename}"
    
    @staticmethod
    def _raw_s3_object_key(prefix: str, symbol: str, source: str, raw_filename: str) -> str:
        # e.g., raw/desiquant/corporate_announcements/TCS/bse/corporate_announcements_raw.parquet
        return f"{prefix}/{symbol}/{source}/{raw_filename}"

    def prepare_year_prefixes(self, source: str) -> Tuple[List[int], List[str]]:
        years = list(range(self.config.start_year, self.config.end_year + 1))
        processed_keys = [self._processed_s3_object_key(self.config.announcements_processed_prefix, self.config.symbol, y, source ,self.config.announcements_processed_filename) for y in years]

        remaining_years: List[int] = []
        remaining_processed: List[str] = []
        for y, key in zip(years, processed_keys):
            if self.s3.exists(key):
                logger.info("Corporate Announcements exists; skipping year=%s at s3://%s/%s", y, self.config.s3_bucket, key)
            else:
                remaining_years.append(y)
                remaining_processed.append(key)

        logger.debug("Years pending Corporate Announcements ingestion for %s (%s): %s", self.config.symbol, source, remaining_years)
        return remaining_years, remaining_processed

    def fetch_raw_data(self, source: str) -> pd.DataFrame:
        raw_key = self._raw_s3_object_key(self.config.announcements_raw_prefix, self.config.symbol, source, self.config.announcements_raw_filename)
        logger.info("Reading RAW corporate announcements from s3://%s/%s", self.config.s3_bucket, raw_key)

        with tempfile.TemporaryDirectory() as td:
            raw_local = os.path.join(td, f"{self.config.symbol}_raw.parquet")
            if not self.s3.exists(raw_key):
                raise FileNotFoundError(f"RAW corporate announcements not found at s3://{self.config.s3_bucket}/{raw_key}")

            self.s3.download(raw_key, raw_local)
            raw_df = pd.read_parquet(raw_local)

            logger.info("Raw rows: %d", len(raw_df))
            logger.debug("Raw corporate announcements columns for %s/%s: %s", self.config.symbol, source, raw_df.columns.tolist())

            return raw_df

    def ingest_year(self, year: int, processed_key: str, raw_df: pd.DataFrame, earliest: dt.date, latest: dt.date, source: str) -> None:
        start_date = _year_start_date(year, earliest)
        end_date = _year_end_date(year, latest)

        logger.info("Ingesting Corporate Announcements for year %s: %s to %s", year, start_date, end_date)

        logger.info("Filtering Corporate Announcements for year %s: %s to %s...", year, start_date, end_date)

        filtered_df = raw_df[(raw_df["date"].dt.date >= start_date) & (raw_df["date"].dt.date <= end_date)].drop(columns=["date"])
        logger.debug("Filtered %d Corporate Announcement rows for %s (%s) year %s", len(filtered_df), self.config.symbol, source, year)

        clean_df = clean_desiq_corporate_announcements(filtered_df, symbol=self.config.symbol, source=source)
        logger.info("Cleaned Corporate Announcements has %s rows for %s (%s) year %s", len(clean_df), self.config.symbol, source, year)

        out_dir = self.config.local_output / "processed" / "corporate_announcements" / self.config.symbol / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / self.config.announcements_processed_filename
        logger.debug("Writing cleaned Corporate Announcements parquet to %s", out_path)
        clean_df.to_parquet(out_path, index=False)

        self.s3.upload(out_path, processed_key)
        logger.info("Uploaded processed corp announcements to s3://%s/%s", self.config.s3_bucket, processed_key)

    def run(self) -> None:
        logger.info("Starting Corporate Announcements ingestion for %s (%s-%s)", self.config.symbol, self.config.start_year, self.config.end_year)

        if self.config.start_year > self.config.end_year:
            raise ValueError("Start year cannot be after end year.")

        for source in self.annoucements_sources:
            logger.info("Processing source: %s", source)

            dtcol_name = self.dtcol_name_mapping.get(source.lower())
            logger.debug("Using datetime column %s for source %s", dtcol_name, source)

            years, processed_keys = self.prepare_year_prefixes(source)
            if not processed_keys:
                logger.info("All Corporate Announcements data already exists in S3 for %s. Nothing to ingest.", source)
                continue

            logger.info("Need to ingest Corporate Announcements data for %s year(s).", len(processed_keys))

            raw_df = self.fetch_raw_data(source)

            logger.info("Raw Columns: %s", raw_df.columns.tolist())

            logger.info("Parsing %s column for date filtering...", dtcol_name)
            dtcol = pd.to_datetime(raw_df[dtcol_name], errors="coerce", utc=True)
            raw_df = raw_df.assign(date=dtcol)
            invalid_dates = raw_df["date"].isna().sum()
            raw_df = raw_df.dropna(subset=["date"])
            if invalid_dates:
                logger.debug("Dropped %s Corporate Announcement rows with invalid %s for %s", invalid_dates, dtcol_name, source)

            earliest = raw_df["date"].min().date()
            latest = raw_df["date"].max().date()
            logger.info("Available raw Corporate Announcements date range: %s to %s", earliest, latest)

            for y, pk in zip(years, processed_keys):
                self.ingest_year(y, pk, raw_df, earliest, latest, source)

            logger.info("Completed ingestion for source: %s", source)

        logger.info("Raw to processed ingestion pipeline for Corporate Announcements completed.")


class FinancialResultsIngestor:
    """
    Financial Results ingestion pipeline:
    1. Check which years need processing (based on processed S3 keys)
    2. Download raw Financial Results data from S3
    3. For each year to process:
       a. Filter raw Financial Results by date range
       b. Clean Financial Results data
       c. Upload processed Financial Results to S3
    """
    def __init__(self, config: IngestConfig, s3_client_factory: Callable[[str], S3Client] = S3Client):
        self.config = config
        self.s3 = s3_client_factory(config.s3_bucket)
        self.dtcol_name = "filingdate"

    @staticmethod
    def _processed_s3_object_key(prefix: str, symbol: str, year: int, processed_filename: str) -> str:
        # e.g., processed/financial_results/TCS/2023/financial_results_processed.parquet
        return f"{prefix}/{symbol}/{year}/{processed_filename}"
    
    @staticmethod
    def _raw_s3_object_key(prefix: str, symbol: str, raw_filename: str) -> str:
        # e.g., raw/desiquant/financial_results/TCS/financial_results_raw.parquet
        return f"{prefix}/{symbol}/{raw_filename}"

    def prepare_year_prefixes(self) -> Tuple[List[int], List[str]]:
        years = list(range(self.config.start_year, self.config.end_year + 1))
        processed_keys = [self._processed_s3_object_key(self.config.results_raw_prefix, self.config.symbol, y, self.config.results_processed_filename) for y in years]

        remaining_years: List[int] = []
        remaining_processed: List[str] = []
        for y, key in zip(years, processed_keys):
            if self.s3.exists(key):
                logger.info("Financial Results exists; skipping year=%s at s3://%s/%s", y, self.config.s3_bucket, key)
            else:
                remaining_years.append(y)
                remaining_processed.append(key)

        logger.debug("Years pending Financial Results ingestion for %s: %s", self.config.symbol, remaining_years)
        return remaining_years, remaining_processed

    def fetch_raw_data(self) -> pd.DataFrame:
        raw_key = self._raw_s3_object_key(self.config.results_raw_prefix, self.config.symbol, self.config.results_raw_filename)
        logger.info("Reading RAW financial results from s3://%s/%s", self.config.s3_bucket, raw_key)

        with tempfile.TemporaryDirectory() as td:
            raw_local = os.path.join(td, f"{self.config.symbol}_raw.parquet")
            if not self.s3.exists(raw_key):
                raise FileNotFoundError(f"RAW financial results not found at s3://{self.config.s3_bucket}/{raw_key}")

            self.s3.download(raw_key, raw_local)
            raw_df = pd.read_parquet(raw_local)

            logger.info("Raw rows: %d", len(raw_df))
            logger.debug("Raw Financial Results columns for %s: %s", self.config.symbol, raw_df.columns.tolist())

            return raw_df

    def ingest_year(self, year: int, processed_key: str, raw_df: pd.DataFrame, earliest: dt.date, latest: dt.date) -> None:
        start_date = _year_start_date(year, earliest)
        end_date = _year_end_date(year, latest)

        logger.info("Ingesting Financial Results for year %s: %s to %s", year, start_date, end_date)

        logger.info("Filtering Financial Results for year %s: %s to %s...", year, start_date, end_date)

        filtered_df = raw_df[(raw_df["date"].dt.date >= start_date) & (raw_df["date"].dt.date <= end_date)].drop(columns=["date"])
        logger.debug("Filtered %d Financial Results rows for %s year %s", len(filtered_df), self.config.symbol, year)
        clean_df = clean_desiq_financial_results(filtered_df, symbol=self.config.symbol)
        logger.info("Cleaned Financial Results has %s rows for %s year %s", len(clean_df), self.config.symbol, year)

        out_dir = self.config.local_output / "processed" / "financial_results" / self.config.symbol / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / self.config.results_processed_filename
        logger.debug("Writing cleaned Financial Results parquet to %s", out_path)
        clean_df.to_parquet(out_path, index=False)

        self.s3.upload(out_path, processed_key)
        logger.info("Uploaded processed financial results to s3://%s/%s", self.config.s3_bucket, processed_key)

    def run(self) -> None:
        logger.info("Starting Financial Results ingestion for %s (%s-%s)", self.config.symbol, self.config.start_year, self.config.end_year)

        if self.config.start_year > self.config.end_year:
            raise ValueError("Start year cannot be after end year.")

        years, processed_keys = self.prepare_year_prefixes()
        if not processed_keys:
            logger.info("All Financial Results data already exists in S3. Nothing to ingest.")
            return

        logger.info("Need to ingest Financial Results data for %s year(s).", len(processed_keys))

        raw_df = self.fetch_raw_data()

        logger.info("Raw Columns: %s", raw_df.columns.tolist())

        logger.info("Parsing %s column for date filtering...", self.dtcol_name)

        # Filter by year using filingdate (best signal)
        dtcol = pd.to_datetime(raw_df[self.dtcol_name], errors="coerce", utc=True)
        raw_df = raw_df.assign(date=dtcol)
        invalid_dates = raw_df["date"].isna().sum()
        raw_df = raw_df.dropna(subset=["date"])
        if invalid_dates:
            logger.debug("Dropped %s Financial Results rows with invalid %s", invalid_dates, self.dtcol_name)
            
        earliest = raw_df["date"].min().date()
        latest = raw_df["date"].max().date()
        logger.info("Available raw Financial Results date range: %s to %s", earliest, latest)

        for y, pk in zip(years, processed_keys):
            self.ingest_year(y, pk, raw_df, earliest, latest)

        logger.info("Raw to processed ingestion pipeline for Financial Results completed.")



class MultiModalIngestor:
    """
    Orchestrates running OHLCV ingestion and News ingestion sequentially
    with the same CLI inputs (symbol, start, end).
    """

    def __init__(self, config: IngestConfig):
        self.config = config

    def run(self, run_ohlcv: bool = True, run_news: bool = True, run_corp_ann: bool = True, run_fin_results: bool = True, log_level: Optional[str] = None) -> None:
        setup_logging(log_level)
        logger.info(
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

        if run_corp_ann:
            CorporateAnnouncementsIngestor(self.config).run()

        if run_fin_results:
            FinancialResultsIngestor(self.config).run()

        logger.info("Multi-modal ingestion finished. (OHLCV=%s, NEWS=%s)", run_ohlcv, run_news)


# ---- CLI ----
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-modal ingestion pipeline (OHLCV + News + Relevance).")
    parser.add_argument("--symbol", type=str, required=True, help="Single NSE symbol to ingest (e.g. TCS).")
    parser.add_argument("--start", type=str, required=True, help="Start year YYYY")
    parser.add_argument("--end", type=str, required=True, help="End year YYYY")
    parser.add_argument("--local-output", type=str, default="/tmp/ingest.parquet", help="Local temp parquet stem")
    parser.add_argument("--s3-bucket", type=str, default=storage_constants.S3_BUCKET, help="S3 bucket name")
    parser.add_argument("--ohlcv-only", action="store_true", help="Only ingest OHLCV.")
    parser.add_argument("--news-only", action="store_true", help="Only ingest news.")
    parser.add_argument("--corporate-announcements-only", action="store_true", help="Only ingest Corporate Announcements.")
    parser.add_argument("--financial-results-only", action="store_true", help="Only ingest Financial Results.")
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
    run_corp_ann = True
    run_fin_results = True
    if args.ohlcv_only:
        run_news = False
        run_corp_ann = False
        run_fin_results = False
    if args.news_only:
        run_ohlcv = False
        run_corp_ann = False
        run_fin_results = False
    if args.corporate_announcements_only:
        run_ohlcv = False
        run_news = False
        run_fin_results = False
    if args.financial_results_only:
        run_ohlcv = False
        run_news = False
        run_corp_ann = False

    config = IngestConfig(
        symbol=symbol,
        start_year=start_year,
        end_year=end_year,
        local_output=Path(args.local_output),
        s3_bucket=args.s3_bucket,
    )

    MultiModalIngestor(config).run(run_ohlcv=run_ohlcv, run_news=run_news, run_corp_ann=run_corp_ann, run_fin_results=run_fin_results, log_level=args.log_level)


if __name__ == "__main__":
    main()
