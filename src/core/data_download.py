# src/core/data_download.py

# Standard library imports
import os
import io
import json
import time
import random
import zipfile
import logging
import datetime as dt
import tempfile
from typing import List
from urllib.parse import quote_plus

# Third-party imports
import feedparser
import pandas as pd
import requests

# Optional last-resort fallback
import yfinance as yf

from jugaad_data.nse import stock_df
import jugaad_data.util as jugaad_util
from requests.exceptions import JSONDecodeError as RequestsJSONDecodeError

# Local imports
import src.constants as constants
from src.io.connections import S3Connection

logger = logging.getLogger(__name__)

# -------------------------
# Helpers
# -------------------------
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
    "Connection": "keep-alive",
}


def _daterange(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def _patch_jugaad_cache_makedirs() -> None:
    """
    jugaad-data creates its cache directory in multiple threads without
    exist_ok=True, which can raise FileExistsError. Patch makedirs to
    swallow that specific race.
    """
    if getattr(jugaad_util, "_makedirs_patched", False):
        return

    logger.debug("Patching jugaad-data makedirs to be race-safe")
    original_makedirs = jugaad_util.os.makedirs

    def _safe_makedirs(path, *args, **kwargs):
        try:
            original_makedirs(path, *args, **kwargs)
        except FileExistsError:
            return

    jugaad_util.os.makedirs = _safe_makedirs  # type: ignore[attr-defined]
    jugaad_util._makedirs_patched = True


# -------------------------
# Provider 1 (Recommended for GitHub Actions): NSE Bhavcopy
# -------------------------
def _bhavcopy_urls(d: dt.date) -> List[str]:
    """
    Try a couple of known NSE archive patterns.
    NSE has changed these over time; we try multiple.
    """
    logger.debug("[bhavcopy] Building archive URLs for %s", d)
    dd = d.strftime("%d")
    mmm = d.strftime("%b").upper()          # e.g. DEC
    yyyy = d.strftime("%Y")
    ddmmmyyyy = d.strftime("%d%b%Y").upper()  # e.g. 19DEC2025

    # Common patterns historically used by NSE archives
    return [
        f"https://archives.nseindia.com/content/historical/EQUITIES/{yyyy}/{mmm}/cm{ddmmmyyyy}bhav.csv.zip",
        f"https://www1.nseindia.com/content/historical/EQUITIES/{yyyy}/{mmm}/cm{ddmmmyyyy}bhav.csv.zip",
    ]


def _download_bhavcopy_zip_for_day(session: requests.Session, d: dt.date) -> pd.DataFrame:
    last_err = None
    for url in _bhavcopy_urls(d):
        try:
            logger.debug("[bhavcopy] Requesting %s", url)
            r = session.get(url, headers=DEFAULT_HEADERS, timeout=60)
            if r.status_code == 200 and r.content:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                # usually single CSV inside
                csv_name = z.namelist()[0]
                with z.open(csv_name) as fp:
                    df = pd.read_csv(fp)
                return df
            else:
                last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = repr(e)
            continue
    raise RuntimeError(f"Bhavcopy not available for {d} (last_err={last_err})")


def _download_ohlcv_bhavcopy(symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """
    Build OHLCV for a symbol by reading daily bhavcopy files.
    This is slower than a single API call, but *reliable on GitHub Actions*.
    """
    logger.info("[bhavcopy] Downloading OHLCV for %s from %s to %s", symbol, start_date, end_date)
    symbol = symbol.upper()

    session = requests.Session()

    rows = []
    failures = 0

    for d in _daterange(start_date, end_date):
        # NSE closed on weekends; skip to reduce downloads
        if d.weekday() >= 5:
            logger.debug("[bhavcopy] Skipping weekend %s", d)
            continue

        try:
            day_df = _download_bhavcopy_zip_for_day(session, d)

            # Column names in bhavcopy typically:
            # SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE, TOTTRDQTY, TIMESTAMP, ...
            # We only need EQ series.
            if "SYMBOL" not in day_df.columns:
                continue

            day_df = day_df[day_df["SYMBOL"].astype(str).str.upper() == symbol]
            if "SERIES" in day_df.columns:
                day_df = day_df[day_df["SERIES"].astype(str).str.upper() == "EQ"]

            if day_df.empty:
                continue

            r0 = day_df.iloc[0]

            rows.append(
                {
                    "date": pd.to_datetime(d),
                    "open": float(r0["OPEN"]),
                    "high": float(r0["HIGH"]),
                    "low": float(r0["LOW"]),
                    "close": float(r0["CLOSE"]),
                    "volume": float(r0["TOTTRDQTY"]),
                    "symbol": symbol,
                }
            )

            # Gentle pacing to be polite
            time.sleep(0.15 + random.uniform(0, 0.15))

        except Exception as e:
            failures += 1
            # don’t fail entire year on a couple missing holidays / transient errors
            logger.debug("[bhavcopy] Failed for %s: %s", d, e)
            time.sleep(0.3)

    if not rows:
        raise ValueError(f"[bhavcopy] No OHLCV rows built for {symbol} in {start_date}->{end_date}")

    df = pd.DataFrame(rows)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("[bhavcopy] Completed %s rows (failures=%d)", len(df), failures)
    return df


# -------------------------
# Provider 2: jugaad (kept as secondary)
# -------------------------
def _download_ohlcv_jugaad(symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    logger.info("[jugaad] Downloading OHLCV for %s from %s to %s", symbol, start_date, end_date)
    _patch_jugaad_cache_makedirs()
    df = stock_df(symbol=symbol, from_date=start_date, to_date=end_date)

    if df.empty:
        raise ValueError(f"[jugaad] No OHLCV data downloaded for symbol {symbol!r}.")

    logger.debug("[jugaad] Raw OHLCV data shape for %s: %s", symbol, df.shape)
    df = df.reset_index().rename(
        columns={
            "DATE": "date",
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "TOTTRDQTY": "volume",
        }
    )
    df["symbol"] = symbol
    logger.info("[jugaad] Completed OHLCV download for %s with %s rows", symbol, len(df))
    return df[["date", "open", "high", "low", "close", "volume", "symbol"]]


# -------------------------
# Provider 3: yfinance (last resort; often 429 on GH Actions)
# -------------------------
def _normalize_yf_frame(data: pd.DataFrame, symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    if data is None or len(data) == 0:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]

    data = data.reset_index()

    data.rename(
        columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"},
        inplace=True,
    )

    needed = ["date", "open", "high", "low", "close", "volume"]
    if any(c not in data.columns for c in needed):
        return pd.DataFrame()

    df = data[needed].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]
    df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]
    df["symbol"] = symbol
    return df[["date", "open", "high", "low", "close", "volume", "symbol"]]


def _download_ohlcv_yfinance(symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    ticker = f"{symbol}.NS"
    logger.warning("[yfinance] Fallback OHLCV for %s using ticker %s", symbol, ticker)

    end_plus = end_date + dt.timedelta(days=1)
    max_attempts = 6
    base_sleep = 2.0
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=start_date.isoformat(), end=end_plus.isoformat(), interval="1d", auto_adjust=False)
            df = _normalize_yf_frame(hist, symbol, start_date, end_date)
            if not df.empty:
                return df

            data = yf.download(
                tickers=ticker,
                start=start_date.isoformat(),
                end=end_plus.isoformat(),
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            df = _normalize_yf_frame(data, symbol, start_date, end_date)
            if not df.empty:
                return df

            last_err = ValueError("Empty response (likely throttled)")
            raise last_err

        except Exception as e:
            last_err = e
            sleep_s = min(60.0, base_sleep * (2 ** (attempt - 1))) + random.uniform(0, 1.5)
            logger.warning("[yfinance] Attempt %d/%d failed for %s: %s. Sleeping %.1fs",
                           attempt, max_attempts, ticker, f"{type(e).__name__}: {e}", sleep_s)
            time.sleep(sleep_s)

    raise ValueError(f"[yfinance] No OHLCV data returned for {symbol!r} ({ticker}). Last error: {last_err}")


# -------------------------
# Public function used by pipeline
# -------------------------
def download_ohlcv_nsepy(symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """
    Order of attempts (best for GitHub Actions reliability):
      1) NSE Bhavcopy (archives)
      2) jugaad-data (NSE API)
      3) yfinance (last resort; may 429 on GH runners)
    """
    logger.info("Request received to download OHLCV for %s (%s -> %s)", symbol, start_date, end_date)
    symbol = symbol.strip().upper()

    # 1) bhavcopy
    try:
        return _download_ohlcv_bhavcopy(symbol, start_date, end_date)
    except Exception as e:
        logger.warning("[bhavcopy] Failed: %s. Trying jugaad...", e)

    # 2) jugaad
    try:
        return _download_ohlcv_jugaad(symbol, start_date, end_date)
    except (json.JSONDecodeError, RequestsJSONDecodeError) as exc:
        logger.warning("[jugaad] NSE returned non-JSON; trying yfinance. Error: %s", exc)
    except Exception as exc:
        logger.warning("[jugaad] Failed; trying yfinance. Error: %s", exc)

    # 3) yfinance
    return _download_ohlcv_yfinance(symbol, start_date, end_date)


# -------------------------
# News: Google News RSS (unchanged)
# -------------------------
def _build_google_news_queries(symbol: str) -> List[str]:
    company = constants.SYMBOL_TO_COMPANY.get(symbol, symbol)
    queries = [
        f'"{company}" stock',
        f'"{company}" shares',
        f'"{company}" results',
        f"{company} NSE",
        f"{symbol} stock",
    ]
    seen, out = set(), []
    for q in queries:
        if q not in seen:
            out.append(q)
            seen.add(q)
    return out


def _google_news_rss_url(query: str) -> str:
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"


def fetch_news_from_provider(
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
    max_items_per_query: int = 50,
    sleep_s: float = 0.6,
) -> pd.DataFrame:
    logger.info("[news] Fetching news for %s between %s and %s", symbol, start_date, end_date)
    symbol = symbol.strip().upper()
    queries = _build_google_news_queries(symbol)
    logger.debug("[news] Built %d queries for %s: %s", len(queries), symbol, queries)

    rows = []
    for q in queries:
        url = _google_news_rss_url(q)
        logger.debug("[news] Fetching RSS for query %r", q)
        feed = feedparser.parse(url)
        entries = getattr(feed, "entries", []) or []
        logger.debug("[news] Retrieved %d entries for query %r", len(entries), q)
        for e in entries[:max_items_per_query]:
            rows.append(
                {
                    "headline": getattr(e, "title", None),
                    "source": getattr(getattr(e, "source", None), "title", None) if hasattr(e, "source") else None,
                    "published_at": getattr(e, "published", None) or getattr(e, "updated", None),
                    "link": getattr(e, "link", None),
                    "summary": getattr(e, "summary", None),
                    "query_used": q,
                }
            )
        time.sleep(sleep_s)

    if not rows:
        logger.info("[news] No news rows collected for %s in given range", symbol)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.dropna(subset=["headline"], inplace=True)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)

    df["date"] = df["published_at"].dt.date
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    if df["link"].notna().any():
        df.drop_duplicates(subset=["link"], inplace=True)
    else:
        df.drop_duplicates(subset=["headline"], inplace=True)

    df.drop(columns=["date"], inplace=True, errors="ignore")
    df.sort_values(["published_at", "source"], inplace=True, na_position="last")
    df.reset_index(drop=True, inplace=True)
    return df


class DesiquantCandlesMirror:
    """Handles download of Desiquant candles and mirroring to the raw S3 zone."""

    def __init__(self):
        self.constants = constants
        self.raw_bucket = constants.S3_BUCKET
        self.raw_prefix = constants.RAW_DESIQUANT_OHLCV_PREFIX
        self.default_segment = constants.DEFAULT_CANDLES_SEGMENT
        self.desiquant_bucket = constants.DESIQUANT_CANDLES_BUCKET
        self.raw_s3 = S3Connection(bucket=self.raw_bucket)

    def _desiquant_storage_options(self):
        return {
            "endpoint_url": self.constants.DESIQUANT_ENDPOINT_URL,
            "key": self.constants.DESIQUANT_ACCESS_KEY,
            "secret": self.constants.DESIQUANT_SECRET_KEY,
            "client_kwargs": {
                "region_name": "auto"
            },
        }

    def build_raw_candles_key(self, symbol: str, segment: str | None = None) -> str:
        segment = segment or self.default_segment
        return f"{self.raw_prefix}/{symbol}/{segment}.parquet"

    def build_raw_candles_s3_uri(self, symbol: str, segment: str | None = None) -> str:
        key = self.build_raw_candles_key(symbol, segment)
        return f"s3://{self.raw_bucket}/{key}"

    def read_desiquant_candles_df(self, symbol: str, segment: str | None = None) -> pd.DataFrame:
        """
        Reads Desiquant candles parquet.gz into a DataFrame using S3-compatible access.
        """
        segment = segment or self.default_segment
        key = self.constants.DESIQUANT_CANDLES_KEY_TEMPLATE.format(symbol=symbol, segment=segment)
        src_uri = f"s3://{self.desiquant_bucket}/{key}"

        storage_options = self._desiquant_storage_options()
        logger.info("[RAW] Reading Desiquant candles for %s from %s", symbol, src_uri)
        df = pd.read_parquet(src_uri, storage_options=storage_options)
        logger.debug("[RAW] Retrieved %d rows for %s", len(df), symbol)
        return df

    def _write_parquet_to_raw(self, df: pd.DataFrame, key: str) -> None:
        """Write parquet to raw S3 via the shared connection helper."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            df.to_parquet(tmp.name, index=False)
            temp_path = tmp.name

        try:
            self.raw_s3.upload_file(temp_path, key)
        finally:
            try:
                os.remove(temp_path)
            except OSError as exc:  # pragma: no cover - best-effort cleanup
                logger.warning("Failed to clean up temp parquet %s: %s", temp_path, exc)

    def mirror_to_raw(self, symbol: str, overwrite: bool = False, segment: str | None = None) -> str:
        """
        Downloads candles for a symbol from Desiquant and stores it in your S3 raw zone.
        """
        raw_key = self.build_raw_candles_key(symbol, segment)
        raw_uri = self.build_raw_candles_s3_uri(symbol, segment)

        if (not overwrite) and self.raw_s3.object_exists(raw_key):
            logger.info("[RAW] Exists, skipping: %s", raw_uri)
            return raw_uri

        logger.info("[RAW] Fetching Desiquant candles for %s", symbol)
        df = self.read_desiquant_candles_df(symbol=symbol, segment=segment)

        # Optional: light normalization (column lowercase)
        df.columns = [c.strip().lower() for c in df.columns]

        logger.info("[RAW] Writing to: %s", raw_uri)
        self._write_parquet_to_raw(df, raw_key)
        return raw_uri


class DesiquantNewsMirror:
    """Handles download of Desiquant news and mirroring to the raw S3 zone."""

    def __init__(self):
        self.constants = constants
        self.raw_bucket = constants.S3_BUCKET
        self.raw_prefix = constants.RAW_DESIQUANT_NEWS_PREFIX
        self.desiquant_bucket = constants.DESIQUANT_NEWS_BUCKET
        self.raw_s3 = S3Connection(bucket=self.raw_bucket)

    def _desiquant_storage_options(self):
        return {
            "endpoint_url": self.constants.DESIQUANT_ENDPOINT_URL,
            "key": self.constants.DESIQUANT_ACCESS_KEY,
            "secret": self.constants.DESIQUANT_SECRET_KEY,
            "client_kwargs": {
                "region_name": "auto"
            },
        }

    def build_raw_news_key(self, symbol: str) -> str:
        return f"{self.raw_prefix}/{symbol}/news.parquet"

    def build_raw_news_s3_uri(self, symbol: str) -> str:
        key = self.build_raw_news_key(symbol)
        return f"s3://{self.raw_bucket}/{key}"

    def read_desiquant_news_df(self, symbol: str) -> pd.DataFrame:
        """
        Reads Desiquant news parquet into a DataFrame using S3-compatible access.
        """
        key = self.constants.DESIQUANT_NEWS_KEY_TEMPLATE.format(symbol=symbol)
        src_uri = f"s3://{self.desiquant_bucket}/{key}"

        storage_options = self._desiquant_storage_options()
        logger.info("[RAW] Reading Desiquant news for %s from %s", symbol, src_uri)
        df = pd.read_parquet(src_uri, storage_options=storage_options)
        logger.debug("[RAW] Retrieved %d rows for %s", len(df), symbol)
        return df

    def _write_parquet_to_raw(self, df: pd.DataFrame, key: str) -> None:
        """Write parquet to raw S3 via the shared connection helper."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            df.to_parquet(tmp.name, index=False)
            temp_path = tmp.name

        try:
            self.raw_s3.upload_file(temp_path, key)
        finally:
            try:
                os.remove(temp_path)
            except OSError as exc:  # pragma: no cover - best-effort cleanup
                logger.warning("Failed to clean up temp parquet %s: %s", temp_path, exc)

    def mirror_to_raw(self, symbol: str, overwrite: bool = False) -> str:
        """
        Downloads news for a symbol from Desiquant and stores it in your S3 raw zone.
        """
        raw_key = self.build_raw_news_key(symbol)
        raw_uri = self.build_raw_news_s3_uri(symbol)

        if (not overwrite) and self.raw_s3.object_exists(raw_key):
            logger.info("[RAW] Exists, skipping: %s", raw_uri)
            return raw_uri

        logger.info("[RAW] Fetching Desiquant news for %s", symbol)
        df = self.read_desiquant_news_df(symbol=symbol)

        # Optional: light normalization (column lowercase)
        df.columns = [c.strip().lower() for c in df.columns]

        logger.info("[RAW] Writing to: %s", raw_uri)
        self._write_parquet_to_raw(df, raw_key)
        return raw_uri
    

class DesiquantFinancialResultsMirror:
    """Handles download of Desiquant financial results and mirroring to the raw S3 zone."""

    def __init__(self):
        self.constants = constants
        self.raw_bucket = constants.S3_BUCKET
        self.raw_prefix = constants.RAW_DESIQUANT_FINANCIALS_PREFIX
        self.desiquant_bucket = constants.DESIQUANT_FINANCIAL_RESULTS_BUCKET
        self.raw_s3 = S3Connection(bucket=self.raw_bucket)

    def _desiquant_storage_options(self):
        return {
            "endpoint_url": self.constants.DESIQUANT_ENDPOINT_URL,
            "key": self.constants.DESIQUANT_ACCESS_KEY,
            "secret": self.constants.DESIQUANT_SECRET_KEY,
            "client_kwargs": {
                "region_name": "auto"
            },
        }

    def build_raw_results_key(self, symbol: str) -> str:
        return f"{self.raw_prefix}/{symbol}/financial_results.parquet"

    def build_raw_results_s3_uri(self, symbol: str) -> str:
        key = self.build_raw_results_key(symbol)
        return f"s3://{self.raw_bucket}/{key}"

    def read_desiquant_results_df(self, symbol: str) -> pd.DataFrame:
        """
        Reads Desiquant financial results parquet into a DataFrame using S3-compatible access.
        """
        key = self.constants.DESIQUANT_FINANCIAL_RESULTS_KEY_TEMPLATE.format(symbol=symbol)
        src_uri = f"s3://{self.desiquant_bucket}/{key}"

        storage_options = self._desiquant_storage_options()
        logger.info("[RAW] Reading Desiquant financial results for %s from %s", symbol, src_uri)
        df = pd.read_parquet(src_uri, storage_options=storage_options)
        logger.debug("[RAW] Retrieved %d rows for %s", len(df), symbol)
        return df

    def _write_parquet_to_raw(self, df: pd.DataFrame, key: str) -> None:
        """Write parquet to raw S3 via the shared connection helper."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            df.to_parquet(tmp.name, index=False)
            temp_path = tmp.name

        try:
            self.raw_s3.upload_file(temp_path, key)
        finally:
            try:
                os.remove(temp_path)
            except OSError as exc:  # pragma: no cover - best-effort cleanup
                logger.warning("Failed to clean up temp parquet %s: %s", temp_path, exc)

    def mirror_to_raw(self, symbol: str, overwrite: bool = False) -> str:
        """
        Downloads news for a symbol from Desiquant and stores it in your S3 raw zone.
        """
        raw_key = self.build_raw_results_key(symbol)
        raw_uri = self.build_raw_results_s3_uri(symbol)

        if (not overwrite) and self.raw_s3.object_exists(raw_key):
            logger.info("[RAW] Exists, skipping: %s", raw_uri)
            return raw_uri

        logger.info("[RAW] Fetching Desiquant financial results for %s", symbol)
        df = self.read_desiquant_results_df(symbol=symbol)

        # Optional: light normalization (column lowercase)
        df.columns = [c.strip().lower() for c in df.columns]

        logger.info("[RAW] Writing to: %s", raw_uri)
        self._write_parquet_to_raw(df, raw_key)
        return raw_uri


class DesiquantCorporateAnnouncementsMirror:
    """Handles download of Desiquant corporate announcements and mirroring to the raw S3 zone."""

    def __init__(self):
        self.constants = constants
        self.raw_bucket = constants.S3_BUCKET
        self.raw_prefix = constants.RAW_DESIQUANT_ANNOUNCEMENTS_PREFIX
        self.desiquant_bucket = constants.DESIQUANT_CORP_ANNOUNCEMENTS_BUCKET
        self.raw_s3 = S3Connection(bucket=self.raw_bucket)

    def _desiquant_storage_options(self):
        return {
            "endpoint_url": self.constants.DESIQUANT_ENDPOINT_URL,
            "key": self.constants.DESIQUANT_ACCESS_KEY,
            "secret": self.constants.DESIQUANT_SECRET_KEY,
            "client_kwargs": {
                "region_name": "auto"
            },
        }

    def build_raw_announcements_key(self, symbol: str, source: str = constants.DEFAULT_ANNOUNCEMENTS_SOURCE) -> str:
        return f"{self.raw_prefix}/{symbol}/{source}/corporate_announcements.parquet"

    def build_raw_announcements_s3_uri(self, symbol: str, source: str = constants.DEFAULT_ANNOUNCEMENTS_SOURCE) -> str:
        key = self.build_raw_announcements_key(symbol, source)
        return f"s3://{self.raw_bucket}/{key}"

    def read_desiquant_announcements_df(self, symbol: str, source: str = constants.DEFAULT_ANNOUNCEMENTS_SOURCE) -> pd.DataFrame:
        """
        Reads Desiquant corporate announcements parquet into a DataFrame using S3-compatible access.
        """
        key = self.constants.DESIQUANT_CORP_ANNOUNCEMENTS_KEY_TEMPLATE.format(source=source, symbol=symbol)
        src_uri = f"s3://{self.desiquant_bucket}/{key}"

        storage_options = self._desiquant_storage_options()
        logger.info("[RAW] Reading Desiquant corporate announcements for %s from %s", symbol, src_uri)
        df = pd.read_parquet(src_uri, storage_options=storage_options)
        logger.debug("[RAW] Retrieved %d rows for %s", len(df), symbol)
        return df

    def _write_parquet_to_raw(self, df: pd.DataFrame, key: str) -> None:
        """Write parquet to raw S3 via the shared connection helper."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            df.to_parquet(tmp.name, index=False)
            temp_path = tmp.name

        try:
            self.raw_s3.upload_file(temp_path, key)
        finally:
            try:
                os.remove(temp_path)
            except OSError as exc:  # pragma: no cover - best-effort cleanup
                logger.warning("Failed to clean up temp parquet %s: %s", temp_path, exc)

    def mirror_to_raw(self, symbol: str, overwrite: bool = False, source: str = constants.DEFAULT_ANNOUNCEMENTS_SOURCE) -> str:
        """
        Downloads corporate announcements for a symbol from Desiquant and stores it in your S3 raw zone.
        """
        raw_key = self.build_raw_announcements_key(symbol, source)
        raw_uri = self.build_raw_announcements_s3_uri(symbol, source)

        if (not overwrite) and self.raw_s3.object_exists(raw_key):
            logger.info("[RAW] Exists, skipping: %s", raw_uri)
            return raw_uri

        logger.info("[RAW] Fetching Desiquant corporate announcements for %s", symbol)
        df = self.read_desiquant_announcements_df(symbol=symbol, source=source)

        # Optional: light normalization (column lowercase)
        df.columns = [c.strip().lower() for c in df.columns]

        logger.info("[RAW] Writing to: %s", raw_uri)
        self._write_parquet_to_raw(df, raw_key)
        return raw_uri