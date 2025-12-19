# src/core/data_download.py

# Standard library imports
import time
import datetime as dt
import logging
import json
import random
from typing import List
from urllib.parse import quote_plus

# Third-party imports
import feedparser
import pandas as pd
import yfinance as yf
from jugaad_data.nse import stock_df
import jugaad_data.util as jugaad_util
from requests.exceptions import JSONDecodeError as RequestsJSONDecodeError

# Local imports
from src.constants import SYMBOL_TO_COMPANY

logger = logging.getLogger(__name__)


# -------------------------
# OHLCV: NSE (jugaad) + fallback
# -------------------------
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


def _download_ohlcv_jugaad(symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """
    Download daily OHLCV data from NSE using jugaad-data (stock_df)
    and return a DataFrame with columns:
        date, open, high, low, close, volume, symbol
    """
    logger.info("Downloading OHLCV via jugaad-data for %s from %s to %s", symbol, start_date, end_date)
    _patch_jugaad_cache_makedirs()
    df = stock_df(symbol=symbol, from_date=start_date, to_date=end_date)

    if df.empty:
        logger.error("[jugaad] No OHLCV data returned for %s in window %s -> %s", symbol, start_date, end_date)
        raise ValueError(f"No OHLCV data downloaded for symbol {symbol!r}.")

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


def _download_ohlcv_yfinance(symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """
    Fallback provider: yfinance (Yahoo).
    Uses NSE tickers: SYMBOL.NS
    """
    ticker = f"{symbol}.NS"
    logger.warning("[yfinance] Fallback OHLCV for %s using ticker %s", symbol, ticker)

    # yfinance end can behave like exclusive; add +1 day and filter back
    end_plus = end_date + dt.timedelta(days=1)

    data = yf.download(
        tickers=ticker,
        start=start_date.isoformat(),
        end=end_plus.isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data is None or len(data) == 0:
        raise ValueError(f"[yfinance] No OHLCV data returned for {symbol!r} ({ticker}).")

    # Handle MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]

    data = data.reset_index()

    # Standardize column names
    data.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )

    needed = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        logger.debug("[yfinance] Missing expected columns %s; columns=%s", missing, list(data.columns))
        return pd.DataFrame()

    df = data[needed].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]
    df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]
    df["symbol"] = symbol
    logger.info("[yfinance] Completed OHLCV download for %s with %s rows", symbol, len(df))

    return df[["date", "open", "high", "low", "close", "volume", "symbol"]]


def _download_ohlcv_yfinance(symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """
    Fallback provider: Yahoo Finance via yfinance.

    GitHub runners often get 429 rate-limited by Yahoo.
    We implement retries + prefer Ticker().history() which is often more reliable
    than yf.download() under throttling.
    """
    ticker = f"{symbol}.NS"
    logger.warning("[yfinance] Fallback OHLCV for %s using ticker %s", symbol, ticker)

    # yfinance end can behave like exclusive; add +1 day and filter back
    end_plus = end_date + dt.timedelta(days=1)

    max_attempts = 6
    base_sleep = 2.0

    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            # 1) Try history() first
            t = yf.Ticker(ticker)
            hist = t.history(
                start=start_date.isoformat(),
                end=end_plus.isoformat(),
                interval="1d",
                auto_adjust=False,
            )
            df = _normalize_yf_frame(hist, symbol, start_date, end_date)
            if not df.empty:
                logger.info("[yfinance] history() succeeded for %s with %d rows", ticker, len(df))
                return df

            # 2) Fallback to yf.download()
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
                logger.info("[yfinance] download() succeeded for %s with %d rows", ticker, len(df))
                return df

            # If both returned empty, treat as retryable (could be throttled / transient)
            last_err = ValueError("Empty response from yfinance (possible 429 throttling)")
            raise last_err

        except Exception as e:
            last_err = e
            # Exponential backoff + jitter (helps with 429)
            sleep_s = min(60.0, base_sleep * (2 ** (attempt - 1))) + random.uniform(0, 1.5)
            logger.warning(
                "[yfinance] Attempt %d/%d failed for %s: %s. Sleeping %.1fs",
                attempt,
                max_attempts,
                ticker,
                f"{type(e).__name__}: {e}",
                sleep_s,
            )
            time.sleep(sleep_s)

    raise ValueError(f"[yfinance] No OHLCV data returned for {symbol!r} ({ticker}). Last error: {last_err}")


def download_ohlcv_nsepy(symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """
    Primary: NSE via jugaad-data.
    Fallback: Yahoo Finance via yfinance (with retries for 429).
    """
    symbol = symbol.strip().upper()

    try:
        df = _download_ohlcv_jugaad(symbol, start_date, end_date)
        logger.info("[jugaad] Completed OHLCV download for %s with %s rows", symbol, len(df))
        return df

    except (json.JSONDecodeError, RequestsJSONDecodeError) as exc:
        logger.warning(
            "[jugaad] NSE returned non-JSON for %s %s->%s; falling back to yfinance. Error: %s",
            symbol,
            start_date,
            end_date,
            exc,
        )
        df = _download_ohlcv_yfinance(symbol, start_date, end_date)
        logger.info("[yfinance] Completed OHLCV download for %s with %s rows", symbol, len(df))
        return df

    except Exception as exc:
        logger.warning(
            "[jugaad] Failed for %s %s->%s (%s). Falling back to yfinance.",
            symbol,
            start_date,
            end_date,
            f"{type(exc).__name__}: {exc}",
        )
        df = _download_ohlcv_yfinance(symbol, start_date, end_date)
        logger.info("[yfinance] Completed OHLCV download for %s with %s rows", symbol, len(df))
        return df


# -------------------------
# News: Google News RSS
# -------------------------
def _build_google_news_queries(symbol: str) -> List[str]:
    """
    Build multiple queries to improve relevance and coverage.
    Google News RSS is query-sensitive, so we try a few.
    """
    company = SYMBOL_TO_COMPANY.get(symbol, symbol)

    # Finance-context anchors reduce irrelevant hits
    queries = [
        f'"{company}" stock',
        f'"{company}" shares',
        f'"{company}" results',
        f"{company} NSE",
        f"{symbol} stock",
    ]

    # Remove duplicates while preserving order
    seen = set()
    out = []
    for q in queries:
        if q not in seen:
            out.append(q)
            seen.add(q)

    logger.debug("Built %s Google News queries for %s: %s", len(out), symbol, out)
    return out


def _google_news_rss_url(query: str) -> str:
    # hl=en-IN, gl=IN, ceid=IN:en to bias towards India + English
    # Sorting is handled internally by Google News.
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"


def fetch_news_from_provider(
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
    max_items_per_query: int = 50,
    sleep_s: float = 0.6,
) -> pd.DataFrame:
    """
    Fetch news using Google News RSS.

    Returns a DataFrame with:
      - headline
      - source
      - published_at
      - link
      - summary
      - query_used

    Notes:
    - RSS provides limited history; for long backfills you may need repeated runs / pagination isn't supported.
    - This function is best for daily/weekly ingestion and near-history.
    """
    symbol = symbol.strip().upper()

    queries = _build_google_news_queries(symbol)

    rows = []
    for q in queries:
        url = _google_news_rss_url(q)
        logger.info("Fetching RSS for %s using query %r", symbol, q)
        feed = feedparser.parse(url)
        entries = getattr(feed, "entries", []) or []
        logger.debug("Received %s entries for query %r", len(entries), q)

        for e in entries[:max_items_per_query]:
            title = getattr(e, "title", None)
            link = getattr(e, "link", None)

            # feedparser: published or updated may exist depending on feed
            published = getattr(e, "published", None) or getattr(e, "updated", None)

            # source handling: sometimes e.source.title exists
            source = None
            if hasattr(e, "source") and isinstance(e.source, dict):
                source = e.source.get("title")
            elif hasattr(e, "source") and hasattr(e.source, "title"):
                source = e.source.title

            summary = getattr(e, "summary", None)

            rows.append(
                {
                    "headline": title,
                    "source": source,
                    "published_at": published,
                    "link": link,
                    "summary": summary,
                    "query_used": q,
                }
            )

        # polite pacing
        logger.debug("Sleeping %.2fs between RSS requests", sleep_s)
        time.sleep(sleep_s)

    if not rows:
        logger.warning("No news rows collected for %s between %s and %s", symbol, start_date, end_date)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.debug("Constructed news DataFrame with %s rows before cleanup", len(df))

    # Basic cleanup / normalization
    before_headline = len(df)
    df.dropna(subset=["headline"], inplace=True)
    logger.debug("Dropped %s rows missing headline", before_headline - len(df))

    # Parse datetime (RSS dates are usually RFC822)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)

    # Filter by requested date range (best-effort)
    df["date"] = df["published_at"].dt.date
    before_range = len(df)
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    logger.debug("Filtered %s rows outside requested date range", before_range - len(df))

    # Deduplicate (prefer link, fallback to headline)
    if "link" in df.columns and df["link"].notna().any():
        before_dupes = len(df)
        df.drop_duplicates(subset=["link"], inplace=True)
        logger.debug("Removed %s duplicate rows by link", before_dupes - len(df))
    else:
        before_dupes = len(df)
        df.drop_duplicates(subset=["headline"], inplace=True)
        logger.debug("Removed %s duplicate rows by headline", before_dupes - len(df))

    # Drop helper
    df.drop(columns=["date"], inplace=True, errors="ignore")

    # Sort stable
    df.sort_values(["published_at", "source"], inplace=True, na_position="last")
    df.reset_index(drop=True, inplace=True)
    logger.info("Completed news fetch for %s with %s rows", symbol, len(df))
    return df