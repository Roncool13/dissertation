# src/core/data_download.py

# Standard library imports
import time
import datetime as dt
import logging
import json
from typing import List
from urllib.parse import quote_plus

# Third-party imports
import feedparser
import pandas as pd
from jugaad_data.nse import stock_df
import jugaad_data.util as jugaad_util
from requests.exceptions import JSONDecodeError as RequestsJSONDecodeError

# Local imports
from src.constants import SYMBOL_TO_COMPANY  # adjust import path as per your repo

logger = logging.getLogger(__name__)


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


def download_ohlcv_nsepy(
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    """
    Download daily OHLCV data from NSE using jugaad-data (stock_df)
    and return a DataFrame with columns:
        date, open, high, low, close, volume, symbol
    """
    logger.info("Downloading OHLCV via jugaad-data for %s from %s to %s", symbol, start_date, end_date)
    _patch_jugaad_cache_makedirs()
    try:
        df = stock_df(symbol=symbol, from_date=start_date, to_date=end_date)
    except (json.JSONDecodeError, RequestsJSONDecodeError) as exc:
        logger.error(
            "NSE endpoint returned non-JSON for %s %s->%s; possible site change or temporary block. Error: %s",
            symbol,
            start_date,
            end_date,
            exc,
        )
        raise RuntimeError(
            f"NSE response could not be parsed as JSON for {symbol} ({start_date} -> {end_date}); "
            "retry later or check connectivity/cookies."
        ) from exc

    if df.empty:
        logger.error("No OHLCV data returned for %s in window %s -> %s", symbol, start_date, end_date)
        raise ValueError(f"No OHLCV data downloaded for symbol {symbol!r}.")

    logger.debug("Raw OHLCV data shape for %s: %s", symbol, df.shape)
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
    logger.info("Completed OHLCV download for %s with %s rows", symbol, len(df))
    return df


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
        f'{company} NSE',          # sometimes helps for India context
        f'{symbol} stock',         # fallback
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
