# src/core/data_clean.py

# Standard library imports
import json
import logging
import datetime as dt

# Third-party imports
import pandas as pd

COLUMNS = ["date", "open", "high", "low", "close", "volume"]

logger = logging.getLogger(__name__)


def clean_ohlcv_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Clean OHLCV data and return canonical minimal schema:
    date, symbol, open, high, low, close, volume
    """

    logger.debug("Starting OHLCV clean with %s rows and columns: %s", len(df), list(df.columns))
    df = df.copy()

    # Normalize volume naming if needed
    if "VOLUME" in df.columns:
        logger.info("Renaming VOLUME column to volume for OHLCV data")
        df.rename(columns={"VOLUME": "volume"}, inplace=True)

    # Keep strict canonical ordering
    logger.debug("Reordering OHLCV columns to canonical schema: %s", COLUMNS)
    df = df[COLUMNS]

    # Correct dtypes
    logger.debug("Converting OHLCV date column to datetime")
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        logger.debug("Converting column %s to numeric", col)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove bad rows
    before_dropna = len(df)
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    logger.debug("Dropped %s rows with NaNs in price columns", before_dropna - len(df))
    before_dupes = len(df)
    df.drop_duplicates(subset=["date"], inplace=True)
    logger.debug("Removed %s duplicate OHLCV rows", before_dupes - len(df))

    # filter out obviously bad rows
    before_filter = len(df)
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    df = df[df["high"] >= df["low"]]
    logger.debug("Filtered %s rows with invalid price data", before_filter - len(df))

    df.sort_values(["date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Insert symbol column
    df.insert(0, 'symbol', symbol)

    logger.info("Completed OHLCV clean with %s rows for %s symbols", len(df), df["symbol"].nunique())

    return df


def clean_news_data(
    df: pd.DataFrame,
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    """
    Clean raw news DataFrame and transform into canonical schema.
    Keeps summary/link/query_used if present (useful for relevance + debugging).
    """
    if df.empty:
        logger.info("Received empty news DataFrame for %s; skipping clean", symbol)
        return df

    logger.debug(
        "Starting news clean for %s with %s rows, columns: %s, date window: %s -> %s",
        symbol,
        len(df),
        list(df.columns),
        start_date,
        end_date,
    )
    df = df.copy()

    # Standardize column names depending on provider
    rename_map = {}
    if "title" in df.columns and "headline" not in df.columns:
        rename_map["title"] = "headline"
    if "time_published" in df.columns and "published_at" not in df.columns:
        rename_map["time_published"] = "published_at"
    df.rename(columns=rename_map, inplace=True)
    if rename_map:
        logger.debug("Applied news column renames: %s", rename_map)

    required_cols = ["headline", "source", "published_at"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error("News DataFrame missing required columns: %s", missing)
        raise ValueError(f"News DataFrame missing columns: {missing}")

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["date"] = df["published_at"].dt.date
    before_date_filter = len(df)
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    logger.debug("Filtered %s news rows outside date window", before_date_filter - len(df))
    df["symbol"] = symbol

    # Keep optional columns if present
    for optional in ["summary", "link", "query_used"]:
        if optional not in df.columns:
            df[optional] = pd.NA

    # Placeholder sentiment (FinBERT pipeline will fill later)
    df["sentiment_score"] = 0.0
    df["sentiment_label"] = "neutral"

    df = df[
        [
            "date",
            "symbol",
            "headline",
            "summary",
            "source",
            "published_at",
            "link",
            "query_used",
            "sentiment_score",
            "sentiment_label",
        ]
    ]

    before_dropna = len(df)
    df.dropna(subset=["headline", "published_at"], inplace=True)
    logger.debug("Dropped %s news rows missing headline/published_at", before_dropna - len(df))
    before_dupes = len(df)
    df.drop_duplicates(subset=["symbol", "headline", "published_at"], inplace=True)
    logger.debug("Removed %s duplicate news rows", before_dupes - len(df))
    df.sort_values(["symbol", "date", "published_at"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(
        "Completed news clean for %s with %s rows between %s and %s",
        symbol,
        len(df),
        df["date"].min(),
        df["date"].max(),
    )
    return df


# --- Desiquant raw -> canonical cleaners (news / corp announcements / financial results) ---

def _safe_json(x):
    if pd.isna(x):
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}


def clean_desiq_news_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Desiquant news raw schema -> canonical news schema.

    Expected raw cols (from your tcs_news.csv):
      title, description, link, date, publisher, feed_type, ...
    Output cols:
      date, symbol, headline, summary, source, published_at, link,
      relevance_score, sentiment_score, sentiment_label
    """
    if df.empty:
        return df

    d = df.copy()

    # Normalize columns
    if "title" not in d.columns:
        raise ValueError("Desiquant news missing column: title")
    if "date" not in d.columns:
        raise ValueError("Desiquant news missing column: date")

    d["published_at"] = pd.to_datetime(d["date"], errors="coerce", utc=True)
    d["date"] = d["published_at"].dt.date

    d["headline"] = d["title"].astype(str).str.strip()
    d["summary"] = d["description"] if "description" in d.columns else pd.NA
    d["link"] = d["link"] if "link" in d.columns else pd.NA

    if "publisher" in d.columns:
        pub = d["publisher"].apply(_safe_json)
        d["source"] = pub.apply(lambda x: x.get("name") or x.get("title") or "unknown")
    else:
        d["source"] = "unknown"

    d["symbol"] = symbol

    # Placeholders (you’ll fill later with FinBERT + relevance model)
    d["relevance_score"] = 1.0
    d["sentiment_score"] = 0.0
    d["sentiment_label"] = "neutral"

    out = d[
        [
            "date",
            "symbol",
            "headline",
            "summary",
            "source",
            "published_at",
            "link",
            "relevance_score",
            "sentiment_score",
            "sentiment_label",
        ]
    ].copy()

    out.dropna(subset=["headline", "published_at"], inplace=True)
    out.drop_duplicates(subset=["symbol", "headline", "published_at"], inplace=True)
    out.sort_values(["symbol", "date", "published_at"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def clean_desiq_corporate_announcements(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Desiquant corporate announcements raw -> canonical announcements schema.

    Your CSV looks like it has:
      NEWS_DT, HEADLINE, NEWSSUB, CATEGORYNAME, SUBCATNAME, ATTACHMENTNAME, ...
    Output cols:
      date, symbol, headline, details, category, subcategory, attachment_name, attachment_link, relevance_score
    """
    if df.empty:
        return df

    d = df.copy()

    # Date
    if "NEWS_DT" not in d.columns:
        raise ValueError("Corporate announcements missing column: NEWS_DT")
    d["published_at"] = pd.to_datetime(d["NEWS_DT"], errors="coerce", utc=True)
    d["date"] = d["published_at"].dt.date

    # Text fields
    d["headline"] = d["HEADLINE"].astype(str).str.strip() if "HEADLINE" in d.columns else pd.NA
    d["details"] = d["NEWSSUB"] if "NEWSSUB" in d.columns else pd.NA

    d["category"] = d["CATEGORYNAME"] if "CATEGORYNAME" in d.columns else pd.NA
    d["subcategory"] = d["SUBCATNAME"] if "SUBCATNAME" in d.columns else pd.NA

    d["attachment_name"] = d["ATTACHMENTNAME"] if "ATTACHMENTNAME" in d.columns else pd.NA
    d["attachment_link"] = d["ATTACHMENTLINK"] if "ATTACHMENTLINK" in d.columns else pd.NA

    d["symbol"] = symbol
    d["relevance_score"] = 1.0  # placeholder

    out = d[
        [
            "date",
            "symbol",
            "headline",
            "details",
            "category",
            "subcategory",
            "published_at",
            "attachment_name",
            "attachment_link",
            "relevance_score",
        ]
    ].copy()

    out.dropna(subset=["headline", "published_at"], inplace=True)
    out.drop_duplicates(subset=["symbol", "headline", "published_at"], inplace=True)
    out.sort_values(["symbol", "date", "published_at"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def clean_desiq_financial_results(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Desiquant financial results raw -> canonical financial-results events schema.

    Your CSV has many columns; we normalize to:
      filing_date, period_end, fiscal_year, fiscal_quarter, res_type, con_noncon, net_sales, net_profit, eps, text_blob
    Output cols:
      date, symbol, filing_date, period_end, fiscal_year, fiscal_quarter,
      res_type, con_noncon, net_sales, net_profit, eps, text_blob, relevance_score
    """
    if df.empty:
        return df

    d = df.copy()

    # Common date columns (observed in these dumps)
    filing_col = "filingDate" if "filingDate" in d.columns else None
    period_col = "periodEndDT" if "periodEndDT" in d.columns else None

    if not filing_col:
        raise ValueError("Financial results missing column: filingDate")

    d["filing_date"] = pd.to_datetime(d[filing_col], errors="coerce", utc=True)
    d["date"] = d["filing_date"].dt.date

    d["period_end"] = pd.to_datetime(d[period_col], errors="coerce", utc=True) if period_col else pd.NaT

    # Useful descriptors (if present)
    d["res_type"] = d["resType"] if "resType" in d.columns else pd.NA
    d["con_noncon"] = d["conNonCon"] if "conNonCon" in d.columns else pd.NA
    d["fiscal_year"] = d["fiscalYear"] if "fiscalYear" in d.columns else pd.NA
    d["fiscal_quarter"] = d["quarter"] if "quarter" in d.columns else pd.NA

    # Numeric highlights (if present)
    def pick_num(cols):
        for c in cols:
            if c in d.columns:
                return pd.to_numeric(d[c], errors="coerce")
        return pd.Series([pd.NA] * len(d))

    d["net_sales"] = pick_num(["resultsData2.re_net_sale", "resultsData.re_net_sale"])
    d["net_profit"] = pick_num(["resultsData2.re_net_profit", "resultsData.re_net_profit"])
    d["eps"] = pick_num(["resultsData2.re_eps", "resultsData.re_eps"])

    # Text blob for later relevance/sentiment enrichment
    d["text_blob"] = (
        "Financial results filed"
        + " | res_type=" + d["res_type"].astype(str)
        + " | con_noncon=" + d["con_noncon"].astype(str)
        + " | fy=" + d["fiscal_year"].astype(str)
        + " | q=" + d["fiscal_quarter"].astype(str)
    )

    d["symbol"] = symbol
    d["relevance_score"] = 1.0  # placeholder

    out = d[
        [
            "date",
            "symbol",
            "filing_date",
            "period_end",
            "fiscal_year",
            "fiscal_quarter",
            "res_type",
            "con_noncon",
            "net_sales",
            "net_profit",
            "eps",
            "text_blob",
            "relevance_score",
        ]
    ].copy()

    out.dropna(subset=["filing_date"], inplace=True)
    out.drop_duplicates(subset=["symbol", "filing_date", "period_end", "res_type", "con_noncon"], inplace=True)
    out.sort_values(["symbol", "date", "filing_date"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out
