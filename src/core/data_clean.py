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


def clean_desiq_corporate_announcements(
    df: pd.DataFrame,
    symbol: str,
    source: str | None = None,   # "bse" | "nse" | None(auto-detect)
) -> pd.DataFrame:
    """
    Clean DesiQ corporate announcements data (BSE or NSE) into a single canonical schema.

    Canonical output columns:
      date, symbol, exchange, published_at, headline, subject, details,
      category, subcategory, announcement_type, url,
      attachment_url, attachment_name, raw_id

    Notes:
    - BSE feed typically has: NEWSID, HEADLINE/NEWSSUB, DT_TM, CATEGORYNAME, SUBCATNAME, NSURL, ATTACHMENTNAME...
    - NSE feed typically has: desc, sort_date/an_dt/dt/exchdisstime, attchmntFile, attchmntText, smIndustry, seq_id...
    """
    if df is None or df.empty:
        logger.info("Received empty corporate announcements DataFrame for %s; skipping clean", symbol)
        return df

    df = df.copy()
    df.rename(columns={c: c.upper() for c in df.columns}, inplace=True)
    cols = set(df.columns)

    # --- Auto-detect source if not provided ---
    if source is None:
        if {"NEWSID", "SCRIP_CD", "DT_TM"}.intersection(cols) and ("CATEGORYNAME" in cols or "SUBCATNAME" in cols):
            source = "bse"
        # elif {"attchmntFile", "attchmntText", "sm_isin", "seq_id"}.intersection(cols) or "smIndustry" in cols:
        elif {"ATTACHMNTFILE", "ATTCHMNTTEXT", "SM_ISIN", "SEQ_ID"}.intersection(cols) or "SMINDUSTRY" in cols:
            source = "nse"
        else:
            raise ValueError(
                f"Unable to auto-detect corporate announcements source from columns: {sorted(cols)}"
            )

    source = source.lower().strip()
    if source not in {"bse", "nse"}:
        raise ValueError("source must be one of: 'bse', 'nse', or None(auto-detect)")

    # --- Build canonical frame depending on source ---
    if source == "bse":
        exchange = "BSE"

        # Published timestamp
        published_at = pd.to_datetime(df.get("DT_TM"), errors="coerce")
        if published_at.isna().all() and "NEWS_DT" in df.columns:
            published_at = pd.to_datetime(df.get("NEWS_DT"), errors="coerce")

        out = pd.DataFrame(
            {
                "published_at": published_at,
                "headline": df.get("HEADLINE"),
                "subject": df.get("NEWSSUB"),
                "details": df.get("MORE"),
                "category": df.get("CATEGORYNAME"),
                "subcategory": df.get("SUBCATNAME"),
                "announcement_type": df.get("ANNOUNCEMENT_TYPE"),
                "url": df.get("NSURL"),
                "attachment_url": df.get("NSURL"),  # BSE often uses same URL for attachment/doc
                "attachment_name": df.get("ATTACHMENTNAME"),
                "raw_id": df.get("NEWSID"),
                "relevance_score": 1.0,  # placeholder
                "relevance_label": "high"
            }
        )

    else:
        exchange = "NSE"

        # Choose best timestamp column available
        ts_col = None
        for c in ["SORT_DATE", "EXCHDISSTIME", "AN_DT", "DT"]:
            if c in df.columns:
                ts_col = c
                break
        published_at = pd.to_datetime(df.get(ts_col), errors="coerce") if ts_col else pd.to_datetime(pd.NA)

        # NSE "desc" is usually the short headline; "attchmntText" is the longer text.
        headline = df.get("DESC")
        details = df.get("ATTCHMNTTEXT")

        out = pd.DataFrame(
            {
                "published_at": published_at,
                "headline": headline,
                "subject": pd.NA,  # not present in NSE feed
                "details": details,
                "category": df.get("SMINDUSTRY"),
                "subcategory": pd.NA,
                "announcement_type": pd.NA,
                "url": df.get("ATTCHMNTFILE"),       # often the document URL
                "attachment_url": df.get("ATTCHMNTFILE"),
                "attachment_name": pd.NA,            # no filename field; keep NA
                "raw_id": df.get("SEQ_ID"),
                "relevance_score": 1.0,  # placeholder
                "relevance_label": "high"
            }
        )

    # --- Add shared columns ---
    out.insert(0, "exchange", exchange)
    out.insert(0, "symbol", symbol)

    # Derive date
    out["published_at"] = pd.to_datetime(out["published_at"], errors="coerce")
    out["date"] = out["published_at"].dt.date

    # Ensure optional text fields exist and are strings/NA-friendly
    for c in ["headline", "subject", "details", "category", "subcategory", "announcement_type", "url", "attachment_url", "attachment_name"]:
        if c not in out.columns:
            out[c] = pd.NA

    # Drop unusable rows (no timestamp AND no headline/details)
    before = len(out)
    out.dropna(subset=["published_at"], inplace=True)
    logger.debug("Dropped %s corporate announcement rows missing published_at", before - len(out))

    # Deduplicate: NSE can repeat same desc+timestamp; BSE can repeat NEWSID etc.
    before = len(out)
    out.drop_duplicates(subset=["symbol", "exchange", "published_at", "headline"], inplace=True)
    logger.debug("Removed %s duplicate corporate announcement rows", before - len(out))

    # Sort + final column order
    out.sort_values(["symbol", "exchange", "published_at"], inplace=True)
    out.reset_index(drop=True, inplace=True)

    out = out[
        [
            "date",
            "symbol",
            "exchange",
            "published_at",
            "headline",
            "subject",
            "details",
            "category",
            "subcategory",
            "announcement_type",
            "url",
            "attachment_url",
            "attachment_name",
            "raw_id",
            "relevance_score",
            "relevance_label"
        ]
    ]

    logger.info(
        "Completed corporate announcements clean (%s) for %s with %s rows (%s -> %s)",
        exchange,
        symbol,
        len(out),
        out["date"].min(),
        out["date"].max(),
    )
    return out


def clean_desiq_financial_results(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Desiquant financial results raw -> canonical financial-results events schema.

    CSV has many columns; we normalize to:
      filing_date, period_end, fiscal_year, fiscal_quarter, res_type, con_noncon, net_sales, net_profit, eps, text_blob
    Output cols:
      date, symbol, filing_date, period_end, fiscal_year, fiscal_quarter,
      res_type, con_noncon, net_sales, net_profit, eps, text_blob, relevance_score
    """
    if df.empty:
        return df

    d = df.copy()

    # Common date columns (observed in these dumps)
    filing_col = "filingdate" if "filingdate" in d.columns else None
    period_col = "periodEndDT" if "periodEndDT" in d.columns else None

    if not filing_col:
        raise ValueError("Financial results missing column: filingdate")

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
    d["relevance_label"] = "high"


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
            "relevance_label"
        ]
    ].copy()

    out.dropna(subset=["filing_date"], inplace=True)
    out.drop_duplicates(subset=["symbol", "filing_date", "period_end", "res_type", "con_noncon"], inplace=True)
    out.sort_values(["symbol", "date", "filing_date"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out
