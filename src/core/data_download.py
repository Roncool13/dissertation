# src/data/ohlcv_download.py

# Standard library imports
import datetime as dt
from typing import List

# Third-party imports
import pandas as pd
from jugaad_data.nse import stock_df
import jugaad_data.util as jugaad_util


def _patch_jugaad_cache_makedirs() -> None:
    """
    jugaad-data creates its cache directory in multiple threads without
    exist_ok=True, which can raise FileExistsError. Patch makedirs to
    swallow that specific race.
    """
    if getattr(jugaad_util, "_makedirs_patched", False):
        return

    original_makedirs = jugaad_util.os.makedirs

    def _safe_makedirs(path, *args, **kwargs):
        try:
            original_makedirs(path, *args, **kwargs)
        except FileExistsError:
            return

    jugaad_util.os.makedirs = _safe_makedirs
    jugaad_util._makedirs_patched = True


def download_ohlcv_nsepy(
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    """
    Download daily OHLCV data from NSE using jugaad-data (stock_df)
    and merge into a single DataFrame.

    NOTE:
    - Name kept as `download_ohlcv_nsepy` to match the earlier notebook.
    """
    all_data = []

    # you already had similar logic in the notebook
    _patch_jugaad_cache_makedirs()
    df = stock_df(symbol=symbol, from_date=start_date, to_date=end_date)

    if df.empty:
        raise ValueError("No data downloaded for the symbol.")

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
    return df


def fetch_news_from_provider(
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    """
    TODO: Implement this using your chosen news API/provider.

    It should return a DataFrame with at least:
      - published_at (timestamp)
      - title or headline
      - source

    For now it's a stub.
    """
    # Example structure (dummy):
    # rows = [
    #     {"published_at": "2025-01-01T10:00:00Z", "headline": "News...", "source": "X"},
    # ]
    # return pd.DataFrame(rows)

    raise NotImplementedError("Implement fetch_news_from_provider for your news source.")
