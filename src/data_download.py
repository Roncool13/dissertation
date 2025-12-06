# src/data/ohlcv_download.py
import datetime as dt
from typing import List

import pandas as pd
from jugaad_data.nse import stock_df


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