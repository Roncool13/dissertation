import numpy as np
import pandas as pd

def build_pattern_features(ohlcv: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Input columns required: ['symbol','date','open','high','low','close'] (+ optional volume)
    Output: symbol,date + anatomy + pattern flags + rolling counts
    """
    df = ohlcv.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Candle anatomy
    o = df["open"].astype("float32")
    h = df["high"].astype("float32")
    l = df["low"].astype("float32")
    c = df["close"].astype("float32")

    real_body = (c - o).abs()
    candle_range = (h - l).replace(0, np.nan)

    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    df["body"] = real_body
    df["range"] = (h - l)
    df["upper_wick"] = upper_wick
    df["lower_wick"] = lower_wick

    df["body_to_range"] = (real_body / candle_range).astype("float32")
    df["upper_wick_to_range"] = (upper_wick / candle_range).astype("float32")
    df["lower_wick_to_range"] = (lower_wick / candle_range).astype("float32")

    # Direction helpers
    bullish = (c > o)
    bearish = (c < o)

    # Patterns (simple robust thresholds)
    # Doji: tiny body vs range
    df["pattern_doji"] = (df["body_to_range"] <= 0.10).astype("int8")

    # Hammer: long lower wick, small body, small upper wick
    df["pattern_hammer"] = (
        (df["lower_wick"] >= 2.0 * df["body"]) &
        (df["upper_wick"] <= 0.3 * df["body"]) &
        (df["body_to_range"] <= 0.35)
    ).astype("int8")

    # Shooting star: long upper wick
    df["pattern_shooting_star"] = (
        (df["upper_wick"] >= 2.0 * df["body"]) &
        (df["lower_wick"] <= 0.3 * df["body"]) &
        (df["body_to_range"] <= 0.35)
    ).astype("int8")

    # Engulfing uses prev candle
    prev_o = df.groupby("symbol")["open"].shift(1).astype("float32")
    prev_c = df.groupby("symbol")["close"].shift(1).astype("float32")

    prev_bull = (prev_c > prev_o)
    prev_bear = (prev_c < prev_o)

    # Bullish engulfing: prev bearish, current bullish, current body engulfs prev body
    df["pattern_bullish_engulfing"] = (
        prev_bear &
        bullish &
        (np.minimum(o, c) <= np.minimum(prev_o, prev_c)) &
        (np.maximum(o, c) >= np.maximum(prev_o, prev_c))
    ).astype("int8")

    # Bearish engulfing
    df["pattern_bearish_engulfing"] = (
        prev_bull &
        bearish &
        (np.minimum(o, c) <= np.minimum(prev_o, prev_c)) &
        (np.maximum(o, c) >= np.maximum(prev_o, prev_c))
    ).astype("int8")

    # Rolling counts (per symbol)
    pat_cols = [c for c in df.columns if c.startswith("pattern_")]
    for col in pat_cols:
        df[f"{col}_cnt_{lookback}"] = (
            df.groupby("symbol")[col]
              .rolling(lookback, min_periods=1)
              .sum()
              .reset_index(level=0, drop=True)
              .astype("float32")
        )

    keep = ["symbol", "date"] + \
           ["body","range","upper_wick","lower_wick","body_to_range","upper_wick_to_range","lower_wick_to_range"] + \
           pat_cols + [f"{c}_cnt_{lookback}" for c in pat_cols]

    return df[keep].reset_index(drop=True)