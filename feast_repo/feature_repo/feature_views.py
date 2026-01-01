# feature_repo/feature_views.py
from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Float32, Int64, String

from .entities import stock
from .data_sources import OHLCV_FEATURES_SOURCE

ohlcv_features_fv = FeatureView(
    name="ohlcv_tech_indicators",
    entities=[stock],
    ttl=timedelta(days=3650),
    schema=[
        # Base OHLCV
        Field(name="open", dtype=Float32),
        Field(name="high", dtype=Float32),
        Field(name="low", dtype=Float32),
        Field(name="close", dtype=Float32),
        Field(name="volume", dtype=Float32),

        # Engineered features
        Field(name="ret_1d", dtype=Float32),
        Field(name="log_ret_1d", dtype=Float32),
        Field(name="sma_5", dtype=Float32),
        Field(name="sma_10", dtype=Float32),
        Field(name="sma_20", dtype=Float32),
        Field(name="ema_12", dtype=Float32),
        Field(name="ema_26", dtype=Float32),
        Field(name="macd", dtype=Float32),
        Field(name="macd_signal", dtype=Float32),
        Field(name="macd_hist", dtype=Float32),
        Field(name="rsi_14", dtype=Float32),
        Field(name="atr_14", dtype=Float32),
        Field(name="volatility_20", dtype=Float32),
        Field(name="hl_range", dtype=Float32),
        Field(name="oc_change", dtype=Float32),
        Field(name="oc_change_pct", dtype=Float32),
    ],
    source=OHLCV_FEATURES_SOURCE,
    online=False,  # offline-only for training
    tags={"owner": "dissertation"},
)
