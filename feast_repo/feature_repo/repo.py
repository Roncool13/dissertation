# feature_repo/repo.py
"""
Feast requires a top-level module that exposes entities/feature views.
This is convenient for `feast apply` discovery.
"""
from .entities import stock
from .feature_views import ohlcv_features_fv

__all__ = ["stock", "ohlcv_features_fv"]
