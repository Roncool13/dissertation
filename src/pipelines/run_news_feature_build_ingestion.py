"""src/pipelines/run_news_feature_build_ingestion.py

Build DAILY sentiment features from *processed* news parquet and write a consolidated
features parquet to S3 for Feast / model training.

Inputs (processed news):
  s3://{bucket}/processed/news/{SYMBOL}/{YEAR}/news_processed.parquet
  Expected canonical columns (from clean_desiq_news_data + relevance_scoring):
    - date (python date or datetime-like)
    - symbol
    - headline
    - summary
    - relevance_score, relevance_label
    - sentiment_score, sentiment_label   (placeholders may be present)

Outputs (news features):
  s3://{bucket}/features/news/news_sentiment_features.parquet
  plus metadata json alongside it.

Key design:
- Optionally run FinBERT scoring (slow) OR skip and use existing sentiment columns.
- Aggregates to daily per (symbol, date) features.
- Adds lag features (t-1..t-k) and simple rolling stats.

CLI examples:
  python src/pipelines/run_news_feature_build_ingestion.py \
    --symbols TCS,INFY \
    --start-year 2021 --end-year 2023 \
    --s3-bucket dissertation-databucket \
    --score-mode finbert \
    --lags 5

  # fast path (if sentiment already present):
  python src/pipelines/run_news_feature_build_ingestion.py \
    --symbols TCS \
    --start-year 2021 --end-year 2023 \
    --score-mode skip
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# Add parent directory to path for local imports (matching your OHLCV feature builder style)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import setup_logging
from src.io.connections import S3Connection
from src.constants import storage as storage_constants

# Reuse your relevance scoring module output columns
from src.nlp.sentiment_finbert import FinBertConfig, score_finbert

logger = logging.getLogger(__name__)


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class NewsFeatureBuildConfig:
    # S3 buckets
    processed_bucket: str
    features_bucket: str

    symbols: List[str]
    start_year: int
    end_year: int

    processed_news_prefix: str = storage_constants.PROCESSED_NEWS_PREFIX
    processed_news_filename: str = storage_constants.PROCESSED_NEWS_FILENAME

    # output location
    output_key_prefix: str = storage_constants.FEATURE_STORE_NEWS_PREFIX
    output_features_filename: str = storage_constants.FEATURE_STORE_NEWS_FILENAME
    output_metadata_filename: str = storage_constants.FEATURE_STORE_NEWS_METADATA_FILENAME

    # scoring
    score_mode: str = "skip"  # finbert|skip
    finbert_batch_size: int = 16
    finbert_max_length: int = 128
    finbert_device: int = -1

    # feature engineering
    lags: int = 5
    target_horizon_days: int = 5
    label_col_name: str = "y_up_5d"


# -----------------------------
# Helpers
# -----------------------------

def _safe_text(x: object) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def _ensure_datetime_date(col: pd.Series) -> pd.Series:
    """Ensure the 'date' column is pandas datetime64[ns] normalized to midnight."""
    # processed pipeline sets `date` as python `dt.date`; handle both.
    d = pd.to_datetime(col, errors="coerce")
    return d.dt.normalize()


def _input_key(prefix: str, symbol: str, year: int, filename: str) -> str:
    return f"{prefix}/{symbol}/{year}/{filename}"


def _output_key(prefix: str, filename: str) -> str:
    return f"{prefix}/{filename}"


# -----------------------------
# Main builder
# -----------------------------
class NewsSentimentFeatureBuildIngestor:
    def __init__(self, cfg: NewsFeatureBuildConfig):
        self.cfg = cfg
        self.s3_processed = S3Connection(bucket=cfg.processed_bucket)
        self.s3_features = S3Connection(bucket=cfg.features_bucket)

    def _download_parquet(self, key: str, dst: Path) -> None:
        self.s3_processed.s3_client.download_file(self.cfg.processed_bucket, key, str(dst))

    def _write_local_and_s3(self, filename: str, s3_key: str, write_func: Callable[[Path], None]) -> None:
        """Write artifact locally (for DVC) and upload the same temp file to S3."""
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td) / filename
            logger.debug("Writing %s to temp path %s", filename, tmp_path)
            write_func(tmp_path)

            local_dir = Path("data/features")
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / filename
            logger.debug("Writing %s to local DVC path %s", filename, local_path)
            write_func(local_path)

            logger.info("Uploading %s to s3://%s/%s", filename, self.cfg.features_bucket, s3_key)
            self.s3_features.upload_file(tmp_path, s3_key)

    def _load_news_year(self, symbol: str, year: int) -> pd.DataFrame:
        key = _input_key(self.cfg.processed_news_prefix, symbol, year, self.cfg.processed_news_filename)
        if not self.s3_processed.object_exists(key):
            logger.warning("Missing processed NEWS: s3://%s/%s (skipping)", self.cfg.processed_bucket, key)
            return pd.DataFrame()

        with tempfile.TemporaryDirectory() as td:
            local = Path(td) / f"{symbol}_{year}_news.parquet"
            logger.info("Downloading %s -> %s", key, local)
            self._download_parquet(key, local)
            df = pd.read_parquet(local)

        if df.empty:
            return df

        # normalize schema
        df = df.copy()
        if "symbol" not in df.columns:
            df.insert(0, "symbol", symbol)
        df["symbol"] = df["symbol"].astype(str)

        if "date" not in df.columns:
            raise ValueError(f"Processed news missing 'date' for {symbol} {year}. Columns={list(df.columns)}")

        df["date"] = _ensure_datetime_date(df["date"])

        # some safety: drop rows without headline
        if "headline" in df.columns:
            df = df.dropna(subset=["headline"]).copy()
        else:
            raise ValueError(f"Processed news missing 'headline' for {symbol} {year}. Columns={list(df.columns)}")

        # make sure relevance exists (processed pipeline should have it)
        if "relevance_score" not in df.columns:
            df["relevance_score"] = 1.0
        if "relevance_label" not in df.columns:
            df["relevance_label"] = "unknown"

        # ensure sentiment cols exist
        if "sentiment_label" not in df.columns:
            df["sentiment_label"] = "neutral"
        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = 0.0

        return df

    def _score_finbert_if_needed(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        mode = (self.cfg.score_mode or "skip").lower().strip()
        if mode == "skip":
            logger.info("score_mode=skip → using existing sentiment_* columns")
            return df

        if mode != "finbert":
            raise ValueError(f"Unknown score_mode: {self.cfg.score_mode}. Use finbert|skip")

        # Build text input (headline is most important; add summary if present)
        headline = df["headline"].map(_safe_text)
        summary = df.get("summary", pd.Series([""] * len(df))).map(_safe_text)
        text = (headline + ". " + summary).str.strip()

        cfg = FinBertConfig(
            batch_size=self.cfg.finbert_batch_size,
            max_length=self.cfg.finbert_max_length,
            device=self.cfg.finbert_device,
        )

        labels, scores = score_finbert(text.tolist(), cfg=cfg)
        out = df.copy()
        out["sentiment_label"] = labels
        out["sentiment_score"] = scores

        return out

    @staticmethod
    def _daily_aggregate(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate article-level sentiments into daily features per symbol/date."""
        if df is None or df.empty:
            return df

        d = df.copy()
        d["date"] = _ensure_datetime_date(d["date"])

        # Map label -> signed polarity
        label = d["sentiment_label"].astype(str).str.lower()
        polarity = np.where(label == "positive", 1.0, np.where(label == "negative", -1.0, 0.0))
        d["polarity"] = polarity.astype(float)

        # Weight by relevance_score (clip to avoid explosions)
        w = pd.to_numeric(d.get("relevance_score", 1.0), errors="coerce").fillna(1.0)
        w = w.clip(lower=0.0, upper=5.0)
        d["w"] = w

        # counts
        d["is_pos"] = (label == "positive").astype(int)
        d["is_neg"] = (label == "negative").astype(int)
        d["is_neu"] = (label == "neutral").astype(int)
        d["is_highrel"] = (d.get("relevance_label", "").astype(str).str.lower() == "high").astype(int)

        # Sentiment score is confidence of predicted label; keep mean of this too
        d["sentiment_score"] = pd.to_numeric(d.get("sentiment_score", 0.0), errors="coerce").fillna(0.0)

        def wmean(x: pd.Series, w: pd.Series) -> float:
            denom = float(w.sum())
            if denom <= 0:
                return float(np.nan)
            return float((x * w).sum() / denom)

        # Weighted polarity mean (make sure apply returns a Series)
        grp_w = d.groupby(["symbol", "date"], sort=False)

        # Daily aggregates
        out = grp_w.agg(
            article_count=("headline", "count"),
            pos_count=("is_pos", "sum"),
            neg_count=("is_neg", "sum"),
            neu_count=("is_neu", "sum"),
            highrel_count=("is_highrel", "sum"),
            rel_score_mean=("relevance_score", "mean"),
            sent_conf_mean=("sentiment_score", "mean"),
            polarity_mean=("polarity", "mean"),
        )

        wp = grp_w.apply(lambda g: wmean(g["polarity"], g["w"]))  # Series
        wp = wp.rename("polarity_wmean").reset_index()           # ok in all pandas

        out = out.merge(wp, on=["symbol", "date"], how="left")

        # Ratios
        out["pos_ratio"] = out["pos_count"] / out["article_count"].replace(0, np.nan)
        out["neg_ratio"] = out["neg_count"] / out["article_count"].replace(0, np.nan)
        out["neu_ratio"] = out["neu_count"] / out["article_count"].replace(0, np.nan)
        out["highrel_ratio"] = out["highrel_count"] / out["article_count"].replace(0, np.nan)

        # Fill NaNs from division-by-zero with 0
        for c in ["pos_ratio", "neg_ratio", "neu_ratio", "highrel_ratio"]:
            out[c] = out[c].fillna(0.0)

        # Sort for lagging
        out.sort_values(["symbol", "date"], inplace=True)
        out.reset_index(drop=True, inplace=True)

        return out

    def _add_lags_and_rolls(self, daily: pd.DataFrame) -> pd.DataFrame:
        if daily is None or daily.empty:
            return daily

        d = daily.copy()
        d.sort_values(["symbol", "date"], inplace=True)

        lag_cols = [
            "polarity_wmean",
            "polarity_mean",
            "pos_ratio",
            "neg_ratio",
            "article_count",
            "highrel_ratio",
            "rel_score_mean",
        ]

        for k in range(1, int(self.cfg.lags) + 1):
            for c in lag_cols:
                d[f"{c}_lag_{k}"] = d.groupby("symbol")[c].shift(k)

        # Simple rolling (using past info only)
        # Example: 3-day and 5-day rolling weighted polarity
        for w in (3, 5):
            d[f"polarity_wmean_roll_{w}"] = d.groupby("symbol")["polarity_wmean"].transform(
                lambda s: s.shift(1).rolling(w, min_periods=w).mean()
            )
            d[f"article_count_roll_{w}"] = d.groupby("symbol")["article_count"].transform(
                lambda s: s.shift(1).rolling(w, min_periods=w).mean()
            )

        return d
    
    def _generate_metadata(self, feats: pd.DataFrame) -> dict:
        from datetime import datetime, timezone, date

        test_year = self.cfg.end_year
        val_year = test_year - 1

        train_start = date(self.cfg.start_year, 1, 1)
        train_end = date(val_year - 1, 12, 31)

        val_start = date(val_year, 1, 1)
        val_end = date(val_year, 12, 31)

        test_start = date(test_year, 1, 1)
        test_end = date(test_year, 12, 31)

        logger.debug(
            "Computed split windows -> train:%s-%s val:%s-%s test:%s-%s",
            train_start,
            train_end,
            val_start,
            val_end,
            test_start,
            test_end,
        )

        splits = {
            "scheme": "global_time_split_v1",
            "train": {
                "start": train_start.isoformat(),
                "end": train_end.isoformat(),
            },
            "val": {
                "start": val_start.isoformat(),
                "end": val_end.isoformat(),
            },
            "test": {
                "start": test_start.isoformat(),
                "end": test_end.isoformat(),
            },
        }

        meta = {
            "dataset": "news_sentiment_features",
            "symbols": self.cfg.symbols,
            "start_year": self.cfg.start_year,
            "end_year": self.cfg.end_year,
            "score_mode": self.cfg.score_mode,
            "finbert": {
                "batch_size": self.cfg.finbert_batch_size,
                "max_length": self.cfg.finbert_max_length,
                "device": self.cfg.finbert_device,
            },
            "lags": int(self.cfg.lags),
            "row_count": int(len(feats)),
            "min_date": str(pd.to_datetime(feats["date"]).min()) if not feats.empty else None,
            "max_date": str(pd.to_datetime(feats["date"]).max()) if not feats.empty else None,
            "feature_columns": [c for c in feats.columns if c not in {"symbol", "date"}],
            "supervision": {
                "target_horizon_days": int(self.cfg.target_horizon_days),
                "label_col_name": self.cfg.label_col_name,
                "join_keys": ["symbol", "date"],
                "asof_definition": "features are computed using news up to (symbol,date) only; lagged/rolling features use past-only shifts"
            },
            "splits": splits,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        logger.debug(
            "Metadata snapshot -> rows:%d columns:%d include_labels:%s",
            meta["row_count"],
            len(meta["feature_columns"]),
            self.cfg.label_col_name,
        )
        return meta

    def build(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for symbol in self.cfg.symbols:
            for year in range(self.cfg.start_year, self.cfg.end_year + 1):
                df_y = self._load_news_year(symbol, year)
                if not df_y.empty:
                    frames.append(df_y)

        if not frames:
            raise RuntimeError("No processed NEWS found for the requested symbols/years.")

        news = pd.concat(frames, ignore_index=True)
        news.sort_values(["symbol", "date"], inplace=True)

        # score sentiment if needed
        news = self._score_finbert_if_needed(news)

        daily = self._daily_aggregate(news)
        daily = self._add_lags_and_rolls(daily)

        # Feast expects event timestamp column
        daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None)

        return daily

    def write_outputs(self, feats: pd.DataFrame) -> Dict[str, str]:
        feat_key = _output_key(self.cfg.output_key_prefix, self.cfg.output_features_filename)
        meta_key = _output_key(self.cfg.output_key_prefix, self.cfg.output_metadata_filename)

        meta = self._generate_metadata(feats)
        logger.info("Writing NEWS features to s3://%s/%s", self.cfg.features_bucket, feat_key)
        self._write_local_and_s3(
            filename=self.cfg.output_features_filename,
            s3_key=feat_key,
            write_func=lambda path: feats.to_parquet(path, index=False),
        )

        payload = json.dumps(meta, indent=2)

        logger.info("Writing NEWS metadata to s3://%s/%s", self.cfg.features_bucket, meta_key)
        self._write_local_and_s3(
            filename=self.cfg.output_metadata_filename,
            s3_key=meta_key,
            write_func=lambda path: path.write_text(payload, encoding="utf-8"),
        )

        return {"features": feat_key, "metadata": meta_key}

    def run(self) -> None:
        logger.info("Starting NEWS sentiment feature build ingestion...")
        feats = self.build()
        out = self.write_outputs(feats)
        logger.info("NEWS sentiment feature build complete. Features=%s Metadata=%s", out["features"], out["metadata"])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build daily sentiment features from processed news.")
    p.add_argument("--symbols", required=True, help="Comma-separated list, e.g., TCS,INFY")
    p.add_argument("--start-year", type=int, required=True)
    p.add_argument("--end-year", type=int, required=True)
    p.add_argument("--s3-processed-bucket", required=True,
                help="Your S3 processed bucket (e.g., dissertation-databucket-processed)")
    p.add_argument("--s3-features-bucket", required=True,
                help="Your S3 features bucket (e.g., dissertation-databucket-dvcstore)")
    p.add_argument("--score-mode", choices=["finbert", "skip"], default="skip")
    p.add_argument("--finbert-batch-size", type=int, default=16)
    p.add_argument("--finbert-max-length", type=int, default=128)
    p.add_argument("--finbert-device", type=int, default=-1)
    p.add_argument("--horizon-days", type=int, default=5)

    p.add_argument("--lags", type=int, default=5)

    p.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Logging level (e.g. DEBUG, INFO). Defaults to LOG_LEVEL env or INFO.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging(args.log_level)

    cfg = NewsFeatureBuildConfig(
        processed_bucket=args.s3_processed_bucket,
        features_bucket=args.s3_features_bucket,
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        start_year=args.start_year,
        end_year=args.end_year,
        score_mode=args.score_mode,
        finbert_batch_size=args.finbert_batch_size,
        finbert_max_length=args.finbert_max_length,
        finbert_device=args.finbert_device,
        lags=args.lags,
        target_horizon_days=args.horizon_days,
        label_col_name=f"y_up_{args.horizon_days}d",
    )

    NewsSentimentFeatureBuildIngestor(cfg).run()


if __name__ == "__main__":
    main()
