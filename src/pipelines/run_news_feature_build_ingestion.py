"""src/pipelines/run_news_feature_build_ingestion_REWRITTEN.py

Daily news sentiment feature builder rewritten to follow the same design
structure as run_news_feature_build_ingestion.py while preserving the extended
feature engineering, FinBERT scoring, and optional OHLCV label attachment logic
from the rewritten prototype.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

# Mirror the original pipeline import pattern
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import setup_logging
from src.io.connections import S3Connection
from src.constants import storage as storage_constants
from src.nlp.sentiment_finbert import score_finbert, FinBertConfig

logger = logging.getLogger(__name__)


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class NewsFeatureBuildConfig:
    processed_bucket: str
    features_bucket: str
    symbols: List[str]
    start_year: int
    end_year: int

    processed_news_prefix: str = storage_constants.PROCESSED_NEWS_PREFIX
    processed_news_filename: str = storage_constants.PROCESSED_NEWS_FILENAME

    local_features_dir: str = "data/features"
    output_features_filename: str = storage_constants.FEATURE_STORE_NEWS_FILENAME
    output_metadata_filename: str = storage_constants.FEATURE_STORE_NEWS_METADATA_FILENAME
    output_key_prefix: str = f"{storage_constants.PROCESSED_S3_PREFIX}/features/news"
    upload_to_s3: bool = True

    score_mode: str = "finbert"  # finbert|skip
    finbert_batch_size: int = 16
    finbert_max_length: int = 128
    finbert_device: int = -1

    lags: int = 5
    roll_windows: Tuple[int, ...] = (3, 5, 10, 20)
    shock_window: int = 20
    shock_z: float = 1.5

    attach_label: bool = True
    ohlcv_filename: str = storage_constants.FEATURE_STORE_OHLCV_FILENAME
    ohlcv_metadata_filename: str = storage_constants.FEATURE_STORE_OHLCV_METADATA_FILENAME
    ohlcv_features_path: str = f"{storage_constants.FEATURE_STORE_OHLCV_PREFIX}/{storage_constants.FEATURE_STORE_OHLCV_FILENAME}"
    ohlcv_metadata_path: str = f"{storage_constants.FEATURE_STORE_OHLCV_PREFIX}/{storage_constants.FEATURE_STORE_OHLCV_METADATA_FILENAME}"
    label_col_fallback: str = "y_up_5d"


# -----------------------------
# Helpers
# -----------------------------
def _ensure_datetime_date(col: pd.Series) -> pd.Series:
    d = pd.to_datetime(col, errors="coerce")
    return d.dt.normalize()


def _safe_text(val: object) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val).strip()


def _weighted_mean(x: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    w = pd.to_numeric(w, errors="coerce")
    mask = x.notna() & w.notna()
    if not mask.any():
        return float("nan")
    x = x[mask].astype(float)
    w = w[mask].astype(float)
    denom = w.sum()
    if denom == 0:
        return float("nan")
    return float((x * w).sum() / denom)


def _infer_label_col(meta: Dict[str, object], fallback: str) -> str:
    horizon = meta.get("horizon_days")
    if horizon is not None:
        return f"y_up_{int(horizon)}d"
    return meta.get("label_col", fallback)


# -----------------------------
# Pipeline implementation
# -----------------------------
class NewsSentimentFeatureBuildPipeline:
    def __init__(self, cfg: NewsFeatureBuildConfig):
        self.cfg = cfg
        self.s3_processed = S3Connection(bucket=cfg.processed_bucket)
        self.s3_features = S3Connection(bucket=cfg.features_bucket)
        self.local_dir = Path(cfg.local_features_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

    # ---------- S3 helpers ----------
    def _news_key(self, symbol: str, year: int) -> str:
        return f"{self.cfg.processed_news_prefix}/{symbol}/{year}/{self.cfg.processed_news_filename}"

    # ---------- Loading ----------
    def _load_news_year(self, symbol: str, year: int) -> pd.DataFrame:
        key = self._news_key(symbol, year)
        logger.debug("Fetching processed news for symbol=%s year=%s key=%s", symbol, year, key)
        if not self.s3_processed.object_exists(key):
            logger.warning(
                "Missing processed news for %s %s at s3://%s/%s",
                symbol,
                year,
                self.cfg.processed_bucket,
                key,
            )
            return pd.DataFrame()

        with tempfile.TemporaryDirectory() as td:
            local_path = Path(td) / f"news_{symbol}_{year}.parquet"
            self.s3_processed.s3_client.download_file(self.cfg.processed_bucket, key, str(local_path))
            df = pd.read_parquet(local_path)
            logger.debug("Downloaded news parquet for %s %s (%d rows)", symbol, year, len(df))

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        if "symbol" not in df.columns:
            df.insert(0, "symbol", symbol)
        else:
            df["symbol"] = df["symbol"].fillna(symbol)

        if "date" not in df.columns:
            if "published_at" in df.columns:
                df["date"] = _ensure_datetime_date(df["published_at"])
            else:
                raise ValueError(f"Processed news parquet missing 'date' (symbol={symbol}, year={year})")
        else:
            df["date"] = _ensure_datetime_date(df["date"])

        # Require at least one rich text column for scoring
        if not any(col in df.columns for col in ("headline", "title", "summary")):
            raise ValueError(f"Processed news parquet missing headline/title columns for {symbol} {year}")

        return df.dropna(subset=["date", "symbol"]).reset_index(drop=True)
    
    def _load_ohlcv_features(self, key, filename):
        logger.debug("Attempting to download OHLCV artifact %s -> %s", key, filename)
        if not self.s3_features.object_exists(key):
            logger.warning(
                "Missing ohlcv features file at s3://%s/%s",
                self.cfg.features_bucket,
                key
            )
            return None
        
        local_path = self.local_dir / filename
        self.s3_features.s3_client.download_file(self.cfg.features_bucket, key, str(local_path))
        logger.debug("Downloaded OHLCV artifact to %s", local_path)
        return local_path

    def _load_all_news(self) -> pd.DataFrame:
        logger.info("Loading processed news for symbols=%s years=%s-%s", self.cfg.symbols, self.cfg.start_year, self.cfg.end_year)
        frames: List[pd.DataFrame] = []
        for symbol in self.cfg.symbols:
            for year in range(self.cfg.start_year, self.cfg.end_year + 1):
                df = self._load_news_year(symbol, year)
                if not df.empty:
                    frames.append(df)
                    logger.debug("Accumulated %d rows for symbol=%s after year=%s", len(df), symbol, year)
        if not frames:
            raise RuntimeError("No processed news data found for requested symbols/years.")
        out = pd.concat(frames, ignore_index=True)
        out = out.dropna(subset=["symbol", "date"]).reset_index(drop=True)
        logger.info("Loaded %d processed news rows across %d symbol-year files", len(out), len(frames))
        return out

    # ---------- Scoring ----------
    def _ensure_sentiment_columns(self, articles: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        df = articles.copy()
        existing = {"finbert_label", "finbert_score"}.issubset(df.columns)

        if cfg.score_mode == "skip" and existing:
            logger.info("score_mode=skip and existing FinBERT columns present → reusing scores")
        elif cfg.score_mode == "skip" and not existing:
            logger.warning("score_mode=skip but FinBERT columns missing → defaulting to neutral sentiment")
            df["finbert_label"] = "neutral"
            df["finbert_score"] = 0.5
        else:
            logger.debug("Running FinBERT scoring for %d articles", len(df))
            # text_cols = [
            #     c
            #     # for c in ["title", "headline", "summary", "description", "content"]
            #     for c in ["headline", "summary"]
            #     if c in df.columns
            # ]
            # if not text_cols:
            #     raise ValueError("No usable text columns available for FinBERT scoring.")
            # df["_text"] = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
            # empty_mask = df["_text"].str.len() == 0
            # df.loc[empty_mask, "_text"] = "."

            # labels, scores = score_finbert(
            #     df["_text"].tolist(),
            #     batch_size=cfg.finbert_batch_size,
            #     max_length=cfg.finbert_max_length,
            #     device=cfg.finbert_device,
            # )
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
            df["finbert_label"] = labels
            df["finbert_score"] = scores
            df = df.drop(columns=["_text"], errors="ignore")
            pos = sum(str(lbl).lower() == "positive" for lbl in labels)
            neg = sum(str(lbl).lower() == "negative" for lbl in labels)
            logger.debug("Finished FinBERT scoring; positive=%d negative=%d", pos, neg)

        lab = df["finbert_label"].astype(str).str.lower()
        df["polarity"] = np.where(lab == "positive", 1.0, np.where(lab == "negative", -1.0, 0.0))

        if "relevance_score" not in df.columns:
            df["relevance_score"] = 0.0
        df["relevance_score"] = pd.to_numeric(df["relevance_score"], errors="coerce").fillna(0.0)
        df["finbert_score"] = pd.to_numeric(df["finbert_score"], errors="coerce").fillna(0.5)
        return df

    # ---------- Feature engineering ----------
    def _build_daily_features(self, articles: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        df = articles.copy()
        logger.debug("Building daily features from %d article rows", len(df))
        df["date"] = _ensure_datetime_date(df["date"])
        df["is_highrel"] = (df.get("relevance_score", 0.0) >= 0.30).astype(int)
        df["w"] = (df["relevance_score"].clip(lower=0) + 1e-6) * df["finbert_score"].clip(lower=1e-6)

        grp = df.groupby(["symbol", "date"], as_index=False)
        grp_plain = df.groupby(["symbol", "date"], sort=False)
        out = grp.agg(
            article_count=("polarity", "size"),
            pos_count=("finbert_label", lambda x: int((pd.Series(x).str.lower() == "positive").sum())),
            neg_count=("finbert_label", lambda x: int((pd.Series(x).str.lower() == "negative").sum())),
            neu_count=("finbert_label", lambda x: int((pd.Series(x).str.lower() == "neutral").sum())),
            highrel_count=("is_highrel", "sum"),
            rel_score_mean=("relevance_score", "mean"),
            sent_conf_mean=("finbert_score", "mean"),
            polarity_mean=("polarity", "mean"),
            polarity_std=("polarity", "std"),
        )

        wp = grp_plain.apply(lambda g: _weighted_mean(g["polarity"], g["w"]))
        wp = wp.rename("polarity_wmean").reset_index()
        out = out.merge(wp, on=["symbol", "date"], how="left")

        hr_mean = grp_plain.apply(
            lambda g: float(g.loc[g["is_highrel"] == 1, "polarity"].mean()) if (g["is_highrel"] == 1).any() else np.nan
        ).rename("highrel_polarity_mean").reset_index()
        hr_wmean = grp_plain.apply(
            lambda g: _weighted_mean(
                g.loc[g["is_highrel"] == 1, "polarity"],
                g.loc[g["is_highrel"] == 1, "w"],
            ) if (g["is_highrel"] == 1).any() else np.nan,
        ).rename("highrel_polarity_wmean").reset_index()
        out = out.merge(hr_mean, on=["symbol", "date"], how="left")
        out = out.merge(hr_wmean, on=["symbol", "date"], how="left")

        out["article_count"] = out["article_count"].astype(float)
        out["pos_ratio"] = out["pos_count"] / out["article_count"].replace(0, np.nan)
        out["neg_ratio"] = out["neg_count"] / out["article_count"].replace(0, np.nan)
        out["neu_ratio"] = out["neu_count"] / out["article_count"].replace(0, np.nan)
        out["highrel_ratio"] = out["highrel_count"] / out["article_count"].replace(0, np.nan)

        out["sent_cov"] = out["polarity_wmean"] * np.log1p(out["article_count"].fillna(0))
        out["sent_abs"] = out["polarity_wmean"].abs()
        out["imbalance"] = out["pos_ratio"].fillna(0) - out["neg_ratio"].fillna(0)
        out["highrel_impact"] = out["highrel_polarity_wmean"] * out["highrel_ratio"].fillna(0)
        out["news_present"] = (out["article_count"].fillna(0) > 0).astype(float)

        out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

        base_cols = [
            "article_count",
            "pos_ratio",
            "neg_ratio",
            "neu_ratio",
            "highrel_ratio",
            "rel_score_mean",
            "sent_conf_mean",
            "polarity_mean",
            "polarity_std",
            "polarity_wmean",
            "highrel_polarity_mean",
            "highrel_polarity_wmean",
            "sent_cov",
            "sent_abs",
            "imbalance",
            "highrel_impact",
            "news_present",
        ]

        for lag in range(1, cfg.lags + 1):
            for col in base_cols:
                out[f"{col}_lag_{lag}"] = out.groupby("symbol", sort=False)[col].shift(lag)

        for window in cfg.roll_windows:
            shifted_cov = out.groupby("symbol", sort=False)["sent_cov"].shift(1)
            shifted_ac = out.groupby("symbol", sort=False)["article_count"].shift(1)
            out[f"sent_cov_roll_mean_{window}"] = shifted_cov.rolling(window, min_periods=max(2, window // 2)).mean()
            out[f"sent_cov_roll_std_{window}"] = shifted_cov.rolling(window, min_periods=max(2, window // 2)).std()
            out[f"article_count_roll_mean_{window}"] = shifted_ac.rolling(window, min_periods=max(2, window // 2)).mean()

        for col in ["polarity_wmean", "sent_cov", "article_count"]:
            mu = out.groupby("symbol", sort=False)[col].shift(1).rolling(
                cfg.shock_window,
                min_periods=max(5, cfg.shock_window // 4),
            ).mean()
            sigma = out.groupby("symbol", sort=False)[col].shift(1).rolling(
                cfg.shock_window,
                min_periods=max(5, cfg.shock_window // 4),
            ).std()
            z = (out[col] - mu) / sigma.replace(0, np.nan)
            z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out[f"{col}_z{cfg.shock_window}"] = z
            out[f"{col}_pos_shock"] = (z > cfg.shock_z).astype(float)
            out[f"{col}_neg_shock"] = (z < -cfg.shock_z).astype(float)

        feature_cols = [c for c in out.columns if c not in {"symbol", "date"}]
        out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        logger.info("Constructed %d daily feature rows with %d feature columns", len(out), len(feature_cols))
        return out

    # ---------- Label attachment ----------
    def _attach_label_if_available(self, feats: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
        cfg = self.cfg
        supervision = {"enabled": False}
        if not cfg.attach_label:
            return feats, supervision

        # ohlcv_path = Path(cfg.ohlcv_features_path)
        # meta_path = Path(cfg.ohlcv_metadata_path)
        ohlcv_path = self._load_ohlcv_features(self.cfg.ohlcv_features_path, self.cfg.ohlcv_filename)
        meta_path = self._load_ohlcv_features(self.cfg.ohlcv_metadata_path, self.cfg.ohlcv_metadata_filename)
        if not ohlcv_path or not meta_path:
            logger.warning(
                "attach_label=True but OHLCV feature/meta files not found: (%s, %s) → skipping label merge",
                ohlcv_path,
                meta_path,
            )
            return feats, supervision

        ohlcv = pd.read_parquet(ohlcv_path)
        ohlcv["date"] = _ensure_datetime_date(ohlcv["date"])
        meta = json.loads(meta_path.read_text())
        label_col = _infer_label_col(meta, cfg.label_col_fallback)
        if label_col not in ohlcv.columns:
            logger.warning("Label column %s missing in OHLCV features → skipping labels", label_col)
            return feats, supervision

        labels = ohlcv[["symbol", "date", label_col]].dropna(subset=[label_col]).copy()
        merged = feats.merge(labels, on=["symbol", "date"], how="inner")
        logger.info("Attached labels using column %s; rows reduced from %d to %d", label_col, len(feats), len(merged))
        supervision = {
            "enabled": True,
            "label_col_name": label_col,
            "source": {
                "ohlcv_features_path": cfg.ohlcv_features_path,
                "ohlcv_metadata_path": cfg.ohlcv_metadata_path,
            },
            "horizon_days": meta.get("horizon_days"),
            "splits": meta.get("splits"),
        }
        return merged, supervision

    # ---------- Output ----------
    def _write_outputs(self, feats: pd.DataFrame, supervision: Dict[str, object]) -> Dict[str, str]:
        feat_key = f"{self.cfg.output_key_prefix}/{self.cfg.output_features_filename}"
        meta_key = f"{self.cfg.output_key_prefix}/{self.cfg.output_metadata_filename}"

        feature_cols = [c for c in feats.columns if c not in {"symbol", "date"}]
        meta = {
            "name": "news_sentiment_features",
            "symbols": self.cfg.symbols,
            "year_range": [self.cfg.start_year, self.cfg.end_year],
            "feature_cols": feature_cols,
            "n_features": len(feature_cols),
            "n_rows": int(len(feats)),
            "supervision": supervision,
            "feature_engineering": {
                "lags": self.cfg.lags,
                "roll_windows": list(self.cfg.roll_windows),
                "shock_window": self.cfg.shock_window,
                "shock_z": self.cfg.shock_z,
                "causal": True,
                "note": "Lag/roll/shock features rely on shift(1) so they only depend on prior days.",
            },
            "scoring": {
                "score_mode": self.cfg.score_mode,
                "batch_size": self.cfg.finbert_batch_size,
                "max_length": self.cfg.finbert_max_length,
                "device": self.cfg.finbert_device,
            },
        }
        payload = json.dumps(meta, indent=2)
        logger.debug("Metadata prepared with %d feature columns", len(feature_cols))

        logger.info("Writing NEWS features to s3://%s/%s", self.cfg.features_bucket, feat_key)
        self._write_local_and_s3(
            filename=self.cfg.output_features_filename,
            s3_key=feat_key,
            write_func=lambda path: feats.to_parquet(path, index=False),
        )

        logger.info("Writing NEWS metadata to s3://%s/%s", self.cfg.features_bucket, meta_key)
        self._write_local_and_s3(
            filename=self.cfg.output_metadata_filename,
            s3_key=meta_key,
            write_func=lambda path: path.write_text(payload, encoding="utf-8"),
        )

        return {"features": feat_key, "metadata": meta_key}

    def _write_local_and_s3(self, filename: str, s3_key: str, write_func: Callable[[Path], None]) -> None:
        """Write artifact locally (for DVC) and upload the same temp file to S3."""
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td) / filename
            logger.debug("Writing %s to temp path %s", filename, tmp_path)
            write_func(tmp_path)

            local_path = self.local_dir / filename
            logger.debug("Writing %s to local path %s", filename, local_path)
            write_func(local_path)

            if self.cfg.upload_to_s3:
                logger.debug("Uploading %s to s3://%s/%s", filename, self.cfg.features_bucket, s3_key)
                self.s3_features.upload_file(tmp_path, s3_key)
            else:
                logger.info("Skipping upload for %s because upload_to_s3=False", filename)

    # ---------- Orchestration ----------
    def run(self) -> None:
        news = self._load_all_news()
        logger.info(
            "Loaded processed news: %d rows across symbols=%s (date range %s → %s)",
            len(news),
            self.cfg.symbols,
            news["date"].min(),
            news["date"].max(),
        )

        news = self._ensure_sentiment_columns(news)
        feats = self._build_daily_features(news)
        logger.info("Daily features shape: rows=%d cols=%d", len(feats), len(feats.columns))

        merged, supervision = self._attach_label_if_available(feats)
        if supervision.get("enabled"):
            logger.info(
                "Attached label column %s; merged rows=%d",
                supervision["label_col_name"],
                len(merged),
            )
        else:
            logger.info("Label attachment disabled or unavailable; rows=%d", len(merged))

        out = self._write_outputs(merged, supervision)
        logger.info(
            "NEWS sentiment feature build complete. Features=%s Metadata=%s",
            out["features"],
            out["metadata"],
        )


# -----------------------------
# CLI
# -----------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily sentiment features from processed news.")
    parser.add_argument("--symbols", required=True, help="Comma-separated list, e.g., TCS,INFY")
    parser.add_argument("--start-year", required=True, type=int)
    parser.add_argument("--end-year", required=True, type=int)
    parser.add_argument("--s3-processed-bucket", required=True, help="Your S3 processed bucket (e.g., dissertation-databucket-processed)")
    parser.add_argument("--s3-features-bucket", required=True, help="Your S3 features bucket (e.g., dissertation-databucket-dvcstore)")

    parser.add_argument("--score-mode", default="finbert", choices=["finbert", "skip"])
    parser.add_argument("--finbert-batch-size", type=int, default=16)
    parser.add_argument("--finbert-max-length", type=int, default=128)
    parser.add_argument("--finbert-device", type=int, default=-1)

    parser.add_argument("--lags", type=int, default=5)
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))

    # parser.add_argument("--attach-label", action="store_true", help="Enable attaching OHLCV-derived label column")
    # parser.add_argument("--no-attach-label", action="store_true", help="Disable label attachment")
    # parser.add_argument("--ohlcv-features-path", default="data/features/ohlcv_features.parquet")
    # parser.add_argument("--ohlcv-metadata-path", default="data/features/ohlcv_feature_metadata.json")

    # parser.add_argument("--no-s3-upload", action="store_true", help="Skip uploading outputs to S3")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging(args.log_level)

    cfg = NewsFeatureBuildConfig(
        processed_bucket=args.s3_processed_bucket,
        features_bucket=args.s3_features_bucket,
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        score_mode=args.score_mode,
        finbert_batch_size=args.finbert_batch_size,
        finbert_max_length=args.finbert_max_length,
        finbert_device=args.finbert_device,
        lags=int(args.lags),
        # attach_label=(False if args.no_attach_label else True),
        # ohlcv_features_path=args.ohlcv_features_path,
        # ohlcv_metadata_path=args.ohlcv_metadata_path,
        # upload_to_s3=not args.no_s3_upload,
    )

    logger.info("Config: %s", json.dumps(asdict(cfg), indent=2, default=str))
    pipeline = NewsSentimentFeatureBuildPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
