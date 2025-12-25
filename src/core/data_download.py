# src/core/data_download.py

# Standard library imports
import os
import logging
import tempfile
from abc import ABC

# Third-party imports
import pandas as pd

# Local imports
import src.constants.storage as storage_constants
import src.constants.desiquant as desiquant_constants
from src.io.connections import S3Connection

logger = logging.getLogger(__name__)


class _BaseDesiquantMirror(ABC):
    """Shared helpers for Desiquant mirroring to raw S3."""

    def __init__(self, raw_prefix: str, raw_filename: str, desiquant_bucket: str, logger_label: str):
        self.storage_constants = storage_constants
        self.desiquant_constants = desiquant_constants
        self.raw_bucket = storage_constants.S3_BUCKET
        self.raw_prefix = raw_prefix
        self.raw_filename = raw_filename
        self.desiquant_bucket = desiquant_bucket
        self.raw_s3 = S3Connection(bucket=self.raw_bucket)
        self.logger_label = logger_label
        logger.info("[RAW] Initialized %s mirror for bucket=%s, prefix=%s", logger_label, self.raw_bucket, raw_prefix)

    def _desiquant_storage_options(self):
        logger.debug("[RAW] Building storage options for %s endpoint %s", self.logger_label, self.desiquant_constants.DESIQUANT_ENDPOINT_URL)
        return {
            "key": self.desiquant_constants.DESIQUANT_ACCESS_KEY,
            "secret": self.desiquant_constants.DESIQUANT_SECRET_KEY,
            "client_kwargs": {
                "endpoint_url": self.desiquant_constants.DESIQUANT_ENDPOINT_URL,
                "region_name": "auto",
            },
        }

    def _write_parquet_to_raw(self, df: pd.DataFrame, key: str) -> None:
        """Write parquet to raw S3 via the shared connection helper."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            df.to_parquet(tmp.name, index=False)
            temp_path = tmp.name

        try:
            self.raw_s3.upload_file(temp_path, key)
        finally:
            try:
                os.remove(temp_path)
            except OSError as exc:  # pragma: no cover - best-effort cleanup
                logger.warning("Failed to clean up temp parquet %s: %s", temp_path, exc)

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip().lower() for c in df.columns]
        return df

    def _raw_uri(self, key: str) -> str:
        return f"s3://{self.raw_bucket}/{key}"

    def _mirror_single(self, raw_key: str, read_fn, overwrite: bool, log_ident: str) -> str:
        raw_uri = self._raw_uri(raw_key)
        logger.debug("[RAW] %s mirror evaluating overwrite=%s for %s", self.logger_label, overwrite, raw_uri)
        if (not overwrite) and self.raw_s3.object_exists(raw_key):
            logger.info("[RAW] Exists, skipping: %s", raw_uri)
            return raw_uri

        df = read_fn()
        df = self._normalize_columns(df)

        logger.info("[RAW] Writing to: %s", raw_uri)
        self._write_parquet_to_raw(df, raw_key)
        logger.info("[RAW] Completed mirroring for %s", log_ident)
        return raw_uri


class DesiquantCandlesMirror(_BaseDesiquantMirror):
    """Handles download of Desiquant candles and mirroring to the raw S3 zone."""

    def __init__(self):
        super().__init__(
            raw_prefix=storage_constants.RAW_DESIQUANT_OHLCV_PREFIX,
            raw_filename=storage_constants.RAW_OHLCV_FILENAME,
            desiquant_bucket=desiquant_constants.DESIQUANT_CANDLES_BUCKET,
            logger_label="Candles",
        )
        self.default_segment = desiquant_constants.DEFAULT_CANDLES_SEGMENT

    def build_raw_candles_key(self, symbol: str, segment: str | None = None) -> str:
        segment = segment or self.default_segment
        return f"{self.raw_prefix}/{symbol}/{self.raw_filename}"

    # def build_raw_candles_s3_uri(self, symbol: str, segment: str | None = None) -> str:
    #     key = self.build_raw_candles_key(symbol, segment)
    #     logger.debug("[RAW] Computed raw candles key=%s", key)
    #     return self._raw_uri(key)

    def read_desiquant_candles_df(self, symbol: str, segment: str | None = None) -> pd.DataFrame:
        """
        Reads Desiquant candles parquet.gz into a DataFrame using S3-compatible access.
        """
        segment = segment or self.default_segment
        key = self.desiquant_constants.DESIQUANT_CANDLES_KEY_TEMPLATE.format(symbol=symbol, segment=segment)
        src_uri = f"s3://{self.desiquant_bucket}/{key}"

        storage_options = self._desiquant_storage_options()
        logger.info("[RAW] Reading Desiquant candles for %s from %s", symbol, src_uri)
        df = pd.read_parquet(src_uri, storage_options=storage_options)
        logger.debug("[RAW] Retrieved %d rows for %s", len(df), symbol)
        return df

    def mirror_to_raw(self, symbol: str, overwrite: bool = False, segment: str | None = None) -> str:
        """
        Downloads candles for a symbol from Desiquant and stores it in your S3 raw zone.
        """
        raw_key = self.build_raw_candles_key(symbol, segment)
        return self._mirror_single(
            raw_key=raw_key,
            read_fn=lambda: self.read_desiquant_candles_df(symbol=symbol, segment=segment),
            overwrite=overwrite,
            log_ident=f"{symbol} candles",
        )


class DesiquantNewsMirror(_BaseDesiquantMirror):
    """Handles download of Desiquant news and mirroring to the raw S3 zone."""

    def __init__(self):
        super().__init__(
            raw_prefix=storage_constants.RAW_DESIQUANT_NEWS_PREFIX,
            raw_filename=storage_constants.RAW_NEWS_FILENAME,
            desiquant_bucket=desiquant_constants.DESIQUANT_NEWS_BUCKET,
            logger_label="News",
        )
    def build_raw_news_key(self, symbol: str) -> str:
        return f"{self.raw_prefix}/{symbol}/{self.raw_filename}"

    # def build_raw_news_s3_uri(self, symbol: str) -> str:
    #     key = self.build_raw_news_key(symbol)
    #     logger.debug("[RAW] Computed raw news key=%s", key)
    #     return f"s3://{self.raw_bucket}/{key}"

    def read_desiquant_news_df(self, symbol: str) -> pd.DataFrame:
        """
        Reads Desiquant news parquet into a DataFrame using S3-compatible access.
        """
        key = self.desiquant_constants.DESIQUANT_NEWS_KEY_TEMPLATE.format(symbol=symbol)
        src_uri = f"s3://{self.desiquant_bucket}/{key}"

        storage_options = self._desiquant_storage_options()
        logger.info("[RAW] Reading Desiquant news for %s from %s", symbol, src_uri)
        df = pd.read_parquet(src_uri, storage_options=storage_options)
        logger.debug("[RAW] Retrieved %d rows for %s", len(df), symbol)
        return df

    def mirror_to_raw(self, symbol: str, overwrite: bool = False) -> str:
        """
        Downloads news for a symbol from Desiquant and stores it in your S3 raw zone.
        """
        raw_key = self.build_raw_news_key(symbol)
        return self._mirror_single(
            raw_key=raw_key,
            read_fn=lambda: self.read_desiquant_news_df(symbol=symbol),
            overwrite=overwrite,
            log_ident=f"{symbol} news",
        )
    

class DesiquantFinancialResultsMirror(_BaseDesiquantMirror):
    """Handles download of Desiquant financial results and mirroring to the raw S3 zone."""

    def __init__(self):
        super().__init__(
            raw_prefix=storage_constants.RAW_DESIQUANT_FIN_RESULTS_PREFIX,
            raw_filename=storage_constants.RAW_FIN_RESULTS_FILENAME,
            desiquant_bucket=desiquant_constants.DESIQUANT_FINANCIAL_RESULTS_BUCKET,
            logger_label="Financial Results",
        )

    def build_raw_results_key(self, symbol: str) -> str:
        return f"{self.raw_prefix}/{symbol}/{self.raw_filename}"

    # def build_raw_results_s3_uri(self, symbol: str) -> str:
    #     key = self.build_raw_results_key(symbol)
    #     logger.debug("[RAW] Computed raw financial results key=%s", key)
    #     return f"s3://{self.raw_bucket}/{key}"

    def read_desiquant_results_df(self, symbol: str) -> pd.DataFrame:
        """
        Reads Desiquant financial results parquet into a DataFrame using S3-compatible access.
        """
        key = self.desiquant_constants.DESIQUANT_FINANCIAL_RESULTS_KEY_TEMPLATE.format(symbol=symbol)
        src_uri = f"s3://{self.desiquant_bucket}/{key}"

        storage_options = self._desiquant_storage_options()
        logger.info("[RAW] Reading Desiquant financial results for %s from %s", symbol, src_uri)
        df = pd.read_parquet(src_uri, storage_options=storage_options)
        logger.debug("[RAW] Retrieved %d rows for %s", len(df), symbol)
        return df

    def mirror_to_raw(self, symbol: str, overwrite: bool = False) -> str:
        """
        Downloads news for a symbol from Desiquant and stores it in your S3 raw zone.
        """
        raw_key = self.build_raw_results_key(symbol)
        return self._mirror_single(
            raw_key=raw_key,
            read_fn=lambda: self.read_desiquant_results_df(symbol=symbol),
            overwrite=overwrite,
            log_ident=f"{symbol} financial_results",
        )


class DesiquantCorporateAnnouncementsMirror(_BaseDesiquantMirror):
    """Handles download of Desiquant corporate announcements and mirroring to the raw S3 zone."""

    def __init__(self):
        super().__init__(
            raw_prefix=storage_constants.RAW_DESIQUANT_CORP_ANN_PREFIX,
            raw_filename=storage_constants.RAW_CORP_ANN_FILENAME,
            desiquant_bucket=desiquant_constants.DESIQUANT_CORP_ANNOUNCEMENTS_BUCKET,
            logger_label="Corporate Announcements",
        )
        self.source_list = desiquant_constants.SUPPORTED_ANNOUNCEMENTS_SOURCES

    def build_raw_announcements_key(self, symbol: str, source: str = desiquant_constants.DEFAULT_ANNOUNCEMENTS_SOURCE) -> str:
        return f"{self.raw_prefix}/{symbol}/{source}/{self.raw_filename}"

    # def build_raw_announcements_s3_uri(self, symbol: str, source: str = desiquant_constants.DEFAULT_ANNOUNCEMENTS_SOURCE) -> str:
    #     key = self.build_raw_announcements_key(symbol, source)
    #     logger.debug("[RAW] Computed raw corporate announcements key=%s", key)
    #     return self._raw_uri(key)

    def read_desiquant_announcements_df(self, symbol: str, source: str = desiquant_constants.DEFAULT_ANNOUNCEMENTS_SOURCE) -> pd.DataFrame:
        """
        Reads Desiquant corporate announcements parquet into a DataFrame using S3-compatible access.
        """
        key = self.desiquant_constants.DESIQUANT_CORP_ANNOUNCEMENTS_KEY_TEMPLATE.format(source=source, symbol=symbol)
        src_uri = f"s3://{self.desiquant_bucket}/{key}"

        storage_options = self._desiquant_storage_options()
        logger.info("[RAW] Reading Desiquant corporate announcements for %s from %s", symbol, src_uri)
        df = pd.read_parquet(src_uri, storage_options=storage_options)
        logger.debug("[RAW] Retrieved %d rows for %s", len(df), symbol)
        return df

    def mirror_to_raw(self, symbol: str, overwrite: bool = False, source: str = desiquant_constants.DEFAULT_ANNOUNCEMENTS_SOURCE) -> str:
        """
        Downloads corporate announcements for a symbol from Desiquant and stores it in your S3 raw zone.
        """
        for source in self.source_list:
            raw_key = self.build_raw_announcements_key(symbol, source)
            self._mirror_single(
                raw_key=raw_key,
                read_fn=lambda src=source: self.read_desiquant_announcements_df(symbol=symbol, source=src),
                overwrite=overwrite,
                log_ident=f"{symbol} announcements [{source}]",
            )
