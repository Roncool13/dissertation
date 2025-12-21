# src/io/connections.py
# Standard library imports
import logging
from pathlib import Path

# Third party imports
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3Connection:
    """Manages S3 connection details."""

    def __init__(self, bucket: str):
        self.bucket = bucket
        logger.info("Initializing S3 client for bucket %s", bucket)
        self.s3_client = boto3.client("s3")

    def object_exists(self, key: str) -> bool:
        """
        Returns True if object exists.

        IMPORTANT:
        - 404/NotFound => definitely does not exist
        - 403/AccessDenied => we cannot HEAD; treat as exists to avoid repeated ingestion
          (otherwise pipeline will keep re-downloading every run)
        """
        logger.debug("Checking if s3://%s/%s exists", self.bucket, key)
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            logger.debug("Found object at s3://%s/%s", self.bucket, key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                logger.debug("Object not found at s3://%s/%s (%s)", self.bucket, key, code)
                return False

            if code in ("403", "AccessDenied"):
                logger.warning(
                    "HeadObject forbidden (403) for s3://%s/%s. Treating as exists to avoid re-ingestion. (%s)",
                    self.bucket,
                    key,
                    code,
                )
                return True

            logger.error("Unexpected S3 HeadObject error for s3://%s/%s: %s", self.bucket, key, exc)
            raise

    def upload_file(self, local_path: str | Path, key: str) -> None:
        logger.info("Uploading %s to s3://%s/%s", local_path, self.bucket, key)
        self.s3_client.upload_file(str(local_path), self.bucket, key)
        logger.info("Upload complete for s3://%s/%s", self.bucket, key)

    def download_file(self, key: str, local_path: str | Path) -> None:
        logger.info("Downloading s3://%s/%s to %s", self.bucket, key, local_path)
        self.s3_client.download_file(self.bucket, key, str(local_path))
        logger.info("Download complete for s3://%s/%s", self.bucket, key)