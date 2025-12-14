# src/connections.py
# Standard library imports
import logging
from zipfile import Path

# Third party imports
import boto3

logger = logging.getLogger(__name__)


class S3Connection:
    """Manages S3 connection details."""

    def __init__(self, bucket: str):
        self.bucket = bucket
        logger.info("Initializing S3 client for bucket %s", bucket)
        self.s3_client = boto3.client("s3")

    def object_exists(self, key: str) -> bool:
        logger.debug("Checking if s3://%s/%s exists", self.bucket, key)
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            logger.debug("Found object at s3://%s/%s", self.bucket, key)
            return True
        except self.s3_client.exceptions.ClientError as exc:
            logger.debug("Object s3://%s/%s not found or inaccessible: %s", self.bucket, key, exc)
            return False

    def upload_file(self, local_path: str | Path, key: str) -> None:
        logger.info("Uploading %s to s3://%s/%s", local_path, self.bucket, key)
        self.s3_client.upload_file(str(local_path), self.bucket, key)
        logger.info("Upload complete for s3://%s/%s", self.bucket, key)
