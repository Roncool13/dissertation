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

            logger.error("Unexpected S3 HeadObject error for s3://%s/%s: %s", self.bucket, key, exc.response.get("Error", {}))
            raise

    def object_exists_with_pattern(self, prefix: str, pattern: str = None) -> bool:
        """
        Returns True if at least one object exists matching the prefix/pattern.
        
        Args:
            prefix: S3 prefix to search (e.g., "processed/ohlcv/")
            pattern: Optional glob-style pattern to match against keys (e.g., "ohlcv_*.parquet")
                    If None, returns True if any object exists under the prefix.
        
        Example:
            exists = conn.object_exists_with_pattern("processed/ohlcv/", "ohlcv_*.parquet")
        """
        import fnmatch
        
        logger.debug("Checking for objects at s3://%s/%s with pattern: %s", self.bucket, prefix, pattern or "any")
        
        try:
            # List objects with the given prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                MaxKeys=1000  # Adjust based on expected number of objects
            )
            
            if 'Contents' not in response:
                logger.debug("No objects found at s3://%s/%s", self.bucket, prefix)
                return False
            
            # If no pattern specified, return True if any objects exist
            if pattern is None:
                logger.debug("Found %d object(s) at s3://%s/%s", len(response['Contents']), self.bucket, prefix)
                return True
            
            # Check if any object matches the pattern
            for obj in response['Contents']:
                key = obj['Key']
                # Extract just the filename for pattern matching
                filename = key.split('/')[-1]
                if fnmatch.fnmatch(filename, pattern):
                    logger.debug("Found matching object at s3://%s/%s", self.bucket, key)
                    return True
            
            logger.debug("No objects matching pattern '%s' at s3://%s/%s", pattern, self.bucket, prefix)
            return False
            
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("403", "AccessDenied"):
                logger.warning(
                    "ListObjects forbidden (403) for s3://%s/%s. Treating as exists to avoid re-ingestion.",
                    self.bucket, prefix
                )
                return True
            
            logger.error("Unexpected S3 ListObjects error for s3://%s/%s: %s", self.bucket, prefix, exc.response.get("Error", {}))
            raise

    def list_objects_with_pattern(self, prefix: str, pattern: str = None) -> list[str]:
        """
        Returns a list of S3 keys matching the prefix/pattern.
        
        Args:
            prefix: S3 prefix to search (e.g., "processed/ohlcv/")
            pattern: Optional glob-style pattern to match (e.g., "ohlcv_*.parquet")
        
        Returns:
            List of matching S3 keys
        """
        import fnmatch
        
        logger.debug("Listing objects at s3://%s/%s with pattern: %s", self.bucket, prefix, pattern or "any")
        
        try:
            matching_keys = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    key = obj['Key']
                    if pattern is None:
                        matching_keys.append(key)
                    else:
                        filename = key.split('/')[-1]
                        if fnmatch.fnmatch(filename, pattern):
                            matching_keys.append(key)
            
            logger.debug("Found %d matching object(s) at s3://%s/%s", len(matching_keys), self.bucket, prefix)
            return matching_keys
            
        except ClientError as exc:
            logger.error("Error listing objects at s3://%s/%s: %s", self.bucket, prefix, exc.response.get("Error", {}))
            raise

    def upload_file(self, local_path: str | Path, key: str) -> None:
        logger.info("Uploading %s to s3://%s/%s", local_path, self.bucket, key)
        self.s3_client.upload_file(str(local_path), self.bucket, key)
        logger.info("Upload complete for s3://%s/%s", self.bucket, key)

    def download_file(self, key: str, local_path: str | Path) -> None:
        logger.info("Downloading s3://%s/%s to %s", self.bucket, key, local_path)
        self.s3_client.download_file(self.bucket, key, str(local_path))
        logger.info("Download complete for s3://%s/%s", self.bucket, key)