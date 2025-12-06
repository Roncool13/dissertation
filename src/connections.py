# src/connections.py
# Standard library imports
from zipfile import Path

# Third party imports
import boto3


class S3Connection:
    """Manages S3 connection details."""

    def __init__(self, bucket: str):
        self.bucket = bucket
        self.s3_client = boto3.client("s3")

    def object_exists(self, key: str) -> bool:
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.s3_client.exceptions.ClientError:
            return False

    def upload_file(self, local_path: str | Path, key: str) -> None:
        self.s3_client.upload_file(str(local_path), self.bucket, key)