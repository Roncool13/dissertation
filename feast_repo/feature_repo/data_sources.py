# feature_repo/data_sources.py
import sys
from pathlib import Path

from feast import FileSource

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import storage as storage_constants

# NOTE:
# This source points to the consolidated engineered-features parquet you build with the feature ingestor.
# In Colab, install s3fs and set AWS creds; Feast (via fsspec) can usually read s3:// paths.
OHLCV_FEATURES_SOURCE = FileSource(
    name="ohlcv_features_source",
    path=f"s3://{storage_constants.S3_BUCKET}/processed/features/*/ohlcv_features.parquet",
    timestamp_field="date",
    created_timestamp_column=None,
)
