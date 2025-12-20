# run_raw_desiquant_ingestion.py

# Standard library imports
import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

# Add parent directory to path for local imports (consistent with run_data_ingestion)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Local imports
from src.config import setup_logging
import src.constants as constants
from src.core.data_download import (
    DesiquantCandlesMirror,
    DesiquantNewsMirror,
    DesiquantCorporateAnnouncementsMirror,
    DesiquantFinancialResultsMirror
)




@dataclass
class RawDesiquantConfig:
    """Configuration for raw Desiquant ingestion."""

    symbols: List[str]
    AWS_BUCKET_NAME: str
    overwrite: bool = False
    log_level: Optional[str] = None


class RawDesiquantIngestor:
    """
    Encapsulates mirroring Desiquant candles, news, financials and corporate announcements into the raw S3 zone.
    """

    logger = logging.getLogger(__name__)

    def __init__(self, config: RawDesiquantConfig):
        self.config = config
        self.candles_mirror = DesiquantCandlesMirror()
        self.news_mirror = DesiquantNewsMirror()
        self.announcements_mirror = DesiquantCorporateAnnouncementsMirror()
        self.financials_mirror = DesiquantFinancialResultsMirror()

    def run(self) -> None:
        setup_logging(self.config.log_level)
        self.logger.info(
            "Starting raw Desiquant ingestion for %d symbol(s) into bucket %s",
            len(self.config.symbols),
            self.config.AWS_BUCKET_NAME,
        )

        for sym in self.config.symbols:
            symbol = sym.strip().upper()
            self.logger.info("Processing symbol %s (overwrite=%s)", symbol, self.config.overwrite)
            self.candles_mirror.mirror_to_raw(symbol=symbol, overwrite=self.config.overwrite)
            self.news_mirror.mirror_to_raw(symbol=symbol, overwrite=self.config.overwrite)
            self.announcements_mirror.mirror_to_raw(symbol=symbol, overwrite=self.config.overwrite)
            self.financials_mirror.mirror_to_raw(symbol=symbol, overwrite=self.config.overwrite)

        self.logger.info("Raw Desiquant ingestion complete.")


# ---- CLI ----
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror Desiquant candles into the raw S3 zone.")
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols to mirror (e.g. TCS INFY HDFCBANK)")
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=constants.S3_BUCKET,
        help="Target S3 bucket for raw data (defaults to constants.S3_BUCKET)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite if object already exists in S3.")
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Logging level (e.g. DEBUG, INFO). Defaults to LOG_LEVEL env or INFO.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = RawDesiquantConfig(
        symbols=args.symbols,
        AWS_BUCKET_NAME=args.s3_bucket,
        overwrite=args.overwrite,
        log_level=args.log_level,
    )
    RawDesiquantIngestor(config).run()


if __name__ == "__main__":
    main()
