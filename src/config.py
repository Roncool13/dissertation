import logging
import os
from typing import Optional


def setup_logging(level: Optional[str] = None) -> None:
    """
    Configure application-wide logging.

    If a level is not provided, LOG_LEVEL env var is used (defaults to INFO).
    Calling this multiple times is safe; subsequent calls will be no-ops
    if handlers are already configured.
    """
    if logging.getLogger().handlers:
        return

    env_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=env_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
