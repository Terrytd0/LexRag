"""Structured logging configuration, applied once at process startup.

Usage:
    from configs.logging import configure_logging
    configure_logging()
"""

from __future__ import annotations

import logging
import sys

from configs.settings import get_settings

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s :: %(message)s"


def configure_logging() -> None:
    """Configure root logging handlers/level from settings. Idempotent."""
    settings = get_settings()
    root = logging.getLogger()
    if root.handlers:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())
