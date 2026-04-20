"""Logging utilities for ChorusAgents."""

import logging
import os
from typing import Optional


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(
    level: Optional[str] = None,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> None:
    """
    Configure ChorusAgents's logging output.

    Parameters
    ----------
    level:
        Log level string (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, etc.).
        Defaults to the ``AGENTFABRIC_LOG_LEVEL`` env var, or ``"WARNING"``.
    fmt:
        Log format string.
    """
    level = level or os.environ.get("AGENTFABRIC_LOG_LEVEL", "WARNING")
    numeric = getattr(logging, level.upper(), logging.WARNING)
    logging.basicConfig(level=numeric, format=fmt)
    logging.getLogger("chorusagents").setLevel(numeric)
