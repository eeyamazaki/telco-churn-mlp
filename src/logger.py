"""Structured logging configuration using structlog.

Usage in any module:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("training started", epoch=1, lr=0.001)
"""

import logging
import sys

import structlog

_configured = False


def setup_logging(level: str = "INFO") -> None:
    """Configure structlog with console-friendly output. Idempotente."""
    global _configured  # noqa: PLW0603
    if _configured:
        return
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    """Return a named logger. Calls setup_logging on first use."""
    setup_logging()
    return structlog.get_logger(name)
