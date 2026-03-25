import logging
import sys
import time
from contextlib import contextmanager

from app.config import settings


_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED

    if _CONFIGURED:
        return

    root_logger = logging.getLogger()

    if root_logger.handlers:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

    level_name = settings.LOG_LEVEL.upper()
    default_level = getattr(logging, level_name, logging.INFO)

    if settings.is_dev:
        root_logger.setLevel(default_level)
    else:
        # Never show DEBUG logs outside dev
        root_logger.setLevel(max(default_level, logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    logger = logging.getLogger(name)

    if settings.is_dev:
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    else:
        logger.setLevel(max(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO), logging.INFO))

    logger.propagate = True
    return logger


@contextmanager
def log_timing(logger: logging.Logger, step_name: str, request_id: str = "-"):
    start = time.perf_counter()
    logger.info(f"[request_id={request_id}] START {step_name}")
    try:
        yield
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(f"[request_id={request_id}] END {step_name} | duration_ms={duration_ms}")