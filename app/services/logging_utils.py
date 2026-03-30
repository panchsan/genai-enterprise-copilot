import logging
import sys
import time
from contextlib import contextmanager

from app.config import settings


_CONFIGURED = False


NOISY_LOGGERS = [
    "azure",
    "azure.core",
    "azure.identity",
    "azure.storage",
    "azure.storage.blob",
    "azure.search",
    "azure.search.documents",
    "httpx",
    "urllib3",
    "openai",
]


def _resolve_level(level_name: str, fallback: int = logging.INFO) -> int:
    return getattr(logging, (level_name or "").upper(), fallback)


def configure_logging() -> None:
    global _CONFIGURED

    if _CONFIGURED:
        return

    root_logger = logging.getLogger()

    if root_logger.handlers:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

    app_level = _resolve_level(settings.LOG_LEVEL, logging.INFO)

    if settings.is_dev:
        root_logger.setLevel(app_level)
    else:
        root_logger.setLevel(max(app_level, logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Suppress noisy SDK/client logs
    sdk_level = _resolve_level(settings.SDK_LOG_LEVEL, logging.WARNING)
    for logger_name in NOISY_LOGGERS:
        noisy_logger = logging.getLogger(logger_name)
        noisy_logger.setLevel(sdk_level)
        noisy_logger.propagate = True

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    logger = logging.getLogger(name)

    app_level = _resolve_level(settings.LOG_LEVEL, logging.INFO)

    if settings.is_dev:
        logger.setLevel(app_level)
    else:
        logger.setLevel(max(app_level, logging.INFO))

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