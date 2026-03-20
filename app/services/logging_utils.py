import logging
import sys
import time
from contextlib import contextmanager


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

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