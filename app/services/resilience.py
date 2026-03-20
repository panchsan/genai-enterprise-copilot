import time
from functools import wraps

from app.services.logging_utils import get_logger

logger = get_logger("app.resilience")


class RetryableError(Exception):
    pass


def retry_sync(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    allowed_exceptions: tuple[type[Exception], ...] = (Exception,),
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay_seconds

            while True:
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions as exc:
                    if attempt >= max_attempts:
                        logger.error(
                            f"Retry failed after {attempt} attempts for {func.__name__}: {exc}"
                        )
                        raise

                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {exc}. "
                        f"Retrying in {current_delay}s"
                    )
                    time.sleep(current_delay)
                    attempt += 1
                    current_delay *= backoff_multiplier

        return wrapper

    return decorator