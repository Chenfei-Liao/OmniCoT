import functools
import random
import time
from typing import Callable, Optional, Tuple, Type


def _retry_settings(instance) -> dict:
    config = getattr(instance, 'config', {}) or {}
    generation = config.get('generation', {}) if isinstance(config, dict) else {}
    return {
        'max_retries': int(generation.get('max_retries', 3)),
        'retry_delay': float(generation.get('retry_delay', 5)),
        'exponential_backoff': bool(generation.get('retry_exponential_backoff', True)),
        'retry_max_delay': float(generation.get('retry_max_delay', 60)),
    }


def api_retry_with_backoff(
    max_retries: Optional[int] = None,
    retry_delay: Optional[float] = None,
    retry_max_delay: Optional[float] = None,
    exponential_backoff: Optional[bool] = None,
    retry_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> Callable:
    """Retry OpenAI-compatible API calls with bounded exponential backoff."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            instance_settings = _retry_settings(args[0]) if args else {}
            attempts = max_retries if max_retries is not None else instance_settings.get('max_retries', 3)
            base_delay = retry_delay if retry_delay is not None else instance_settings.get('retry_delay', 5.0)
            max_delay = retry_max_delay if retry_max_delay is not None else instance_settings.get('retry_max_delay', 60.0)
            use_backoff = (
                exponential_backoff
                if exponential_backoff is not None
                else instance_settings.get('exponential_backoff', True)
            )

            last_error = None
            for attempt in range(max(1, attempts)):
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as error:
                    last_error = error
                    if attempt >= attempts - 1:
                        raise

                    delay = base_delay
                    if use_backoff:
                        delay = min(max_delay, base_delay * (2 ** attempt))

                    jitter = random.uniform(0, min(1.0, delay * 0.1))
                    time.sleep(delay + jitter)

            raise last_error

        return wrapper

    return decorator

