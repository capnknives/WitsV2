import logging
from functools import wraps
import time
from typing import Any, Callable

def log_async_execution_time(logger: logging.Logger):
    """Decorator to log execution time for async functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                logger.debug(f"{func.__name__} completed in {elapsed_time:.2f} seconds")
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {elapsed_time:.2f} seconds: {str(e)}")
                raise
        return wrapper
    return decorator
