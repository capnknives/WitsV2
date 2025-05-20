import logging
from functools import wraps
import time
from typing import Any, Callable, AsyncGenerator

def log_async_execution_time(logger: logging.Logger):
    """Decorator to log execution time for async functions and async generators."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                async for item in func(*args, **kwargs):
                    yield item
                elapsed_time = time.time() - start_time
                logger.debug(f"{func.__name__} completed in {elapsed_time:.2f} seconds")
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {elapsed_time:.2f} seconds: {str(e)}")
                raise
        return wrapper
    return decorator
