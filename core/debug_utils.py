# Welcome to debug paradise! Where we turn chaos into... organized chaos! \\o/
import logging  # Because print() is so 2010 =P
import logging.handlers
import os
import sys
from datetime import datetime  # Time tracking for the time lords! ^_^
from typing import Optional, Dict, Any, Callable, TypeVar, Union, Awaitable, AsyncGenerator, overload, cast  # Type hints galore! O.o
from typing_extensions import ParamSpec  # Extra typing powers! \\o/
from functools import wraps  # Decorator magic! ✨
import time  # For when we need to know how slow things are x.x
import json  # JSON: Because string parsing is for masochists! >.>
import traceback  # When things go boom, we want details! =D
from pathlib import Path  # Paths made easy! ^_^
from pydantic import BaseModel  # Our data validation bestie! \\o/
import inspect  # For when we need to get creepily introspective O.o

# Type variables for our fancy decorators (because we're organized like that! =P)
P = ParamSpec('P')  # P is for Parameters! ^_^
R = TypeVar('R')  # R is for Return! \\o/
F = TypeVar('F', bound=Callable[..., Any])  # F is for Function! =D
YT = TypeVar('YT')  # YT is for Yield Type! (async stuff is complicated x.x)

# Helper types for async functions (because async isn't complicated enough! >.>)
AsyncCallable = Callable[P, Awaitable[R]]  # Regular async functions
AsyncGeneratorCallable = Callable[P, AsyncGenerator[YT, None]]  # Async generators (spicy! O.o)

# A type that can be either kind of async function (we're flexible like that! ^_^)
F_async_flexible = TypeVar('F_async_flexible', bound=Callable[..., Union[Awaitable[Any], AsyncGenerator[Any, None]]])


class DebugInfo(BaseModel):
    """
    Our super-organized debug info container! Everything has its place! \\o/
    Think of it as a fancy box where we put all our debug treasures! ^_^
    """
    timestamp: str  # When did it happen? Time is important! =D
    component: str  # Who done it? O.o
    action: str  # What were they trying to do? >.>
    details: Dict[str, Any] = {}  # All the juicy details! \\o/
    duration_ms: float  # How long did it take? (Time is money! x.x)
    success: bool  # Did it work? *crosses fingers* ^_^
    error: Optional[str] = None  # If it didn't work, what went wrong? =P
    
    # Extra fields for LLM debugging (because LLMs need extra attention! \\o/)
    model_name: Optional[str] = None  # Which AI friend are we talking to? ^_^
    prompt_length: Optional[int] = None  # How chatty were we? O.o
    response_length: Optional[int] = None  # How chatty was the AI? =D
    response_preview: Optional[str] = None  # A sneak peek at what they said! >.>
    tokens_processed: Optional[int] = None  # How many tokens did we feed it? x.x
    tokens_generated: Optional[int] = None  # How many tokens did it spit out? =P
    error_message: Optional[str] = None  # Detailed oopsie report! \\o/

class PerformanceMonitor:
    """Time to see how fast (or slow) things are! Ready, set, go! ^_^"""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = time.time()
        self.checkpoints: Dict[str, float] = {}
        self.logger = logging.getLogger(f'WITS.Performance.{name}')
    
    def checkpoint(self, name: str) -> None:
        """Record a timing checkpoint."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        self.checkpoints[name] = elapsed
        self.logger.debug(f"Checkpoint '{name}': {elapsed*1000:.2f}ms")
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all recorded metrics."""
        metrics = {k: round(v*1000, 2) for k, v in self.checkpoints.items()}
        metrics['total_ms'] = round((time.time() - self.start_time)*1000, 2)
        return metrics

def log_debug_info(logger: logging.Logger, info: DebugInfo) -> None:
    """Log structured debug information."""
    if info.success:
        log_level = logging.DEBUG
    else:
        log_level = logging.ERROR
    
    logger.log(
        log_level,
        f"{info.component}.{info.action} - "
        f"Duration: {info.duration_ms:.2f}ms - "
        f"Details: {json.dumps(info.details)}"
        + (f" - Error: {info.error}" if info.error else "")
    )

# Configure logging
def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Set up logging with configuration from config.yaml.
    
    Args:
        config: The application configuration dictionary
        
    Returns:
        logging.Logger: Configured logger instance with rotating file handlers 
                       and appropriate log levels
    """
    debug_config = config.get('debug', {})
    log_level = debug_config.get('log_level', 'INFO').upper()
    log_dir = Path(debug_config.get('log_directory', 'logs'))
    
    # Create logs directory if it doesn't exist
    log_dir.mkdir(exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger('WITS')
    logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)-8s | %(message)s'
    )
      # Rotating file handlers - separate files for different severities
    handlers = []
    
    # Debug log (all messages)
    if debug_config.get('file_logging_enabled', True):
        debug_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'wits_debug.log',
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(file_formatter)
        handlers.append(debug_handler)
        
        # Error log (errors and critical only)
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'wits_error.log',
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        handlers.append(error_handler)
        
        # Performance log for metrics
        if debug_config.get('performance_monitoring', True):
            perf_handler = logging.handlers.RotatingFileHandler(
                log_dir / 'wits_performance.log',
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=2,
                encoding='utf-8'
            )
            perf_handler.setLevel(logging.DEBUG)
            perf_handler.setFormatter(file_formatter)
            perf_handler.addFilter(lambda record: 'Performance:' in record.getMessage())
            handlers.append(perf_handler)
    
    # Console handler
    if debug_config.get('console_logging_enabled', True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, debug_config.get('console_log_level', 'INFO').upper()))
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
    
    # Add all handlers to logger
    for handler in handlers:
        logger.addHandler(handler)
    
    return logger

def log_execution_time(logger: logging.Logger):
    """
    Decorator to log execution time of functions with structured debug info.
    
    Usage:
    @log_execution_time(logger)
    def my_function():
        ...
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                
                debug_info = DebugInfo(
                    timestamp=datetime.now().isoformat(),
                    component=func.__module__ or 'unknown',
                    action=func.__name__,
                    details={
                        'args': str(args) if args else None,
                        'kwargs': str(kwargs) if kwargs else None
                    },
                    duration_ms=execution_time,
                    success=True
                )
                log_debug_info(logger, debug_info)
                
                return result
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                debug_info = DebugInfo(
                    timestamp=datetime.now().isoformat(),
                    component=func.__module__ or 'unknown',
                    action=func.__name__,
                    details={
                        'args': str(args) if args else None,
                        'kwargs': str(kwargs) if kwargs else None,
                        'error_type': type(e).__name__,
                        'traceback': traceback.format_exc()
                    },
                    duration_ms=execution_time,
                    success=False,
                    error=str(e)
                )
                log_debug_info(logger, debug_info)
                raise
        
        return wrapper
    return decorator

@overload
def log_async_execution_time(logger: logging.Logger) -> Callable[[AsyncGeneratorCallable[P, YT]], AsyncGeneratorCallable[P, YT]]: ...

@overload
def log_async_execution_time(logger: logging.Logger) -> Callable[[AsyncCallable[P, R]], AsyncCallable[P, R]]: ...

def log_async_execution_time(logger: logging.Logger) -> Callable[[Callable[P, Any]], Callable[P, Any]]:
    """
    Decorator to log execution time of async functions and async generators
    with structured debug info.
    
    Usage:
    @log_async_execution_time(logger)
    async def my_async_function(): # or async def my_async_generator():
        ...
    """
    def decorator(func: Callable[P, Any]) -> Callable[P, Any]:
        func_module = getattr(func, '__module__', 'unknown_module')
        func_name = getattr(func, '__name__', 'unknown_action')

        if inspect.isasyncgenfunction(func):
            # The func is an async generator function.
            # The overload ensures that YT is correctly inferred by the caller.
            # The wrapper itself can be typed with AsyncGenerator[Any, None]
            # as the specific yield type YT is handled by the overload.
            
            @wraps(func) # Pass the original func to wraps
            async def async_gen_wrapper(*args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[Any, None]: # Use Any for yield type here
                start_time = time.time()
                try:
                    # Call the original function (which is an async generator)
                    # The type of func here is Callable[P, AsyncGenerator[Any, None]] due to isasyncgenfunction check
                    # but we cast it to be explicit for the call.
                    async_gen_func = cast(Callable[P, AsyncGenerator[Any, None]], func)
                    async for item in async_gen_func(*args, **kwargs):
                        yield item
                    
                    execution_time = (time.time() - start_time) * 1000
                    debug_info = DebugInfo(
                        timestamp=datetime.now().isoformat(),
                        component=func_module,
                        action=func_name,
                        details={
                            'args': str(args)[:500] if args else None,
                            'kwargs': str(kwargs)[:500] if kwargs else None,
                            'is_async_gen': True
                        },
                        duration_ms=execution_time,
                        success=True
                    )
                    log_debug_info(logger, debug_info)
                except Exception as e:
                    execution_time = (time.time() - start_time) * 1000
                    debug_info = DebugInfo(
                        timestamp=datetime.now().isoformat(),
                        component=func_module,
                        action=func_name,
                        details={
                            'args': str(args)[:500] if args else None,
                            'kwargs': str(kwargs)[:500] if kwargs else None,
                            'is_async_gen': True,
                            'error_type': type(e).__name__,
                            'traceback': traceback.format_exc()[:2000]
                        },
                        duration_ms=execution_time,
                        success=False,
                        error=str(e)
                    )
                    log_debug_info(logger, debug_info)
                    raise
            return async_gen_wrapper
        else:
            # The func is a regular async function (returns an Awaitable).
            # The overload ensures R is correctly inferred by the caller.
            # The wrapper itself can be typed with Awaitable[Any] or async def ... -> Any.
            @wraps(func) # Pass the original func to wraps
            async def awaitable_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any: # Use Any for return type here
                start_time = time.time()
                try:
                    # Call the original async function.
                    # The type of func here is Callable[P, Awaitable[Any]]
                    async_awaitable_func = cast(Callable[P, Awaitable[Any]], func)
                    result = await async_awaitable_func(*args, **kwargs)
                    execution_time = (time.time() - start_time) * 1000
                    debug_info = DebugInfo(
                        timestamp=datetime.now().isoformat(),
                        component=func_module,
                        action=func_name,
                        details={
                            'args': str(args)[:500] if args else None,
                            'kwargs': str(kwargs)[:500] if kwargs else None,
                            'is_async_gen': False
                        },
                        duration_ms=execution_time,
                        success=True
                    )
                    log_debug_info(logger, debug_info)
                    return result
                except Exception as e:
                    execution_time = (time.time() - start_time) * 1000
                    debug_info = DebugInfo(
                        timestamp=datetime.now().isoformat(),
                        component=func_module,
                        action=func_name,
                        details={
                            'args': str(args)[:500] if args else None,
                            'kwargs': str(kwargs)[:500] if kwargs else None,
                            'is_async_gen': False,
                            'error_type': type(e).__name__,
                            'traceback': traceback.format_exc()[:2000]
                        },
                        duration_ms=execution_time,
                        success=False,
                        error=str(e)
                    )
                    log_debug_info(logger, debug_info)
                    raise
            return awaitable_wrapper
    return decorator
