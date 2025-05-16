# core/debug_utils.py
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any, Callable, TypeVar, Union
from functools import wraps
import time
import json
import traceback
from pathlib import Path
from pydantic import BaseModel

# Type variable for decorators
F = TypeVar('F', bound=Callable[..., Any])

class DebugInfo(BaseModel):
    """Structured debug information for all components."""
    timestamp: str
    component: str
    action: str
    details: Dict[str, Any] = {}
    duration_ms: float
    success: bool
    error: Optional[str] = None

class PerformanceMonitor:
    """Performance monitoring utility."""
    
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
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
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
                        'kwargs': str(kwargs) if kwargs else None,
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

def log_async_execution_time(logger: logging.Logger):
    """
    Decorator to log execution time of async functions with structured debug info.
    
    Usage:
    @log_async_execution_time(logger)
    async def my_async_function():
        ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                
                debug_info = DebugInfo(
                    timestamp=datetime.now().isoformat(),
                    component=func.__module__ or 'unknown',
                    action=func.__name__,
                    details={
                        'args': str(args) if args else None,
                        'kwargs': str(kwargs) if kwargs else None,
                        'is_async': True
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
                        'is_async': True,
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

# Keeping the existing PerformanceMonitor class instead of this duplicate
