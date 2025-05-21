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
    
    # Fields for enhanced AI autonomy system
    autonomy_level: Optional[int] = None  # How autonomous was this action? (0-5) 
    self_modified: Optional[bool] = None  # Did the AI modify its own code? O.o
    learning_source: Optional[str] = None  # Where did the AI learn this behavior?
    tool_simulation: Optional[bool] = None  # Was this a simulated tool run?
    mcp_tool_id: Optional[str] = None  # ID for dynamic MCP tools
    agent_id: Optional[str] = None  # Which agent performed this action?
    parent_agent_id: Optional[str] = None  # If created by another agent, which one?
    code_safety_check: Optional[Dict[str, Any]] = None  # Results of safety checks for code mods
    file_access_details: Optional[Dict[str, Any]] = None  # Details about file access operations

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

class AutonomyMonitor:
    """
    Specialized monitor for tracking AI autonomy operations, learning events,
    and safety-critical operations like self-modifications and file access.
    
    This class extends the standard performance monitoring with specialized
    metrics and tracking capabilities for the enhanced autonomy system.
    """
    
    def __init__(self, component_name: str, agent_id: Optional[str] = None):
        self.name = component_name
        self.agent_id = agent_id
        self.start_time = time.time()
        self.autonomy_level = 0
        self.checkpoints: Dict[str, float] = {}
        self.safety_checks: Dict[str, Any] = {}
        self.learning_events: list[Dict[str, Any]] = []
        self.tool_usage: list[Dict[str, Any]] = []
        self.file_operations: list[Dict[str, Any]] = []
        self.code_modifications: list[Dict[str, Any]] = []
        self.logger = logging.getLogger(f'WITS.Autonomy.{component_name}')
        
    def set_autonomy_level(self, level: int) -> None:
        """Set the autonomy level for this operation (0-5)."""
        self.autonomy_level = max(0, min(5, level))
        self.logger.debug(f"Autonomy level set to {level}")
        
    def record_learning_event(self, source: str, content_type: str, content_preview: str) -> None:
        """Record a learning event where the AI acquired new knowledge."""
        self.learning_events.append({
            "timestamp": time.time(),
            "source": source,
            "content_type": content_type,
            "content_preview": content_preview[:200] if content_preview else ""
        })
        self.logger.debug(f"Learning event recorded from {source}: {content_type}")
        
    def record_tool_usage(self, tool_id: str, is_mcp: bool, success: bool, error_msg: Optional[str] = None) -> None:
        """Record usage of a tool, including MCP dynamic tools."""
        self.tool_usage.append({
            "timestamp": time.time(),
            "tool_id": tool_id,
            "is_mcp": is_mcp,
            "success": success,
            "error": error_msg
        })
        if success:
            self.logger.debug(f"Tool usage: {tool_id} {'(MCP)' if is_mcp else ''} - Success")
        else:
            self.logger.warning(f"Tool usage: {tool_id} {'(MCP)' if is_mcp else ''} - Failed: {error_msg}")
    
    def record_file_operation(self, operation: str, file_path: str, success: bool, 
                              security_cleared: bool, error_msg: Optional[str] = None) -> None:
        """Record file access operation with security details."""
        self.file_operations.append({
            "timestamp": time.time(),
            "operation": operation,
            "file_path": file_path,
            "success": success,
            "security_cleared": security_cleared,
            "error": error_msg
        })
        log_msg = (f"File {operation}: {file_path} - "
                  f"{'Success' if success else 'Failed'} "
                  f"{'(Security cleared)' if security_cleared else '(SECURITY BLOCKED)'}")
        if success:
            self.logger.debug(log_msg)
        else:
            self.logger.warning(f"{log_msg}: {error_msg}")
    
    def record_code_modification(self, file_path: str, modification_type: str, 
                                safety_check_passed: bool, modification_size: int) -> None:
        """Record code modification with safety check results."""
        self.code_modifications.append({
            "timestamp": time.time(),
            "file_path": file_path,
            "modification_type": modification_type,
            "safety_check_passed": safety_check_passed,
            "modification_size": modification_size
        })
        self.logger.debug(f"Code modification: {modification_type} to {file_path} - "
                        f"Safety {'PASSED' if safety_check_passed else 'FAILED'}")
    
    def record_safety_check(self, check_type: str, passed: bool, details: Dict[str, Any]) -> None:
        """Record results of a safety check."""
        self.safety_checks[check_type] = {
            "timestamp": time.time(),
            "passed": passed,
            "details": details
        }
        log_level = logging.DEBUG if passed else logging.WARNING
        self.logger.log(log_level, f"Safety check ({check_type}): {'PASSED' if passed else 'FAILED'}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics about this autonomy operation."""
        total_time = (time.time() - self.start_time) * 1000  # ms
        
        metrics = {
            "duration_ms": round(total_time, 2),
            "autonomy_level": self.autonomy_level,
            "agent_id": self.agent_id,
            "component": self.name,
            "checkpoints": {k: round(v*1000, 2) for k, v in self.checkpoints.items()},
            "learning_events_count": len(self.learning_events),
            "tool_usage_count": len(self.tool_usage),
            "file_operations_count": len(self.file_operations),
            "code_modifications_count": len(self.code_modifications),
            "safety_checks": self.safety_checks,
            # Include recent events (limited number for size control)
            "recent_learning": self.learning_events[-5:] if self.learning_events else [],
            "recent_tool_usage": self.tool_usage[-5:] if self.tool_usage else [],
            "recent_file_ops": self.file_operations[-5:] if self.file_operations else [],
            "recent_code_mods": self.code_modifications[-5:] if self.code_modifications else []
        }
        return metrics
    
    def log_to_debug_info(self) -> DebugInfo:
        """Create a DebugInfo object from this monitor's state for logging."""
        metrics = self.get_metrics()
        
        # Calculate success based on safety checks
        safety_passed = all(check.get("passed", True) for check in self.safety_checks.values())
        tool_success = all(tool.get("success", True) for tool in self.tool_usage)
        file_success = all(op.get("success", True) for op in self.file_operations)
        overall_success = safety_passed and tool_success and file_success
        
        # Format a summary of what happened
        file_paths = [op.get("file_path", "") for op in self.file_operations]
        tools_used = [tool.get("tool_id", "") for tool in self.tool_usage]
        
        # Create the debug info
        return DebugInfo(
            timestamp=datetime.now().isoformat(),
            component=self.name,
            action="autonomy_operation",
            details={
                "metrics": metrics,
                "files_accessed": file_paths[:10],  # limit to 10 for readability
                "tools_used": tools_used[:10]  # limit to 10 for readability
            },
            duration_ms=metrics["duration_ms"],
            success=overall_success,
            error=None if overall_success else "Safety checks failed or operation errors occurred",
            # Autonomy-specific fields
            autonomy_level=self.autonomy_level,
            self_modified=len(self.code_modifications) > 0,
            agent_id=self.agent_id,
            code_safety_check={k: v["passed"] for k, v in self.safety_checks.items()},
            file_access_details={"count": len(self.file_operations), "operations": [op["operation"] for op in self.file_operations][:5]}
        )

def log_autonomy_operation(logger: logging.Logger, component: str = "AutonomyOperation", agent_id: Optional[str] = None):
    """
    Decorator to log autonomy operations with comprehensive monitoring.
    
    Args:
        logger: Logger instance to use for logging
        component: Component name for the autonomy operation
        agent_id: Optional ID of the agent performing the operation
        
    Usage:
    @log_autonomy_operation(logger, "CodeModifier", "agent-1234")
    def modify_code(...):
        ...
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Create autonomy monitor
            monitor = AutonomyMonitor(component, agent_id)
            
            # Extract autonomy level from kwargs if present
            if 'autonomy_level' in kwargs and isinstance(kwargs['autonomy_level'], int):
                monitor.set_autonomy_level(kwargs['autonomy_level'])
            
            # Add monitor to kwargs for use within the function
            kwargs['_monitor'] = monitor
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Log the operation
                debug_info = monitor.log_to_debug_info()
                log_debug_info(logger, debug_info)
                
                return result
                
            except Exception as e:
                # Record the error
                execution_time = (time.time() - monitor.start_time) * 1000
                debug_info = monitor.log_to_debug_info()
                debug_info.success = False
                debug_info.error = str(e)
                debug_info.details["error_type"] = type(e).__name__
                debug_info.details["traceback"] = traceback.format_exc()
                log_debug_info(logger, debug_info)
                raise
        
        return wrapper
    return decorator

def log_mcp_tool_execution(logger: logging.Logger):
    """
    Decorator specifically for MCP tool execution with enhanced monitoring.
    
    Usage:
    @log_mcp_tool_execution(logger)
    def execute_mcp_tool(...):
        ...
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.time()
            
            # Extract tool info with explicit type handling
            tool_id = "unknown-mcp-tool"
            if 'tool_id' in kwargs:
                if isinstance(kwargs['tool_id'], str):
                    tool_id = kwargs['tool_id']
                
            try:
                # Execute the MCP tool
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                # Create debug info with MCP-specific details
                debug_info = DebugInfo(
                    timestamp=datetime.now().isoformat(),
                    component="MCPToolExecution",
                    action=f"execute_{tool_id}",
                    details={
                        "tool_id": tool_id,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                        "result_preview": str(result)[:200] if result else None,
                    },
                    duration_ms=execution_time,
                    success=True,
                    # MCP-specific fields
                    mcp_tool_id=tool_id
                )
                log_debug_info(logger, debug_info)
                
                return result
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                debug_info = DebugInfo(
                    timestamp=datetime.now().isoformat(),
                    component="MCPToolExecution",
                    action=f"execute_{tool_id}",
                    details={
                        "tool_id": tool_id,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                        "error_type": type(e).__name__,
                        "traceback": traceback.format_exc()
                    },
                    duration_ms=execution_time,
                    success=False,
                    error=str(e),
                    mcp_tool_id=tool_id
                )
                log_debug_info(logger, debug_info)
                raise
        
        return wrapper
    return decorator
