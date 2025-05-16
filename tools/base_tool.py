# tools/base_tool.py
from abc import ABC, abstractmethod
import logging
import time
import traceback
from datetime import datetime
from typing import Any, ClassVar, Dict, Type, Optional
from pydantic import BaseModel, create_model

from core.debug_utils import DebugInfo, log_debug_info

class BaseTool(ABC):
    """
    Base class for all tools in the WITS-NEXUS v2 system.
    
    Tools provide specific capabilities to agents, such as web search,
    file operations, or calculations. Each tool is defined by:
    
    1. A name (for lookup and serialization)
    2. A description (for the LLM to understand when to use it)
    3. An arguments schema (Pydantic model defining expected inputs)
    4. An async execute method that performs the actual functionality
    
    Tools can be called by the orchestrator agent when it determines
    a specific capability is needed to achieve a goal.
    """
    
    name: ClassVar[str]  # Name of the tool (must be unique in registry)
    description: ClassVar[str]  # Description of what the tool does
    args_schema: ClassVar[Type[BaseModel]]  # Pydantic schema for arguments
    
    @abstractmethod
    async def execute(self, args: BaseModel) -> Any:
        """
        Execute the tool's functionality with the given arguments.
        
        Args:
            args: An instance of the tool's args_schema Pydantic model
            
        Returns:
            Any: The result of the tool execution (often a Pydantic response model)
        """
        pass
    
    def get_llm_schema(self) -> Dict[str, Any]:
        """
        Get the tool's schema in a format suitable for LLM prompts.
        
        Returns:
            Dict[str, Any]: JSON schema representation of the tool
        """
        # Create a schema with name, description, and parameters
        schema = {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_schema.model_json_schema()
        }
        
        return schema
    
    def __str__(self) -> str:
        """String representation of the tool."""
        return f"{self.name}: {self.description}"

class ToolException(Exception):
    """Exception raised when a tool encounters an error during execution."""
    pass
    
class BaseToolWithDebug(BaseTool):
    """Extended base tool with debug capabilities."""
    
    def __init__(self, config=None):
        """Initialize with logger and debug configuration."""
        # Note: ABC doesn't have __init__ to call super() for
        self.logger = logging.getLogger(f'WITS.Tools.{self.name}')
        
        # Set debug configuration
        self.debug_enabled = False
        self.debug_config = None
        
        if config and hasattr(config, 'debug'):
            self.debug_enabled = config.debug.enabled
            if hasattr(config.debug, 'components') and hasattr(config.debug.components, 'tools'):
                self.debug_config = config.debug.components.tools
        
    async def execute_with_debug(self, args: BaseModel) -> Any:
        """Execute the tool with debug tracking."""
        start_time = time.time()
        
        try:
            self.logger.debug(f"Executing tool with args: {args.model_dump_json()}")
            
            # Execute the actual tool functionality
            result = await self.execute(args)
            
            execution_time = (time.time() - start_time) * 1000  # ms
            
            # Log success with debug info
            if self.debug_enabled:
                # Create debug info for successful execution
                debug_info = DebugInfo(
                    timestamp=datetime.now().isoformat(),
                    component=f"Tool.{self.name}",
                    action="execute",
                    details={
                        "args": args.model_dump(),
                        "result_type": type(result).__name__
                    },
                    duration_ms=execution_time,
                    success=True
                )
                log_debug_info(self.logger, debug_info)
                
                # Detailed logging of args and results if enabled
                if self.debug_config and self.debug_config.get('log_args', False):
                    self.logger.debug(f"Tool args: {args.model_dump_json()}")
                if self.debug_config and self.debug_config.get('log_results', False):
                    result_str = str(result)
                    if len(result_str) > 500:
                        result_str = result_str[:497] + "..."
                    self.logger.debug(f"Tool result: {result_str}")
                
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"Tool execution error: {str(e)}"
            self.logger.error(error_msg)
            
            # Log failure with debug info
            if self.debug_enabled:
                debug_info = DebugInfo(
                    timestamp=datetime.now().isoformat(),
                    component=f"Tool.{self.name}",
                    action="execute_failed",
                    details={
                        "args": args.model_dump(),
                        "error_type": type(e).__name__,
                        "traceback": traceback.format_exc()
                    },
                    duration_ms=execution_time,
                    success=False,
                    error=str(e)
                )
                log_debug_info(self.logger, debug_info)
            
            # Re-raise as ToolException
            raise ToolException(error_msg) from e
    
    def __init__(self, tool_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize a ToolException.
        
        Args:
            tool_name: Name of the tool that raised the exception
            message: Error message
            details: Optional additional details about the error
        """
        self.tool_name = tool_name
        self.details = details or {}
        super().__init__(f"[{tool_name}] {message}")
