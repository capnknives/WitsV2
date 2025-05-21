# Welcome to the birthplace of all tools! This is where the magic begins! \\o/
from abc import ABC, abstractmethod  # Abstract classes, because we're fancy like that! ^_^
import logging
import time  # Time tracking (because we care about performance... sometimes x.x)
import traceback  # For when things go boom! >.>
from datetime import datetime
from typing import Any, ClassVar, Dict, Type, Optional, Generic, TypeVar # Our type-safety friends! =D
from pydantic import BaseModel, create_model, Field # Pydantic, our data validation bestie! \\o/

from core.debug_utils import DebugInfo, log_debug_info  # Debug powers, activate! O.o

# Define a TypeVar for the output of a tool
OutputType = TypeVar('OutputType', bound=BaseModel)

class ToolResponse(BaseModel, Generic[OutputType]):
    """Standardized response structure for tools."""
    status_code: int = Field(200, description="HTTP-like status code for the operation.")
    output: Optional[OutputType] = Field(None, description="The actual output data from the tool, conforming to a Pydantic model.")
    error_message: Optional[str] = Field(None, description="Error message if the operation failed.")

class BaseTool(ABC):
    """
    The awesome parent of all our tools! Let's build some cool stuff ^_^
    Every tool in our toolkit inherits from this super-cool base class! \\o/
    
    Think of this as the ultimate tool template:
    
    1. name: What we call it (gotta be unique or things break! x.x)
    2. description: So the LLM knows when to use it =D
    3. args_schema: All the stuff it needs to work (Pydantic keeps us safe \\o/)
    4. execute: Where the magic happens! *waves wand* ✨
    
    The orchestrator can grab any tool when it needs something done.
    Pretty neat, huh? =P
    
    Just remember:
    - Always give your tool a unique name (no copycats allowed! >.>)
    - Write clear descriptions (future you will thank you! ^_^)
    - Define your args properly (Pydantic is your friend! =D)
    - Make execute() do something awesome! \\o/
    """
    
    name: ClassVar[str]  # Our tool's special name (make it memorable! ^_^)
    description: ClassVar[str]  # What does it do? Make it clear! =D
    args_schema: ClassVar[Type[BaseModel]]  # The recipe for success! \\o/
    
    @abstractmethod
    async def execute(self, args: BaseModel) -> ToolResponse[BaseModel]: # Ensure this uses the new ToolResponse
        """
        Time to make the magic happen! This is where tools do their thing! \\o/
        
        Args:
            args: Everything the tool needs (checked by Pydantic for safety! ^_^)
            
        Returns:
            Any: The result of our magical operation! =D
            
        Remember:
        - Always validate your inputs (trust no one! >.>)
        - Handle errors gracefully (no explosions please! x.x)
        - Return something useful (undefined is not a function! O.o)
        """
        pass
    
    def get_llm_schema(self) -> Dict[str, Any]:
        """
        Show off our tool's capabilities to the LLM! Time to shine! \\o/
        
        Returns:
            Dict[str, Any]: Our tool's resume in JSON format! =D
            
        This is like a dating profile for our tool:
        - Here's my name ^_^
        - This is what I do \\o/
        - These are the things I need =P
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
    
    def __init__(self, message: str):
        """Initialize with error message."""
        self.message = message
        super().__init__()

class BaseToolWithDebug(BaseTool):
    """Extended base tool with debug capabilities."""
    
    def __init__(self, config=None):
        """Initialize with logger and debug configuration."""
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
