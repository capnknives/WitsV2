# core/autonomy/enhanced_tool_base.py
"""
Enhanced tool base class that extends the existing BaseTool with learning capabilities.
This is our upgraded tool template that can learn and improve from each use! ^_^
"""
import logging
import time
import traceback
from typing import Any, Dict, Optional, Type
from datetime import datetime

from tools.base_tool import BaseTool, ToolResponse
from pydantic import BaseModel
from core.debug_utils import DebugInfo, log_debug_info

class EnhancedTool(BaseTool):
    """
    Enhanced base tool with learning capabilities.
    Like BaseTool, but with superpowers! \o/
    """
    
    def __init__(self, autonomy_enhancer=None, context: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced tool with learning capabilities.
        
        Args:
            autonomy_enhancer: Optional enhancer for tool autonomy
            context: Additional context for tool execution
        """
        super().__init__()
        self.autonomy_enhancer = autonomy_enhancer
        self.execution_context = context or {}
        self.logger = logging.getLogger(f'WITS.EnhancedTools.{self.name}')
    
    async def execute_with_learning(self, args: BaseModel) -> ToolResponse:
        """
        Execute the tool with learning from the experience.
        Teaches our AI to get better with each use! =D
        
        Args:
            args: Arguments for tool execution
            
        Returns:
            ToolResponse: The response from the tool
        """
        start_time = time.time()
        success = False
        result = None
        
        try:
            self.logger.debug(f"Executing enhanced tool '{self.name}' with args: {args.model_dump_json()}")
            
            # Execute the actual tool functionality
            result = await self.execute(args)
            
            # Mark as successful if we got here without exceptions
            success = True
            
            execution_time = (time.time() - start_time) * 1000  # ms
            
            # Log debug info
            debug_info = DebugInfo(
                timestamp=datetime.now().isoformat(),
                component=f"EnhancedTool.{self.name}",
                action="execute_with_learning",
                details={
                    "args": args.model_dump(),
                    "result_type": type(result).__name__,
                    "success": success
                },
                duration_ms=execution_time,
                success=True
            )
            log_debug_info(self.logger, debug_info)
            
            # Learn from this execution if we have an enhancer
            if self.autonomy_enhancer:
                await self.autonomy_enhancer.learn_from_execution(
                    tool_name=self.name,
                    args=args.model_dump(),
                    result=result.model_dump() if hasattr(result, "model_dump") else result,
                    success=success,
                    context=str(self.execution_context.get("task_description", "")),
                    execution_time=execution_time
                )
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"Enhanced tool execution error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Log failure
            debug_info = DebugInfo(
                timestamp=datetime.now().isoformat(),
                component=f"EnhancedTool.{self.name}",
                action="execute_with_learning_failed",
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
            
            # Learn from failed execution if we have an enhancer
            if self.autonomy_enhancer:
                await self.autonomy_enhancer.learn_from_execution(
                    tool_name=self.name,
                    args=args.model_dump(),
                    result=str(e),
                    success=False,
                    context=str(self.execution_context.get("task_description", "")),
                    execution_time=execution_time
                )
            
            # Create a ToolResponse with error
            if hasattr(self, "create_error_response"):
                return self.create_error_response(str(e))
            else:
                # Generic error response if the tool doesn't have a custom error response method
                return ToolResponse(
                    status_code=500,
                    error_message=str(e),
                    output=None
                )
    
    async def generate_example_prompt(self) -> str:
        """
        Generate a prompt with examples for this tool.
        Makes it easier for the AI to see how to use this tool! ^_^
        
        Returns:
            str: A prompt with examples for this tool
        """
        if self.autonomy_enhancer:
            return await self.autonomy_enhancer.generate_tool_examples_prompt(self.name)
        else:
            return f"Tool '{self.name}': {self.description}"
    
    def set_execution_context(self, context: Dict[str, Any]) -> None:
        """
        Set the execution context for the tool.
        Helps the tool understand what it's being used for! \o/
        
        Args:
            context: The context to set
        """
        self.execution_context = context
