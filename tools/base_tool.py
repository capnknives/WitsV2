# tools/base_tool.py
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Type, Optional
from pydantic import BaseModel, create_model

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
