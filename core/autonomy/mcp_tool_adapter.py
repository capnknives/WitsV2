# core/autonomy/mcp_tool_adapter.py
"""
MCP (Model Context Protocol) Tool Adapter for WITS Nexus v2.
Enables dynamic tool creation and registration from LLM instructions.
Think of it as a factory that can create tools on-demand! \o/
"""

import inspect
import logging
import json
from typing import Dict, Any, Optional, Type, List, Callable, Union, Awaitable
from pydantic import BaseModel, Field, create_model

from tools.base_tool import BaseTool, ToolResponse
from .enhanced_tool_base import EnhancedTool
from core.tool_registry import ToolRegistry

class MCPToolDefinition(BaseModel):
    """Definition for a dynamically created tool via MCP."""
    name: str = Field(..., description="Unique name for the tool")
    description: str = Field(..., description="Detailed description of what the tool does")
    parameters: Dict[str, Dict[str, Any]] = Field(..., description="Parameters schema with name, type, and description")
    handler_code: str = Field(..., description="Python code for the tool's execution handler")
    handler_type: str = Field("async", description="Whether the handler is 'async' or 'sync'")
    
    class Config:
        """Configuration for the model."""
        extra = "allow"  # Allow additional fields for future extensibility

class MCPToolAdapter(EnhancedTool):
    """
    Dynamic tool created via MCP (Model Context Protocol).
    I can transform from simple instructions into a fully functional tool! =D
    """

    def __init__(
        self,
        tool_definition: MCPToolDefinition,
        tool_registry: Optional[ToolRegistry] = None,
        autonomy_enhancer = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a dynamic MCP tool with the provided definition.
        
        Args:
            tool_definition: Definition of the tool including name, description, parameters and handler
            tool_registry: Optional registry to register the tool with
            autonomy_enhancer: Optional autonomy enhancer component
            context: Additional execution context
        """
        # Create the args schema dynamically based on the definition parameters
        args_schema = self._create_args_schema(tool_definition)
        
        # Set class variables that BaseTool expects
        self.__class__.name = tool_definition.name
        self.__class__.description = tool_definition.description
        self.__class__.args_schema = args_schema
        
        # Compile the handler code
        self.handler_code = tool_definition.handler_code
        self.handler_type = tool_definition.handler_type
        try:
            # Create an isolated namespace for the handler
            namespace = {}
            # Add necessary imports that the handler can use
            exec("import os, sys, json, logging, time, asyncio, datetime", namespace)
            # Execute the handler code in this namespace
            exec(self.handler_code, namespace)
            
            # Extract the handler function from the namespace
            if "handler" not in namespace:
                raise ValueError("Handler code must define a 'handler' function")
            
            self.handler = namespace["handler"]
            
        except Exception as e:
            raise ValueError(f"Failed to compile handler code: {str(e)}")
            
        # Initialize the EnhancedTool parent
        super().__init__(autonomy_enhancer, context)
        
        # Register with the provided registry if specified
        if tool_registry:
            tool_registry.register_tool(self)
            
        self.logger = logging.getLogger(f"WITS.MCPTools.{self.name}")
        self.logger.info(f"MCP Tool '{self.name}' created successfully!")
    
    def _create_args_schema(self, tool_definition: MCPToolDefinition) -> Type[BaseModel]:
        """
        Dynamically create a Pydantic model for the tool arguments.
        
        Args:
            tool_definition: The tool definition with parameters
            
        Returns:
            Type[BaseModel]: A dynamically created Pydantic model class
        """
        # Field definitions for the model
        fields = {}
        
        for param_name, param_info in tool_definition.parameters.items():
            # Extract parameter information
            param_type = self._get_type_from_string(param_info.get("type", "str"))
            description = param_info.get("description", "")
            required = param_info.get("required", True)
            default = param_info.get("default", ... if required else None)
            
            # Create the field
            fields[param_name] = (param_type, Field(default, description=description))
        
        # Create and return the Pydantic model class
        model_name = f"{tool_definition.name.title().replace('_', '')}Args"
        return create_model(model_name, **fields, __module__="mcp_dynamic_models")
    
    def _get_type_from_string(self, type_str: str) -> Type:
        """
        Convert a string type representation to an actual type.
        
        Args:
            type_str: String representation of the type
            
        Returns:
            Type: The actual type
        """
        type_mapping = {
            "str": str,
            "string": str,
            "int": int,
            "integer": int,
            "float": float,
            "bool": bool,
            "boolean": bool,
            "dict": dict,
            "list": list,
            "any": Any,
        }
        
        return type_mapping.get(type_str.lower(), str)
    
    async def execute(self, args: BaseModel) -> ToolResponse:
        """
        Execute the tool with the provided arguments.
        This runs the dynamically compiled handler code! =D
        
        Args:
            args: Arguments to pass to the handler
            
        Returns:
            ToolResponse: The result of the tool execution
        """
        self.logger.debug(f"Executing MCP tool '{self.name}' with args: {args.model_dump_json()}")
        
        try:
            # Convert args to dict for easier handling
            args_dict = args.model_dump()
            
            # Execute the handler based on its type
            if self.handler_type == "async":
                # If handler is async, await it
                if inspect.iscoroutinefunction(self.handler):
                    result = await self.handler(args_dict, self.execution_context)
                else:
                    # Handler was defined as async but isn't a coroutine function
                    self.logger.warning("Handler was defined as async but isn't a coroutine function")
                    result = self.handler(args_dict, self.execution_context)
            else:
                # If handler is sync, just call it
                result = self.handler(args_dict, self.execution_context)
                
            # Create a standardized response
            if isinstance(result, dict):
                # If the handler returned a dict, use it for the response
                error = result.get("error")
                if error:
                    return ToolResponse(
                        status_code=result.get("status_code", 500),
                        error_message=error,
                        output=result.get("output")
                    )
                else:
                    return ToolResponse(
                        status_code=result.get("status_code", 200),
                        output=result.get("output", result)  # Use 'output' field or the whole result
                    )
            else:
                # For non-dict returns, just wrap in a successful response
                return ToolResponse(
                    status_code=200,
                    output=result
                )
                
        except Exception as e:
            self.logger.error(f"Error executing MCP tool '{self.name}': {str(e)}", exc_info=True)
            return ToolResponse(
                status_code=500,
                error_message=f"Tool execution error: {str(e)}",
                output=None
            )

class MCPToolRegistrationResponse(BaseModel):
    """Response model for MCP tool registration."""
    tool_name: str = Field(..., description="Name of the registered tool")
    success: bool = Field(..., description="Whether the registration was successful")
    message: str = Field(..., description="Details about the registration")
    schema: Optional[Dict[str, Any]] = Field(None, description="Schema of the registered tool")

class MCPToolManager:
    """
    Manager for MCP tools - handles creating, registering, and managing dynamic tools.
    Like a talent agent for tools - brings them into the spotlight! \o/
    """
    
    def __init__(self, tool_registry: ToolRegistry, autonomy_enhancer=None):
        """
        Initialize the MCP Tool Manager.
        
        Args:
            tool_registry: Registry where tools will be registered
            autonomy_enhancer: Optional autonomy enhancer for tools
        """
        self.tool_registry = tool_registry
        self.autonomy_enhancer = autonomy_enhancer
        self.logger = logging.getLogger("WITS.Autonomy.MCPToolManager")
    
    async def register_mcp_tool(self, tool_definition: Union[Dict[str, Any], MCPToolDefinition]) -> MCPToolRegistrationResponse:
        """
        Register a new MCP tool from a definition.
        
        Args:
            tool_definition: Definition of the tool to register
            
        Returns:
            MCPToolRegistrationResponse: Result of the registration
        """
        try:
            # Convert dict to MCPToolDefinition if needed
            if isinstance(tool_definition, dict):
                tool_definition = MCPToolDefinition(**tool_definition)
                
            # Check if tool with this name already exists
            existing_tool = self.tool_registry.get_tool(tool_definition.name)
            if existing_tool:
                return MCPToolRegistrationResponse(
                    tool_name=tool_definition.name,
                    success=False,
                    message=f"A tool with name '{tool_definition.name}' already exists",
                    schema=None
                )
                
            # Create and register the tool
            tool = MCPToolAdapter(
                tool_definition=tool_definition,
                tool_registry=self.tool_registry,
                autonomy_enhancer=self.autonomy_enhancer
            )
            
            # Get the tool schema for reference
            tool_schema = {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    field_name: {
                        "type": str(field.annotation).replace("<class '", "").replace("'>", ""),
                        "description": field.field_info.description,
                        "required": field.field_info.default is ...,
                    }
                    for field_name, field in tool.args_schema.__annotations__.items()
                },
            }
            
            return MCPToolRegistrationResponse(
                tool_name=tool_definition.name,
                success=True,
                message=f"Tool '{tool_definition.name}' registered successfully",
                schema=tool_schema
            )
            
        except Exception as e:
            self.logger.error(f"Failed to register MCP tool: {str(e)}", exc_info=True)
            return MCPToolRegistrationResponse(
                tool_name=getattr(tool_definition, "name", "unknown"),
                success=False,
                message=f"Failed to register tool: {str(e)}",
                schema=None
            )
    
    def get_all_mcp_tools(self) -> List[str]:
        """
        Get a list of all registered MCP tools.
        
        Returns:
            List[str]: Names of all registered MCP tools
        """
        return [
            tool.name
            for tool in self.tool_registry.get_all_tools()
            if isinstance(tool, MCPToolAdapter)
        ]
    
    async def deregister_mcp_tool(self, tool_name: str) -> bool:
        """
        Deregister an MCP tool.
        
        Args:
            tool_name: Name of the tool to deregister
            
        Returns:
            bool: Whether the deregistration was successful
        """
        # This requires adding a deregister_tool method to ToolRegistry
        if hasattr(self.tool_registry, "deregister_tool"):
            try:
                return await self.tool_registry.deregister_tool(tool_name)
            except Exception as e:
                self.logger.error(f"Failed to deregister MCP tool '{tool_name}': {str(e)}")
                return False
        else:
            self.logger.warning("Tool registry does not support deregistration")
            return False
