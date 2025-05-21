# tools/mcp_tools.py
"""
MCP (Model Context Protocol) Tools for WITS Nexus v2.
Tools for creating and managing MCP-based tools dynamically.
Behold the power of tools that create tools! It's tool-inception! \o/
"""

import logging
from typing import ClassVar, Type, Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

from tools.base_tool import ToolResponse
from core.autonomy.enhanced_tool_base import EnhancedTool
from core.autonomy.mcp_integration import MCPIntegration
from core.autonomy.mcp_tool_adapter import MCPToolDefinition

# --- Create MCP Tool from Description ---

class CreateMCPToolArgs(BaseModel):
    """Arguments for creating an MCP tool from a description."""
    description: str = Field(..., description="Natural language description of what the tool should do")
    name_prefix: str = Field("mcp_", description="Prefix to add to the generated tool name")

class CreateMCPToolResponse(BaseModel):
    """Response from MCP tool creation."""
    success: bool = Field(..., description="Whether the tool was created successfully")
    tool_name: Optional[str] = Field(None, description="Name of the created tool")
    message: str = Field(..., description="Creation status message")
    schema: Optional[Dict[str, Any]] = Field(None, description="Schema of the created tool")
    error: Optional[str] = Field(None, description="Error message if creation failed")

class CreateMCPToolTool(EnhancedTool):
    """
    Create new MCP tools from natural language descriptions.
    I'm the tool that creates other tools! Tool-ception! \o/
    """
    
    name: ClassVar[str] = "create_mcp_tool"
    description: ClassVar[str] = "Create a new MCP tool from a natural language description."
    args_schema: ClassVar[Type[BaseModel]] = CreateMCPToolArgs
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        super().__init__()
        self.config = config or {}
        self.logger = logging.getLogger("WITS.CreateMCPToolTool")
        
        # The MCP integration should be injected by the application
        self.mcp_integration = None
    
    def set_mcp_integration(self, mcp_integration: MCPIntegration):
        """Set the MCP integration."""
        self.mcp_integration = mcp_integration
    
    async def _execute_impl(self, args: CreateMCPToolArgs) -> ToolResponse[CreateMCPToolResponse]:
        """Implementation of MCP tool creation."""
        try:
            if not self.mcp_integration:
                return ToolResponse[CreateMCPToolResponse](
                    status_code=500,
                    error_message="MCP integration is not available",
                    output=CreateMCPToolResponse(
                        success=False,
                        message="MCP integration is not available",
                        error="MCP integration is not available"
                    )
                )
            
            # Create the tool using the MCP integration
            result = await self.mcp_integration.create_tool_from_description(
                description=args.description,
                name_prefix=args.name_prefix
            )
            
            if not result["success"]:
                return ToolResponse[CreateMCPToolResponse](
                    status_code=400,
                    error_message=result["message"],
                    output=CreateMCPToolResponse(
                        success=False,
                        message=result["message"],
                        error=result.get("message")
                    )
                )
            
            return ToolResponse[CreateMCPToolResponse](
                status_code=200,
                output=CreateMCPToolResponse(
                    success=True,
                    tool_name=result.get("tool_name"),
                    message=result["message"],
                    schema=result.get("schema")
                )
            )
            
        except Exception as e:
            self.logger.error(f"Error creating MCP tool: {str(e)}")
            return ToolResponse[CreateMCPToolResponse](
                status_code=500,
                error_message=f"Error creating MCP tool: {str(e)}",
                output=CreateMCPToolResponse(
                    success=False,
                    message=f"Error creating MCP tool: {str(e)}",
                    error=f"Error creating MCP tool: {str(e)}"
                )
            )

# --- List MCP Tools ---

class ListMCPToolsArgs(BaseModel):
    """Arguments for listing MCP tools."""
    pass  # No arguments needed

class ListMCPToolsResponse(BaseModel):
    """Response from listing MCP tools."""
    success: bool = Field(..., description="Whether the listing was successful")
    tools: List[Dict[str, Any]] = Field(default_factory=list, description="List of MCP tools with details")
    count: int = Field(0, description="Number of MCP tools")
    message: str = Field(..., description="Status message")
    error: Optional[str] = Field(None, description="Error message if listing failed")

class ListMCPToolsTool(EnhancedTool):
    """
    List all registered MCP tools.
    I'm like a museum tour guide for your dynamic tools! \o/
    """
    
    name: ClassVar[str] = "list_mcp_tools"
    description: ClassVar[str] = "List all registered MCP tools."
    args_schema: ClassVar[Type[BaseModel]] = ListMCPToolsArgs
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        super().__init__()
        self.config = config or {}
        self.logger = logging.getLogger("WITS.ListMCPToolsTool")
        
        # The MCP integration should be injected by the application
        self.mcp_integration = None
    
    def set_mcp_integration(self, mcp_integration: MCPIntegration):
        """Set the MCP integration."""
        self.mcp_integration = mcp_integration
    
    async def _execute_impl(self, args: ListMCPToolsArgs) -> ToolResponse[ListMCPToolsResponse]:
        """Implementation of MCP tool listing."""
        try:
            if not self.mcp_integration:
                return ToolResponse[ListMCPToolsResponse](
                    status_code=500,
                    error_message="MCP integration is not available",
                    output=ListMCPToolsResponse(
                        success=False,
                        message="MCP integration is not available",
                        error="MCP integration is not available",
                        count=0
                    )
                )
            
            # List the tools using the MCP integration
            tools = await self.mcp_integration.list_mcp_tools()
            
            return ToolResponse[ListMCPToolsResponse](
                status_code=200,
                output=ListMCPToolsResponse(
                    success=True,
                    tools=tools,
                    count=len(tools),
                    message=f"Found {len(tools)} MCP tools"
                )
            )
            
        except Exception as e:
            self.logger.error(f"Error listing MCP tools: {str(e)}")
            return ToolResponse[ListMCPToolsResponse](
                status_code=500,
                error_message=f"Error listing MCP tools: {str(e)}",
                output=ListMCPToolsResponse(
                    success=False,
                    message=f"Error listing MCP tools: {str(e)}",
                    error=f"Error listing MCP tools: {str(e)}",
                    count=0
                )
            )

# --- Remove MCP Tool ---

class RemoveMCPToolArgs(BaseModel):
    """Arguments for removing an MCP tool."""
    tool_name: str = Field(..., description="Name of the MCP tool to remove")

class RemoveMCPToolResponse(BaseModel):
    """Response from removing an MCP tool."""
    success: bool = Field(..., description="Whether the removal was successful")
    tool_name: str = Field(..., description="Name of the removed tool")
    message: str = Field(..., description="Removal status message")
    error: Optional[str] = Field(None, description="Error message if removal failed")

class RemoveMCPToolTool(EnhancedTool):
    """
    Remove an MCP tool from the registry.
    I'm the cleanup crew for dynamic tools! \o/
    """
    
    name: ClassVar[str] = "remove_mcp_tool"
    description: ClassVar[str] = "Remove an MCP tool from the registry."
    args_schema: ClassVar[Type[BaseModel]] = RemoveMCPToolArgs
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        super().__init__()
        self.config = config or {}
        self.logger = logging.getLogger("WITS.RemoveMCPToolTool")
        
        # The MCP integration should be injected by the application
        self.mcp_integration = None
    
    def set_mcp_integration(self, mcp_integration: MCPIntegration):
        """Set the MCP integration."""
        self.mcp_integration = mcp_integration
    
    async def _execute_impl(self, args: RemoveMCPToolArgs) -> ToolResponse[RemoveMCPToolResponse]:
        """Implementation of MCP tool removal."""
        try:
            if not self.mcp_integration:
                return ToolResponse[RemoveMCPToolResponse](
                    status_code=500,
                    error_message="MCP integration is not available",
                    output=RemoveMCPToolResponse(
                        success=False,
                        tool_name=args.tool_name,
                        message="MCP integration is not available",
                        error="MCP integration is not available"
                    )
                )
            
            # Remove the tool using the MCP integration
            result = await self.mcp_integration.deregister_mcp_tool(args.tool_name)
            
            if not result["success"]:
                return ToolResponse[RemoveMCPToolResponse](
                    status_code=400,
                    error_message=result["message"],
                    output=RemoveMCPToolResponse(
                        success=False,
                        tool_name=args.tool_name,
                        message=result["message"],
                        error=result.get("message")
                    )
                )
            
            return ToolResponse[RemoveMCPToolResponse](
                status_code=200,
                output=RemoveMCPToolResponse(
                    success=True,
                    tool_name=args.tool_name,
                    message=result["message"]
                )
            )
            
        except Exception as e:
            self.logger.error(f"Error removing MCP tool: {str(e)}")
            return ToolResponse[RemoveMCPToolResponse](
                status_code=500,
                error_message=f"Error removing MCP tool: {str(e)}",
                output=RemoveMCPToolResponse(
                    success=False,
                    tool_name=args.tool_name,
                    message=f"Error removing MCP tool: {str(e)}",
                    error=f"Error removing MCP tool: {str(e)}"
                )
            )

# --- Create MCP Tool from JSON Definition ---

class CreateMCPToolFromDefinitionArgs(BaseModel):
    """Arguments for creating an MCP tool from a JSON definition."""
    name: str = Field(..., description="Unique name for the tool")
    description: str = Field(..., description="Detailed description of what the tool does")
    parameters: Dict[str, Dict[str, Any]] = Field(..., description="Parameters schema with name, type, and description")
    handler_code: str = Field(..., description="Python code for the tool's execution handler")
    handler_type: str = Field("async", description="Whether the handler is 'async' or 'sync'")

class CreateMCPToolFromDefinitionResponse(BaseModel):
    """Response from creating an MCP tool from a definition."""
    success: bool = Field(..., description="Whether the tool was created successfully")
    tool_name: str = Field(..., description="Name of the created tool")
    message: str = Field(..., description="Creation status message")
    schema: Optional[Dict[str, Any]] = Field(None, description="Schema of the created tool")
    error: Optional[str] = Field(None, description="Error message if creation failed")

class CreateMCPToolFromDefinitionTool(EnhancedTool):
    """
    Create a new MCP tool from a JSON definition.
    I'm the expert assembler of custom tools! \o/
    """
    
    name: ClassVar[str] = "create_mcp_tool_from_definition"
    description: ClassVar[str] = "Create a new MCP tool from a JSON definition."
    args_schema: ClassVar[Type[BaseModel]] = CreateMCPToolFromDefinitionArgs
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        super().__init__()
        self.config = config or {}
        self.logger = logging.getLogger("WITS.CreateMCPToolFromDefinitionTool")
        
        # The MCP integration should be injected by the application
        self.mcp_integration = None
    
    def set_mcp_integration(self, mcp_integration: MCPIntegration):
        """Set the MCP integration."""
        self.mcp_integration = mcp_integration
    
    async def _execute_impl(self, args: CreateMCPToolFromDefinitionArgs) -> ToolResponse[CreateMCPToolFromDefinitionResponse]:
        """Implementation of MCP tool creation from definition."""
        try:
            if not self.mcp_integration:
                return ToolResponse[CreateMCPToolFromDefinitionResponse](
                    status_code=500,
                    error_message="MCP integration is not available",
                    output=CreateMCPToolFromDefinitionResponse(
                        success=False,
                        tool_name=args.name,
                        message="MCP integration is not available",
                        error="MCP integration is not available"
                    )
                )
            
            # Create the tool using the MCP integration
            tool_definition = MCPToolDefinition(
                name=args.name,
                description=args.description,
                parameters=args.parameters,
                handler_code=args.handler_code,
                handler_type=args.handler_type
            )
            
            result = await self.mcp_integration.mcp_manager.register_mcp_tool(tool_definition)
            
            if not result.success:
                return ToolResponse[CreateMCPToolFromDefinitionResponse](
                    status_code=400,
                    error_message=result.message,
                    output=CreateMCPToolFromDefinitionResponse(
                        success=False,
                        tool_name=args.name,
                        message=result.message,
                        error=result.message
                    )
                )
            
            # Save the tool definition if registration was successful
            await self.mcp_integration._save_tool_definition(tool_definition)
            
            return ToolResponse[CreateMCPToolFromDefinitionResponse](
                status_code=200,
                output=CreateMCPToolFromDefinitionResponse(
                    success=True,
                    tool_name=args.name,
                    message=f"Tool '{args.name}' created successfully",
                    schema=result.schema
                )
            )
            
        except Exception as e:
            self.logger.error(f"Error creating MCP tool from definition: {str(e)}")
            return ToolResponse[CreateMCPToolFromDefinitionResponse](
                status_code=500,
                error_message=f"Error creating MCP tool from definition: {str(e)}",
                output=CreateMCPToolFromDefinitionResponse(
                    success=False,
                    tool_name=args.name,
                    message=f"Error creating MCP tool from definition: {str(e)}",
                    error=f"Error creating MCP tool from definition: {str(e)}"
                )
            )
