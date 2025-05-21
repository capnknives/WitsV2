# core/autonomy/mcp_integration.py
"""
MCP (Model Context Protocol) Integration for WITS Nexus v2.
Initializes and configures the MCP system for dynamic tool creation.
I'm the gateway to creating AI-defined tools at runtime! \o/
"""

import logging
import json
import os
import asyncio
from typing import Dict, Any, Optional, List, Union, Callable

from core.tool_registry import ToolRegistry
from core.llm_interface import LLMInterface
from core.autonomy.mcp_tool_adapter import MCPToolManager, MCPToolDefinition
from core.autonomy.autonomy_enhancer import AutonomyEnhancer
from core.autonomy.tool_example_repository import ToolExampleRepository

class MCPIntegration:
    """
    Manages MCP (Model Context Protocol) integration with WITS Nexus.
    Provides high-level functions to initialize and use MCP capabilities.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        autonomy_enhancer: Optional[AutonomyEnhancer] = None,
        llm_interface: Optional[LLMInterface] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize MCP integration.
        
        Args:
            tool_registry: The system's tool registry
            autonomy_enhancer: Optional autonomy enhancer for learning
            llm_interface: Optional LLM interface for validation
            config: Configuration options
        """
        self.tool_registry = tool_registry
        self.autonomy_enhancer = autonomy_enhancer
        self.llm_interface = llm_interface
        self.config = config or {}
        
        # Initialize MCP tool manager
        self.mcp_manager = MCPToolManager(
            tool_registry=tool_registry,
            autonomy_enhancer=autonomy_enhancer
        )
        
        # Set up logging
        self.logger = logging.getLogger('WITS.MCPIntegration')
        
        # Directory for persisting MCP tool definitions
        self.mcp_tools_dir = self.config.get("mcp_tools_directory", "data/mcp_tools")
        os.makedirs(self.mcp_tools_dir, exist_ok=True)
        
        # Load saved tools if specified
        if self.config.get("auto_load_tools", True):
            asyncio.create_task(self._load_saved_tools())
    
    async def _load_saved_tools(self) -> None:
        """Load saved MCP tool definitions from disk."""
        try:
            if not os.path.exists(self.mcp_tools_dir):
                self.logger.info(f"MCP tools directory not found: {self.mcp_tools_dir}")
                return
                
            tool_files = [f for f in os.listdir(self.mcp_tools_dir) if f.endswith('.json')]
            loaded_count = 0
            
            for file_name in tool_files:
                try:
                    file_path = os.path.join(self.mcp_tools_dir, file_name)
                    with open(file_path, 'r') as f:
                        tool_def = json.load(f)
                    
                    # Convert to MCPToolDefinition
                    tool_definition = MCPToolDefinition(**tool_def)
                    
                    # Register the tool
                    result = await self.mcp_manager.register_mcp_tool(tool_definition)
                    
                    if result.success:
                        loaded_count += 1
                        self.logger.info(f"Loaded MCP tool: {tool_definition.name}")
                    else:
                        self.logger.warning(f"Failed to load MCP tool from {file_name}: {result.message}")
                        
                except Exception as e:
                    self.logger.error(f"Error loading MCP tool from {file_name}: {str(e)}")
            
            self.logger.info(f"Loaded {loaded_count} MCP tools from {self.mcp_tools_dir}")
            
        except Exception as e:
            self.logger.error(f"Error loading saved MCP tools: {str(e)}")
    
    async def _save_tool_definition(self, tool_definition: MCPToolDefinition) -> bool:
        """Save a tool definition to disk."""
        try:
            file_path = os.path.join(self.mcp_tools_dir, f"{tool_definition.name}.json")
            with open(file_path, 'w') as f:
                json.dump(tool_definition.dict(), f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Error saving MCP tool definition: {str(e)}")
            return False
    
    async def create_tool_from_description(
        self, 
        description: str,
        name_prefix: str = "mcp_"
    ) -> Dict[str, Any]:
        """
        Create a new tool based on a natural language description.
        Uses the LLM to generate a tool definition.
        
        Args:
            description: Natural language description of what the tool should do
            name_prefix: Prefix to add to the generated tool name
            
        Returns:
            Dict with result of tool creation
        """
        if not self.llm_interface:
            return {
                "success": False,
                "message": "LLM interface is required for tool creation from description"
            }
        
        try:
            # Generate a prompt for the LLM to create a tool definition
            prompt = self._generate_tool_creation_prompt(description, name_prefix)
            
            # Get a response from the LLM
            response = await self.llm_interface.acompletion(
                prompt=prompt,
                temperature=0.2,  # Low temperature for consistent results
                max_tokens=2000
            )
            
            if not response:
                return {
                    "success": False,
                    "message": "Failed to get LLM response for tool creation"
                }
            
            # Extract the JSON from the response
            try:
                # Find JSON content between triple backticks
                content = response.strip()
                if "```json" in content:
                    json_content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_content = content.split("```")[1].strip()
                else:
                    # Try to find the entire content as JSON
                    json_content = content
                
                tool_def = json.loads(json_content)
                
                # Ensure name has the prefix
                if "name" in tool_def and not tool_def["name"].startswith(name_prefix):
                    tool_def["name"] = f"{name_prefix}{tool_def['name']}"
                
                # Convert to MCPToolDefinition
                tool_definition = MCPToolDefinition(**tool_def)
                
                # Register the tool
                result = await self.mcp_manager.register_mcp_tool(tool_definition)
                
                if result.success:
                    # Save the tool definition to disk
                    await self._save_tool_definition(tool_definition)
                    
                    return {
                        "success": True,
                        "message": f"Tool '{tool_definition.name}' created and registered successfully",
                        "tool_name": tool_definition.name,
                        "schema": result.schema
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Failed to register tool: {result.message}"
                    }
                    
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Failed to parse LLM response as tool definition: {str(e)}",
                    "llm_response": response
                }
                
        except Exception as e:
            self.logger.error(f"Error creating tool from description: {str(e)}")
            return {
                "success": False,
                "message": f"Error creating tool from description: {str(e)}"
            }
    
    def _generate_tool_creation_prompt(self, description: str, name_prefix: str) -> str:
        """Generate a prompt for the LLM to create a tool definition."""
        return f"""
You are an expert AI tool creator. Your task is to create a new tool definition based on a description.

DESCRIPTION:
{description}

Create a JSON object for an MCP tool definition with the following fields:
- name: A short, descriptive name for the tool (should start with "{name_prefix}" and use snake_case)
- description: A detailed description of what the tool does
- parameters: A dictionary of parameters with name, type, and description
- handler_code: Python code that implements the tool's functionality (should be async)
- handler_type: "async" (default) or "sync"

The handler_code should:
1. Be complete and executable Python code
2. Follow best practices for error handling
3. Return a success/failure response with appropriate data
4. Use only standard libraries or libraries available in the system (requests, aiohttp, bs4, etc.)

Example tool definition format:
```json
{{
  "name": "{name_prefix}example_tool",
  "description": "A detailed description of what this tool does",
  "parameters": {{
    "param1": {{
      "type": "string",
      "description": "Description of param1"
    }},
    "param2": {{
      "type": "integer",
      "description": "Description of param2"
    }}
  }},
  "handler_code": "async def execute(self, args):\\n    try:\\n        # Implementation here\\n        return {{'success': True, 'result': 'some result'}}\\n    except Exception as e:\\n        return {{'success': False, 'error': str(e)}}",
  "handler_type": "async"
}}
```

Respond ONLY with the JSON definition, nothing else.
"""

    async def list_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Get a list of all registered MCP tools.
        
        Returns:
            List of dictionaries with tool details
        """
        tool_names = self.mcp_manager.get_all_mcp_tools()
        tools = []
        
        for name in tool_names:
            tool = self.tool_registry.get_tool(name)
            if tool:
                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.get_llm_schema()
                })
        
        return tools
    
    async def deregister_mcp_tool(self, tool_name: str) -> Dict[str, Any]:
        """
        Deregister an MCP tool.
        
        Args:
            tool_name: Name of the tool to deregister
            
        Returns:
            Dict with result of deregistration
        """
        try:
            # Check if the tool is an MCP tool
            if tool_name not in self.mcp_manager.get_all_mcp_tools():
                return {
                    "success": False,
                    "message": f"Tool '{tool_name}' is not an MCP tool"
                }
            
            # Deregister the tool
            result = await self.mcp_manager.deregister_mcp_tool(tool_name)
            
            if result:
                # Remove the tool definition from disk
                file_path = os.path.join(self.mcp_tools_dir, f"{tool_name}.json")
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                return {
                    "success": True,
                    "message": f"Tool '{tool_name}' deregistered successfully"
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to deregister tool '{tool_name}'"
                }
                
        except Exception as e:
            self.logger.error(f"Error deregistering MCP tool: {str(e)}")
            return {
                "success": False,
                "message": f"Error deregistering MCP tool: {str(e)}"
            }
