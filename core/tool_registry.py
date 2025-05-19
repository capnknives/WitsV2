# core/tool_registry.py
"""Tool registry for managing and accessing tools."""
from typing import Dict, Any, Optional, List, Union
import logging
import json
from datetime import datetime

from .debug_utils import DebugInfo, log_debug_info
from tools.base_tool import BaseTool

class ToolRegistry:
    """Registry for managing and using tools."""
    
    def __init__(self, config: Optional[Union[Dict[str, Any], Any]] = None):
        """Initialize the tool registry with config."""
        self.config = {} if config is None else config
        if not isinstance(config, dict) and config is not None:
            if hasattr(config, 'model_dump'):
                self.config = config.model_dump()
            elif hasattr(config, 'dict'):
                self.config = config.dict()
        
        # Initialize core components
        self.tools: Dict[str, BaseTool] = {}
        self.logger = logging.getLogger('WITS.ToolRegistry')
    
    def register_tool(self, tool_instance: BaseTool) -> None:
        """Register a tool with the registry."""
        if not isinstance(tool_instance, BaseTool):
            raise TypeError(f"Tool must be an instance of BaseTool, got {type(tool_instance)}")
        
        try:
            if tool_instance.name in self.tools:
                self.logger.warning(f"Tool '{tool_instance.name}' already registered, overwriting.")
            
            self.tools[tool_instance.name] = tool_instance
            
            # Log registration event
            debug_info = DebugInfo(
                timestamp=datetime.now().isoformat(),
                component="ToolRegistry",
                action="register_tool",
                details={
                    "tool_name": tool_instance.name,
                    "total_tools": len(self.tools)
                },
                duration_ms=0,
                success=True
            )
            log_debug_info(self.logger, debug_info)
            
            self.logger.info(f"Tool '{tool_instance.name}' registered successfully.")
            
        except Exception as e:
            self.logger.error(f"Error registering tool '{getattr(tool_instance, 'name', 'UNKNOWN')}' : {e}")
            raise
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        try:
            tool = self.tools.get(tool_name)
            if not tool:
                self.logger.warning(f"Tool '{tool_name}' not found in registry.")
                debug_info = DebugInfo(
                    timestamp=datetime.now().isoformat(),
                    component="ToolRegistry",
                    action="get_tool",
                    details={
                        "requested_tool": tool_name,
                        "available_tools": list(self.tools.keys())
                    },
                    duration_ms=0,
                    success=False,
                    error=f"Tool '{tool_name}' not found"
                )
                log_debug_info(self.logger, debug_info)
            return tool
        except Exception as e:
            self.logger.error(f"Error retrieving tool '{tool_name}': {e}")
            raise
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self.tools.values())
    
    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get tool schemas formatted for LLM consumption."""
        return [tool.get_llm_schema() for tool in self.tools.values()]
    
    def list_tools_for_llm(self) -> str:
        """Get a formatted string of tool information for LLM prompts."""
        if not self.tools:
            return "No tools available."
        
        tools_info = ["Available tools:"]
        for tool_name, tool_instance in self.tools.items():
            try:
                tool_schema = tool_instance.get_llm_schema()
                tools_info.append(f"\n{tool_name}:")
                tools_info.append(tool_instance.description or "No description available.")
                if tool_schema.get('parameters', {}).get('properties'):
                    tools_info.append("Parameters:")
                    for param_name, param_info in tool_schema['parameters']['properties'].items():
                        param_desc = param_info.get('description', 'No description')
                        param_type = param_info.get('type', 'any')
                        tools_info.append(f"- {param_name} ({param_type}): {param_desc}")
            except Exception as e:
                self.logger.error(f"Error formatting tool {tool_name} for LLM: {e}")
                tools_info.append(f"{tool_name}: Error getting tool information")
        
        return "\n".join(tools_info)
