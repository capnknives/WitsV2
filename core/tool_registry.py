# core/tool_registry.py
# Welcome to our magical toolbox! Where dreams and functions come together! \\o/
"""Tool registry for managing and accessing tools. Think of it as a swiss army knife on steroids! =D"""
from typing import Dict, Any, Optional, List, Union  # Our trusty type-checking friends ^_^
import logging
import json
from datetime import datetime  # Time tracking, because timing is everything! >.>

from .debug_utils import DebugInfo, log_debug_info  # For when we need to leave breadcrumbs x.x
from tools.base_tool import BaseTool  # The parent of all tools! \o/

class ToolRegistry:
    """
    Welcome to our epic toolbox! Here's where the magic happens! =D
    A home for all our shiny tools, where they live in perfect harmony ^_^
    Just don't ask about that one time we lost the calculator... >.>
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], Any]] = None):
        """
        Time to set up our magical toolbox! Let's get organized! \\o/
        
        Args:
            config: Our settings and secrets (optional, but who doesn't love options? =P)
        """
        # Config can be fancy or simple - we don't judge! ^_^
        self.config = {} if config is None else config
        if not isinstance(config, dict) and config is not None:
            if hasattr(config, 'model_dump'):  # Ooh, a Pydantic model! Fancy! =P
                self.config = config.model_dump()
            elif hasattr(config, 'dict'):  # Old school but still cool \o/
                self.config = config.dict()
        
        # Our tools need a cozy home! Don't let them get lost x.x
        self.tools: Dict[str, BaseTool] = {}  # Empty toolbox... for now! =D
        self.logger = logging.getLogger('WITS.ToolRegistry')  # For when things go boom >.>
    
    def register_tool(self, tool_instance: BaseTool) -> None:
        """
        Welcome a new tool to our happy family! \\o/
        
        Args:
            tool_instance: The new tool that wants to join our party! ^_^
            
        Raises:
            TypeError: When someone tries to sneak in a non-tool (Nice try! x.x)
        """
        if not isinstance(tool_instance, BaseTool):
            raise TypeError(f"Hey now! That's not a proper tool! We got {type(tool_instance)} instead x.x")
        
        try:
            if tool_instance.name in self.tools:
                self.logger.warning(f"Whoops! Tool '{tool_instance.name}' is already here! Time for an upgrade! =P")
            
            self.tools[tool_instance.name] = tool_instance  # Welcome to the family! ^_^
            
            # Time to document this historic moment! O.o
            debug_info = DebugInfo(
                timestamp=datetime.now().isoformat(),
                component="ToolRegistry",
                action="register_tool",
                details={
                    "tool_name": tool_instance.name,
                    "total_tools": len(self.tools)  # Look at our growing collection! \\o/
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
    
    def deregister_tool(self, tool_name: str) -> bool:
        """
        Say goodbye to a tool! Sometimes tools need to retire too! \\o/
        
        Args:
            tool_name: The name of the tool to deregister
            
        Returns:
            bool: True if the tool was deregistered, False if not found
            
        Raises:
            ValueError: When someone tries to deregister a tool that doesn't exist
        """
        try:
            if tool_name not in self.tools:
                self.logger.warning(f"Tool '{tool_name}' not found for deregistration.")
                return False
            
            # Time to say farewell! It's not goodbye, it's see you later! ^_^
            del self.tools[tool_name]
            
            # Document this bittersweet moment! O.o
            debug_info = DebugInfo(
                timestamp=datetime.now().isoformat(),
                component="ToolRegistry",
                action="deregister_tool",
                details={
                    "tool_name": tool_name,
                    "total_tools": len(self.tools)  # Our collection shrinks a bit >.>
                },
                duration_ms=0,
                success=True
            )
            log_debug_info(self.logger, debug_info)
            
            self.logger.info(f"Tool '{tool_name}' deregistered successfully.")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deregistering tool '{tool_name}': {e}")
            raise
    
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
