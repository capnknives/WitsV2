# core/tool_registry.py
from typing import List, Dict, Any, Type, Optional
import logging
import time
from datetime import datetime

from tools.base_tool import BaseTool
from .debug_utils import DebugInfo, log_debug_info, log_execution_time, PerformanceMonitor

class ToolRegistry:
    """
    Registry for tools that can be used by agents in the WITS-NEXUS v2 system.
    
    This registry maintains a collection of BaseTool instances, indexed by their name,
    and provides methods to register, retrieve, and list available tools.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        
        # Set up logging
        self.logger = logging.getLogger('WITS.ToolRegistry')
        
        if config and 'debug' in config:
            self.debug_enabled = config['debug'].get('enabled', False)
            self.debug_config = config['debug'].get('components', {}).get('tools', {})
        else:
            self.debug_enabled = False
            self.debug_config = {}
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor("ToolRegistry")
        self.perf_monitor = PerformanceMonitor('ToolRegistry')
        
        self.logger.info("Tool Registry initialized")
    def register_tool(self, tool_instance: BaseTool):
        """
        Register a tool in the registry.
        
        Args:
            tool_instance: An instance of BaseTool or its subclass
            
        Raises:
            TypeError: If the object is not an instance of BaseTool
        """
        start_time = time.time()
        
        try:
            if not isinstance(tool_instance, BaseTool):
                raise TypeError("Object to register must be an instance of BaseTool or its subclass.")
            
            if tool_instance.name in self._tools:
                self.logger.warning(f"Tool '{tool_instance.name}' is already registered. Overwriting.")
            
            self._tools[tool_instance.name] = tool_instance
            
            # Log success with debug info
            if self.debug_enabled:
                execution_time = (time.time() - start_time) * 1000
                debug_info = DebugInfo(
                    timestamp=datetime.now().isoformat(),
                    component="ToolRegistry",
                    action="register_tool",
                    details={
                        "tool_name": tool_instance.name,
                        "tool_type": tool_instance.__class__.__name__,
                        "overwritten": is_overwritten,
                        "total_tools": len(self._tools)
                    },
                    duration_ms=execution_time,
                    success=True
                )
                log_debug_info(self.logger, debug_info)
            
            self.logger.info(f"Registered tool: {tool_instance.name}")
            
        except Exception as e:
            if self.debug_enabled:
                debug_info = DebugInfo(
                    timestamp=datetime.now().isoformat(),
                    component="ToolRegistry",
                    action="register_tool",
                    details={
                        "tool_name": getattr(tool_instance, 'name', 'unknown'),
                        "error_type": type(e).__name__
                    },
                    duration_ms=(time.time() - start_time) * 1000,
                    success=False,
                    error=str(e)
                )
                log_debug_info(self.logger, debug_info)
            raise
    
    @log_execution_time(logging.getLogger('WITS.ToolRegistry'))
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            tool_name: The name of the tool to retrieve
            
        Returns:
            BaseTool or None: The tool instance if found, None otherwise
        """
        start_time = time.time()
        
        tool = self._tools.get(tool_name)
        
        # Log attempt and result
        if self.debug_enabled:
            execution_time = (time.time() - start_time) * 1000
            debug_info = DebugInfo(
                timestamp=datetime.now().isoformat(),
                component="ToolRegistry",
                action="get_tool",
                details={
                    "requested_tool": tool_name,
                    "found": tool is not None,
                    "available_tools": list(self._tools.keys())
                },
                duration_ms=execution_time,
                success=tool is not None
            )
            log_debug_info(self.logger, debug_info)
        
        if not tool:
            self.logger.error(f"Tool '{tool_name}' not found.")
        else:
            self.logger.debug(f"Retrieved tool: {tool_name}")
        
        return tool

    def get_all_tools(self) -> List[BaseTool]:
        """
        Get all registered tools.
        
        Returns:
            List[BaseTool]: List of all registered tool instances
        """
        return list(self._tools.values())

    def get_all_tool_llm_schemas(self) -> List[Dict[str, Any]]:
        """
        Get JSON schemas for all registered tools, formatted for LLM prompting.
        
        Returns:
            List[Dict[str, Any]]: List of tool schema dictionaries
        """
        return [tool.get_llm_schema() for tool in self._tools.values()]

    def list_tools_for_llm(self) -> str:
        """
        Generate a formatted string summary of tools for the LLM prompt.
        
        Returns:
            str: A formatted string describing all available tools and their parameters
        """
        if not self._tools:
            return "No tools available."
        
        lines = ["Available Tools for use:"]
        for tool_name, tool_instance in self._tools.items():
            schema = tool_instance.get_llm_schema()  # Gets the full schema dict
            lines.append(f"\n- Tool Name: \"{schema['name']}\"")
            lines.append(f"  Description: {schema['description']}")
            
            # Format parameters for better readability in the prompt
            params_str_parts = []
            if schema['parameters']['properties']:
                for param_name, param_details in schema['parameters']['properties'].items():
                    param_type = param_details.get('type', 'any')
                    param_desc = param_details.get('description', '')
                    is_required = param_name in schema['parameters'].get('required', [])
                    req_str = " (required)" if is_required else ""
                    params_str_parts.append(f"    - '{param_name}' ({param_type}{req_str}): {param_desc}")
            else:
                params_str_parts.append("    (No arguments required)")

            lines.append(f"  Arguments:\n" + "\n".join(params_str_parts))
        
        return "\n".join(lines)
