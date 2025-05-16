# core/tool_registry.py
from typing import List, Dict, Any, Type, Optional
from tools.base_tool import BaseTool

class ToolRegistry:
    """
    Registry for tools that can be used by agents in the WITS-NEXUS v2 system.
    
    This registry maintains a collection of BaseTool instances, indexed by their name,
    and provides methods to register, retrieve, and list available tools.
    """
    
    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        print("[ToolRegistry] Initialized.")

    def register_tool(self, tool_instance: BaseTool):
        """
        Register a tool in the registry.
        
        Args:
            tool_instance: An instance of BaseTool or its subclass
            
        Raises:
            TypeError: If the object is not an instance of BaseTool
        """
        if not isinstance(tool_instance, BaseTool):
            raise TypeError("Object to register must be an instance of BaseTool or its subclass.")
        
        if tool_instance.name in self._tools:
            print(f"[ToolRegistry_WARN] Tool '{tool_instance.name}' is already registered. Overwriting.")
        
        self._tools[tool_instance.name] = tool_instance
        print(f"[ToolRegistry] Registered tool: {tool_instance.name}")

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            tool_name: The name of the tool to retrieve
            
        Returns:
            BaseTool or None: The tool instance if found, None otherwise
        """
        tool = self._tools.get(tool_name)
        if not tool:
            print(f"[ToolRegistry_ERROR] Tool '{tool_name}' not found.")
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
