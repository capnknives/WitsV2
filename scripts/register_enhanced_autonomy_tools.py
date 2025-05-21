# scripts/register_enhanced_autonomy_tools.py
"""
Tool Registration Script for Enhanced AI Autonomy Tools.
Registers all the tools needed for enhanced AI autonomy.
I'm the ultimate toolbox organizer! \o/
"""

import logging
import asyncio
import sys
import os
from typing import Dict, Any, List

# Add the project root to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.tool_registry import ToolRegistry
from core.autonomy import (
    AutonomyEnhancer, 
    ToolExampleRepository,
    ToolSimulator,
    MCPToolManager
)
from core.autonomy.mcp_integration import MCPIntegration
from core.llm_interface import LLMInterface

# Import all tools
from tools.enhanced_file_tools import (
    EnhancedFileSearchTool,
    EnhancedReadFileTool,
    EnhancedWriteFileTool,
    EnhancedDeleteFileTool
)

from tools.code_modification_tool import (
    CodeReadTool,
    CodeModificationTool,
    CodeAnalysisTool
)

from tools.agent_management_tool import (
    AgentCreationTool,
    AgentManagementTool,
    AgentQueryTool,
    AgentTaskTool
)

from tools.mcp_tools import (
    CreateMCPToolTool,
    ListMCPToolsTool,
    RemoveMCPToolTool,
    CreateMCPToolFromDefinitionTool
)

async def register_tools(
    tool_registry: ToolRegistry,
    llm_interface: LLMInterface,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Register all tools for enhanced AI autonomy.
    
    Args:
        tool_registry: The system's tool registry
        llm_interface: LLM interface for AI interactions
        config: Configuration options
        
    Returns:
        Dict with registration results and component instances
    """
    logger = logging.getLogger('WITS.EnhancedAutonomySetup')
    logger.info("Setting up enhanced AI autonomy tools")
    
    # Initialize components
    tool_examples = ToolExampleRepository(config.get("tool_examples", {}))
    tool_simulator = ToolSimulator()
    autonomy_enhancer = AutonomyEnhancer(
        example_repository=tool_examples,
        simulator=tool_simulator,
        config=config.get("autonomy", {})
    )
    
    # Initialize MCP integration
    mcp_integration = MCPIntegration(
        tool_registry=tool_registry,
        autonomy_enhancer=autonomy_enhancer,
        llm_interface=llm_interface,
        config=config.get("mcp", {})
    )
    
    # Register enhanced file tools
    file_tools_config = config.get("enhanced_file_tools", {})
    tools_registered = []
    
    try:
        # Enhanced file tools
        enhanced_file_search = EnhancedFileSearchTool(file_tools_config)
        tool_registry.register_tool(enhanced_file_search)
        tools_registered.append(enhanced_file_search.name)
        
        enhanced_read_file = EnhancedReadFileTool(file_tools_config)
        tool_registry.register_tool(enhanced_read_file)
        tools_registered.append(enhanced_read_file.name)
        
        enhanced_write_file = EnhancedWriteFileTool(file_tools_config)
        tool_registry.register_tool(enhanced_write_file)
        tools_registered.append(enhanced_write_file.name)
        
        enhanced_delete_file = EnhancedDeleteFileTool(file_tools_config)
        tool_registry.register_tool(enhanced_delete_file)
        tools_registered.append(enhanced_delete_file.name)
        
        # Code modification tools
        code_tools_config = config.get("code_modification_tools", {})
        
        code_read = CodeReadTool(code_tools_config)
        tool_registry.register_tool(code_read)
        tools_registered.append(code_read.name)
        
        code_modification = CodeModificationTool(code_tools_config)
        tool_registry.register_tool(code_modification)
        tools_registered.append(code_modification.name)
        
        code_analysis = CodeAnalysisTool(code_tools_config)
        tool_registry.register_tool(code_analysis)
        tools_registered.append(code_analysis.name)
        
        # Agent management tools - these need AgentFactory and AgentManager to be injected
        agent_tools_config = config.get("agent_management_tools", {})
        
        agent_creation = AgentCreationTool(agent_tools_config)
        tool_registry.register_tool(agent_creation)
        tools_registered.append(agent_creation.name)
        
        agent_management = AgentManagementTool(agent_tools_config)
        tool_registry.register_tool(agent_management)
        tools_registered.append(agent_management.name)
        
        agent_query = AgentQueryTool(agent_tools_config)
        tool_registry.register_tool(agent_query)
        tools_registered.append(agent_query.name)
        
        agent_task = AgentTaskTool(agent_tools_config)
        tool_registry.register_tool(agent_task)
        tools_registered.append(agent_task.name)
        
        # MCP tools
        mcp_tools_config = config.get("mcp_tools", {})
        
        create_mcp_tool = CreateMCPToolTool(mcp_tools_config)
        create_mcp_tool.set_mcp_integration(mcp_integration)
        tool_registry.register_tool(create_mcp_tool)
        tools_registered.append(create_mcp_tool.name)
        
        list_mcp_tools = ListMCPToolsTool(mcp_tools_config)
        list_mcp_tools.set_mcp_integration(mcp_integration)
        tool_registry.register_tool(list_mcp_tools)
        tools_registered.append(list_mcp_tools.name)
        
        remove_mcp_tool = RemoveMCPToolTool(mcp_tools_config)
        remove_mcp_tool.set_mcp_integration(mcp_integration)
        tool_registry.register_tool(remove_mcp_tool)
        tools_registered.append(remove_mcp_tool.name)
        
        create_mcp_tool_from_def = CreateMCPToolFromDefinitionTool(mcp_tools_config)
        create_mcp_tool_from_def.set_mcp_integration(mcp_integration)
        tool_registry.register_tool(create_mcp_tool_from_def)
        tools_registered.append(create_mcp_tool_from_def.name)
        
        logger.info(f"Successfully registered {len(tools_registered)} enhanced autonomy tools")
        
    except Exception as e:
        logger.error(f"Error registering enhanced autonomy tools: {str(e)}")
        raise
    
    return {
        "tools_registered": tools_registered,
        "autonomy_enhancer": autonomy_enhancer,
        "tool_examples": tool_examples,
        "tool_simulator": tool_simulator,
        "mcp_integration": mcp_integration
    }

if __name__ == "__main__":
    # This script can be run directly to register tools in a standalone environment
    from core.config import load_config
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        config = load_config()
        tool_registry = ToolRegistry(config)
        llm_interface = LLMInterface(config)
        
        result = await register_tools(tool_registry, llm_interface, config)
        print(f"Registered tools: {result['tools_registered']}")
    
    asyncio.run(main())
