# examples/mcp_tool_example.py
"""
Example of creating and using MCP tools in WITS Nexus v2.
Demonstrates how to create a dynamic tool and use it.
It's like watching a tool being born and grow up! \o/
"""

import asyncio
import logging
import json
import sys
import os
from typing import Dict, Any, List

# Add the project root to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.tool_registry import ToolRegistry
from core.llm_interface import LLMInterface
from core.autonomy.mcp_integration import MCPIntegration
from core.autonomy.mcp_tool_adapter import MCPToolDefinition
from core.autonomy import AutonomyEnhancer, ToolExampleRepository, ToolSimulator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("MCP_Example")

async def mcp_tool_example():
    """Example of creating and using MCP tools."""
    logger.info("Starting MCP tool example")
    
    # Create components needed for MCP
    tool_registry = ToolRegistry({})
    llm_interface = LLMInterface({
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": os.environ.get("OPENAI_API_KEY", "your-api-key")
        }
    })
    
    # Set up autonomy components
    tool_examples = ToolExampleRepository({})
    tool_simulator = ToolSimulator()
    autonomy_enhancer = AutonomyEnhancer(
        example_repository=tool_examples,
        simulator=tool_simulator
    )
    
    # Set up MCP integration
    mcp_integration = MCPIntegration(
        tool_registry=tool_registry,
        autonomy_enhancer=autonomy_enhancer,
        llm_interface=llm_interface,
        config={
            "mcp_tools_directory": "data/mcp_tools_example"
        }
    )
    
    # Example 1: Create a tool from a definition
    logger.info("Creating a weather tool from definition")
    
    weather_tool_def = MCPToolDefinition(
        name="mcp_weather",
        description="Get the current weather for a location",
        parameters={
            "location": {
                "type": "string",
                "description": "The location to get weather for (city name)"
            },
            "units": {
                "type": "string",
                "description": "Temperature units: 'celsius' or 'fahrenheit'",
                "default": "celsius"
            }
        },
        handler_code="""
async def execute(self, args):
    try:
        # In a real implementation, this would call a weather API
        # This is a mock implementation for demonstration
        import random
        
        location = args.location
        units = getattr(args, 'units', 'celsius')
        
        # Mock weather data
        weather_conditions = ['sunny', 'cloudy', 'rainy', 'snowy', 'windy']
        temperature = random.randint(0, 30) if units == 'celsius' else random.randint(32, 86)
        condition = random.choice(weather_conditions)
        
        unit_symbol = '°C' if units == 'celsius' else '°F'
        
        return {
            'success': True,
            'weather': {
                'location': location,
                'temperature': f"{temperature}{unit_symbol}",
                'condition': condition,
                'humidity': f"{random.randint(30, 90)}%",
                'wind': f"{random.randint(0, 30)} km/h"
            }
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
        """,
        handler_type="async"
    )
    
    result = await mcp_integration.mcp_manager.register_mcp_tool(weather_tool_def)
    
    if result.success:
        logger.info(f"Weather tool created successfully: {result.message}")
        
        # Now use the tool
        weather_tool = tool_registry.get_tool("mcp_weather")
        
        if weather_tool:
            # Create arguments for the tool
            tool_args = {"location": "New York", "units": "celsius"}
            
            # Execute the tool
            result = await weather_tool.execute_with_args(tool_args)
            
            logger.info(f"Weather tool result: {json.dumps(result, indent=2)}")
        else:
            logger.error("Failed to get weather tool from registry")
    else:
        logger.error(f"Failed to create weather tool: {result.message}")
    
    # Example 2: Create a tool from a natural language description
    logger.info("Creating a calculator tool from description")
    
    calc_description = """
Create a calculator tool that can perform basic arithmetic operations.
It should accept two numbers and an operation (add, subtract, multiply, divide).
The tool should return the result of the calculation.
"""
    
    result = await mcp_integration.create_tool_from_description(calc_description, "mcp_")
    
    if result["success"]:
        logger.info(f"Calculator tool created successfully: {result['message']}")
        
        # Now use the tool
        calc_tool = tool_registry.get_tool(result["tool_name"])
        
        if calc_tool:
            # Create arguments for the tool
            tool_args = {"num1": 10, "num2": 5, "operation": "add"}
            
            # Execute the tool
            calc_result = await calc_tool.execute_with_args(tool_args)
            
            logger.info(f"Calculator tool result: {json.dumps(calc_result, indent=2)}")
            
            # Try another operation
            tool_args = {"num1": 10, "num2": 5, "operation": "multiply"}
            calc_result = await calc_tool.execute_with_args(tool_args)
            
            logger.info(f"Calculator tool result: {json.dumps(calc_result, indent=2)}")
        else:
            logger.error(f"Failed to get calculator tool from registry: {result['tool_name']}")
    else:
        logger.error(f"Failed to create calculator tool: {result['message']}")
    
    # List all MCP tools
    tools = await mcp_integration.list_mcp_tools()
    logger.info(f"MCP tools available: {len(tools)}")
    
    for tool in tools:
        logger.info(f"- {tool['name']}: {tool['description']}")
    
    # Deregister a tool
    if len(tools) > 0:
        tool_to_remove = tools[0]["name"]
        result = await mcp_integration.deregister_mcp_tool(tool_to_remove)
        
        if result["success"]:
            logger.info(f"Tool deregistered successfully: {result['message']}")
        else:
            logger.error(f"Failed to deregister tool: {result['message']}")

if __name__ == "__main__":
    asyncio.run(mcp_tool_example())
