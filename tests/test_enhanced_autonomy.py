# tests/test_enhanced_autonomy.py
"""
Test script for the enhanced AI autonomy system.
This demonstrates how to set up and use the system with a simple example.
"""
import asyncio
import logging
import sys
import os
import json
from typing import Dict, Any, List

# Add parent directory to path to allow importing WITS modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import WITS modules
from core.autonomy import (
    ToolExampleRepository, 
    ToolExampleUsage,
    ToolSimulator, 
    EnhancedJSONHandler, 
    AutonomyEnhancer, 
    EnhancedTool
)
from core.tool_registry import ToolRegistry
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager
from tools.calculator_tool import CalculatorTool, CalculatorArgs, CalculatorResponse
from tools.datetime_tool import DateTimeTool
from pydantic import BaseModel

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/enhanced_autonomy_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("EnhancedAutonomyTest")

# Example enhanced tool based on CalculatorTool
class EnhancedCalculatorTool(EnhancedTool, CalculatorTool):
    """Calculator tool with enhanced learning capabilities."""
    
    def create_error_response(self, error_message: str) -> CalculatorResponse:
        """Create a standardized error response."""
        return CalculatorResponse(
            expression="Error occurred",
            result=None,
            error=error_message
        )

# Test data for examples
EXAMPLE_DATA = [
    {
        "tool_name": "calculator",
        "args": {"expression": "2 + 2"},
        "result": {"expression": "2 + 2", "result": 4, "error": None},
        "context": "Simple addition calculation",
        "explanation": "This is a basic addition operation that works as expected.",
        "success": True
    },
    {
        "tool_name": "calculator",
        "args": {"expression": "10 / 0"},
        "result": {"expression": "10 / 0", "result": None, "error": "Division by zero"},
        "context": "Division by zero example",
        "explanation": "This example shows what happens when you try to divide by zero.",
        "success": False
    },
    {
        "tool_name": "datetime",
        "args": {"format": "%Y-%m-%d"},
        "result": {"current_date": "2023-05-21", "current_time": "14:30:00"},
        "context": "Getting the current date",
        "explanation": "This shows how to get the current date in YYYY-MM-DD format.",
        "success": True
    }
]

async def main():
    """Run a test of the enhanced autonomy system."""
    logger.info("Starting enhanced autonomy test")
    
    # Set up the components
    logger.info("Setting up components...")
    
    # Create tool registry
    tool_registry = ToolRegistry()
    
    # Create LLM interface
    llm_interface = LLMInterface(model_name="mistral:latest", temperature=0.7)
    
    # Create memory manager
    memory_manager = MemoryManager()
    
    # Create example repository
    example_repo = ToolExampleRepository("data/test_examples")
    
    # Set up tool simulator
    tool_simulator = ToolSimulator(example_repo, tool_registry)
    
    # Set up autonomy enhancer
    autonomy_enhancer = AutonomyEnhancer(
        llm_interface=llm_interface,
        example_repository=example_repo,
        tool_simulator=tool_simulator,
        tool_registry=tool_registry
    )
      # Load example data
    logger.info("Loading example data...")
    for example_dict in EXAMPLE_DATA:
        example = ToolExampleUsage(**example_dict)
        example_repo.add_example(example)
    
    # Create and register tools
    logger.info("Creating tools...")
    calculator_tool = EnhancedCalculatorTool(autonomy_enhancer)
    datetime_tool = DateTimeTool()
    
    tool_registry.register_tool(calculator_tool)
    tool_registry.register_tool(datetime_tool)
    
    # Test tool simulation
    logger.info("Testing tool simulation...")
    
    # Test valid calculator args
    valid_calc_result = await tool_simulator.simulate_tool_execution(
        "calculator", {"expression": "5 * 7"}
    )
    logger.info(f"Valid calculator simulation result: {valid_calc_result}")
    
    # Test invalid calculator args
    invalid_calc_result = await tool_simulator.simulate_tool_execution(
        "calculator", {}  # Missing required 'expression' field
    )
    logger.info(f"Invalid calculator simulation result: {invalid_calc_result}")
    
    # Test tool execution with learning
    logger.info("Testing tool execution with learning...")
    args = CalculatorArgs(expression="3 * (4 + 2)")
    result = await calculator_tool.execute_with_learning(args)
    logger.info(f"Calculator result: {result}")
    
    # Generate example prompts
    logger.info("Generating example prompts...")
    calculator_prompt = await autonomy_enhancer.generate_tool_examples_prompt("calculator")
    logger.info(f"Calculator prompt:\n{calculator_prompt}")
    
    # Generate enhanced tools overview
    logger.info("Generating enhanced tools overview...")
    tools_overview = await autonomy_enhancer.generate_enhanced_tools_prompt()
    logger.info(f"Enhanced tools overview:\n{tools_overview}")
    
    # Log tool usage statistics
    logger.info("Tool usage statistics:")
    autonomy_enhancer.log_tool_usage_statistics()
    
    logger.info("Enhanced autonomy test completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
