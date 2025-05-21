# core/autonomy/autonomy_enhancer.py
"""
AutonomyEnhancer coordinates AI learning and decision-making for improved tool usage.
Think of it as a coach that helps the AI get better at using tools over time! \o/
"""
import logging
import time
import json
from typing import Dict, Any, Tuple, List, Optional

from .tool_example_repository import ToolExampleRepository, ToolExampleUsage
from .tool_simulator import ToolSimulator
from .enhanced_json_handler import EnhancedJSONHandler
from .example_prompt_templates import EnhancedPromptTemplate
from core.tool_registry import ToolRegistry
from core.llm_interface import LLMInterface

class ToolUsageStats:
    """Statistics for tool usage to track performance and learning."""
    
    def __init__(self, tool_name: str):
        """Initialize stats for a tool."""
        self.tool_name = tool_name
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.average_execution_time = 0.0
        self.last_used = 0.0  # timestamp
    
    def record_usage(self, success: bool, execution_time: float):
        """Record a usage of the tool."""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        
        # Update average execution time with weighted average
        if self.total_calls == 1:
            self.average_execution_time = execution_time
        else:
            self.average_execution_time = (
                (self.average_execution_time * (self.total_calls - 1) + execution_time) / self.total_calls
            )
        
        self.last_used = time.time()
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate of the tool."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

class AutonomyEnhancer:
    """
    Enhancer for AI autonomy in tool usage and learning.
    The AI's personal coach and learning manager! ^_^
    """
    
    def __init__(self, llm_interface: LLMInterface, example_repository: ToolExampleRepository, 
                 tool_simulator: ToolSimulator, tool_registry: ToolRegistry):
        """
        Initialize the autonomy enhancer.
        
        Args:
            llm_interface: Interface for LLM interactions
            example_repository: Repository of tool usage examples
            tool_simulator: Simulator for testing tools
            tool_registry: Registry of available tools
        """
        self.llm = llm_interface
        self.example_repository = example_repository
        self.tool_simulator = tool_simulator
        self.tool_registry = tool_registry
        self.logger = logging.getLogger("WITS.Autonomy.AutonomyEnhancer")
        self.json_handler = EnhancedJSONHandler()
        self.tool_stats: Dict[str, ToolUsageStats] = {}
    
    async def enhance_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Enhance a tool call before execution by validating and suggesting improvements.
        
        Args:
            tool_name: Name of the tool to call
            args: Arguments to pass to the tool
            
        Returns:
            Tuple[str, Dict[str, Any]]: The tool name and potentially improved arguments
        """
        self.logger.debug(f"Enhancing tool call for '{tool_name}'")
        
        # Ensure we have stats for this tool
        if tool_name not in self.tool_stats:
            self.tool_stats[tool_name] = ToolUsageStats(tool_name)
        
        # Check if the tool exists
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            self.logger.warning(f"Attempted to use non-existent tool: '{tool_name}'")
            return tool_name, args  # Can't enhance a non-existent tool
        
        # Get expected schema
        try:
            # Simulate tool execution to validate args and get suggestions
            simulation_result = await self.tool_simulator.simulate_tool_execution(tool_name, args)
            
            if not simulation_result.valid:
                self.logger.info(f"Tool '{tool_name}' simulation found invalid args: {simulation_result.error_message}")
                
                if simulation_result.suggested_args:
                    self.logger.info(f"Using suggested args for tool '{tool_name}'")
                    return tool_name, simulation_result.suggested_args
            
        except Exception as e:
            self.logger.warning(f"Error during tool enhancement for '{tool_name}': {str(e)}")
        
        # Return the original or enhanced arguments
        return tool_name, args
    
    async def learn_from_execution(self, tool_name: str, args: Dict[str, Any], 
                                  result: Any, success: bool, context: str = "", 
                                  execution_time: float = 0.0) -> None:
        """
        Learn from a tool execution (successful or not) and update the repository.
        
        Args:
            tool_name: Name of the tool that was executed
            args: Arguments that were passed to the tool
            result: Result or error returned by the tool
            success: Whether the execution was successful
            context: Context in which the tool was used
            execution_time: How long the execution took
        """
        self.logger.debug(f"Learning from tool execution: '{tool_name}' (success={success})")
        
        # Update usage statistics
        if tool_name not in self.tool_stats:
            self.tool_stats[tool_name] = ToolUsageStats(tool_name)
        self.tool_stats[tool_name].record_usage(success, execution_time)
        
        # Generate explanation
        if success:
            explanation = f"Successfully executed {tool_name} with the provided arguments."
        else:
            explanation = f"Failed to execute {tool_name}. Error: {result}"
        
        # Create an example from this execution
        example = ToolExampleUsage(
            tool_name=tool_name,
            args=args,
            result=result,
            context=context,
            explanation=explanation,
            success=success
        )
        
        # Add to repository
        self.example_repository.add_example(example)
        
    async def generate_tool_examples_prompt(self, tool_name: str) -> str:
        """
        Generate a prompt that includes examples of tool usage.
        
        Args:
            tool_name: Name of the tool to generate examples for
            
        Returns:
            str: A prompt with examples for the specified tool
        """
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            self.logger.warning(f"Cannot generate examples for non-existent tool: '{tool_name}'")
            return f"Tool '{tool_name}' does not exist."
        
        examples = self.example_repository.get_successful_examples(tool_name, limit=3)
        
        return EnhancedPromptTemplate.tool_usage_template(
            tool_name=tool_name,
            description=tool.description,
            examples=examples
        )
    
    def get_tool_usage_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get usage statistics for all tracked tools.
        
        Returns:
            Dict[str, Dict[str, Any]]: Statistics for each tool
        """
        stats = {}
        for tool_name, tool_stats in self.tool_stats.items():
            stats[tool_name] = {
                "total_calls": tool_stats.total_calls,
                "successful_calls": tool_stats.successful_calls,
                "failed_calls": tool_stats.failed_calls,
                "success_rate": tool_stats.success_rate,
                "average_execution_time": tool_stats.average_execution_time,
                "last_used": tool_stats.last_used
            }
        return stats
    
    async def generate_enhanced_tools_prompt(self) -> str:
        """
        Generate an enhanced prompt with all available tools, including examples for common ones.
        
        Returns:
            str: A comprehensive tools overview with examples
        """
        tools_info = self.tool_registry.get_tools_for_llm()
        
        # Sort tools by success rate if available, otherwise alphabetically
        def get_tool_priority(tool_info):
            tool_name = tool_info["name"]
            if tool_name in self.tool_stats:
                # Higher success rate is better, higher call count is better
                return (self.tool_stats[tool_name].success_rate, 
                        self.tool_stats[tool_name].total_calls)
            return (0, 0)  # Default priority for unused tools
            
        tools_info.sort(key=get_tool_priority, reverse=True)
        
        return EnhancedPromptTemplate.tools_overview_template(
            tools_info=tools_info,
            with_examples=True,
            example_repository=self.example_repository
        )
    
    def log_tool_usage_statistics(self) -> None:
        """Log tool usage statistics for monitoring."""
        self.logger.info("Tool usage statistics:")
        for tool_name, stats in self.tool_stats.items():
            self.logger.info(f"  {tool_name}: {stats.total_calls} calls, " 
                             f"{stats.success_rate:.1%} success rate, "
                             f"{stats.average_execution_time:.2f}ms avg execution time")
