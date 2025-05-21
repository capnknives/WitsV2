# core/autonomy/integration_example.py
"""
Example of integrating enhanced AI autonomy components with the OrchestratorAgent.

This demonstrates how to integrate the autonomy components into the existing codebase.
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime

from core.autonomy import (
    ToolExampleRepository, 
    ToolSimulator, 
    EnhancedJSONHandler, 
    AutonomyEnhancer, 
    EnhancedTool
)
from core.tool_registry import ToolRegistry
from core.llm_interface import LLMInterface
from core.schemas import StreamData

from agents.base_orchestrator_agent import BaseOrchestratorAgent, OrchestratorLLMResponse

class EnhancedOrchestratorAgent(BaseOrchestratorAgent):
    """
    OrchestratorAgent with enhanced AI autonomy capabilities.
    The same orchestrator you love, but with superpowered learning abilities! \o/
    """
    
    def __init__(self, agent_name, config, llm_interface, memory_manager, 
                 tool_registry, delegation_targets=None, max_iterations=5):
        """Initialize with enhanced autonomy components."""
        super().__init__(agent_name, config, llm_interface, memory_manager, 
                         tool_registry, delegation_targets, max_iterations)
        
        # Initialize autonomy components
        self.autonomy_path = config.get('autonomy_path', 'data/tool_examples')
        self.example_repository = ToolExampleRepository(self.autonomy_path)
        self.tool_simulator = ToolSimulator(self.example_repository, self.tool_registry)
        self.json_handler = EnhancedJSONHandler()
        self.autonomy_enhancer = AutonomyEnhancer(
            llm_interface=self.llm,
            example_repository=self.example_repository,
            tool_simulator=self.tool_simulator,
            tool_registry=self.tool_registry
        )
        
        self.logger.info(f"EnhancedOrchestratorAgent initialized with autonomy components! \o/")
    
    def _build_llm_prompt(self, goal: str, conversation_history: List[str], previous_steps: List[Dict[str, Any]]) -> str:
        """
        Build an enhanced prompt with examples for the LLM.
        This supercharged prompt helps our AI make better decisions! ^_^
        """
        # Start with the enhanced tools overview
        enhanced_tools_prompt = asyncio.run(self.autonomy_enhancer.generate_enhanced_tools_prompt())
        
        # Now add the standard prompt content
        prompt = f"Goal: {goal}\n\n"
        
        # Add conversation history if available
        if conversation_history:
            prompt += "Conversation History:\n"
            prompt += "\n".join([f"- {msg}" for msg in conversation_history]) + "\n\n"
        
        # Add previous steps if any
        if previous_steps:
            prompt += "Previous Steps:\n"
            for i, step in enumerate(previous_steps):
                prompt += f"Step {i+1}:\n"
                prompt += f"Thought: {step.get('thought', 'No thought recorded')}\n"
                if step.get('action'):
                    prompt += f"Action: {json.dumps(step['action'], indent=2)}\n"
                if step.get('observation'):
                    prompt += f"Observation: {step['observation']}\n"
                prompt += "\n"
        
        # Add learning reminder
        prompt += EnhancedPromptTemplate.learning_reminder_template() + "\n\n"
        
        # Add enhanced tools overview
        prompt += enhanced_tools_prompt + "\n\n"
        
        # Add delegation targets
        if self.delegation_targets:
            prompt += "Available Agents:\n"
            prompt += "\n".join([f"- {agent}" for agent in self.delegation_targets]) + "\n"
        
        prompt += "\nWhat would you like to do next?"
        return prompt
    
    async def _execute_tool(self, tool_call):
        """
        Enhanced tool execution with learning capabilities.
        Every tool execution is a learning opportunity! =D
        """
        start_time = time.time()
        tool_name = tool_call.tool_name
        tool_args = tool_call.arguments
        
        # Get the tool from registry
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            error_msg = f"Tool '{tool_name}' not found in registry"
            self.logger.error(error_msg)
            execution_time = (time.time() - start_time) * 1000
            
            # Learn from this failed execution even though the tool doesn't exist
            await self.autonomy_enhancer.learn_from_execution(
                tool_name=tool_name,
                args=tool_args,
                result=error_msg,
                success=False,
                execution_time=execution_time
            )
            
            return {"error": error_msg}
            
        # Enhance the tool call using autonomy enhancer
        enhanced_tool_name, enhanced_tool_args = await self.autonomy_enhancer.enhance_tool_call(
            tool_name=tool_name, 
            args=tool_args
        )
        
        try:
            # Execute the tool with learning if it's an EnhancedTool
            if isinstance(tool, EnhancedTool):
                # Set the execution context
                tool.set_execution_context({"task_description": self.current_goal})
                
                # Execute with learning
                args_instance = tool.args_schema(**enhanced_tool_args)
                result = await tool.execute_with_learning(args_instance)
                
            else:
                # Execute normally if it's not an EnhancedTool
                args_instance = tool.args_schema(**enhanced_tool_args)
                result = await tool.execute(args_instance)
                
                # Learn from execution manually
                execution_time = (time.time() - start_time) * 1000
                await self.autonomy_enhancer.learn_from_execution(
                    tool_name=tool_name,
                    args=enhanced_tool_args,
                    result=result.model_dump() if hasattr(result, "model_dump") else result,
                    success=True,  # Assume success if no exception was raised
                    context=self.current_goal,
                    execution_time=execution_time
                )
                
            return result
            
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            self.logger.error(error_msg)
            
            # Learn from this failed execution
            execution_time = (time.time() - start_time) * 1000
            await self.autonomy_enhancer.learn_from_execution(
                tool_name=tool_name,
                args=enhanced_tool_args,
                result=str(e),
                success=False,
                context=self.current_goal,
                execution_time=execution_time
            )
            
            return {"error": error_msg}
    
    def _parse_llm_response(self, response_str: str, session_id: str) -> Optional[OrchestratorLLMResponse]:
        """
        Parse the LLM response with enhanced JSON handling.
        Better error recovery = fewer frustrating failures! ^_^
        """
        try:
            # First attempt with the parent class implementation
            parsed_response = super()._parse_llm_response(response_str, session_id)
            if parsed_response:
                return parsed_response
            
            # If that fails, try our enhanced JSON handler
            self.logger.warning(f"Standard parsing failed for session '{session_id}', trying enhanced JSON handling")
            
            # Extract the tool call if it looks like one
            tool_call = self.json_handler.extract_tool_call_json(response_str)
            if tool_call and "tool_name" in tool_call and "args" in tool_call:
                # Construct a minimal valid OrchestratorLLMResponse
                from core.schemas import OrchestratorThought, OrchestratorAction, LLMToolCall
                
                # Create the components
                thought = OrchestratorThought(
                    reasoning="Reconstructed from partial JSON using enhanced handling",
                    plan=["Execute the extracted tool call"]
                )
                
                tool_call_obj = LLMToolCall(
                    tool_name=tool_call["tool_name"],
                    args=tool_call["args"],
                    explanation=tool_call.get("explanation", "No explanation provided")
                )
                
                action = OrchestratorAction(
                    action_type="USE_TOOL",
                    tool_calls=[tool_call_obj]
                )
                
                # Combine into the final response
                response = OrchestratorLLMResponse(
                    thought_process=thought,
                    chosen_action=action
                )
                
                self.logger.info(f"Successfully reconstructed OrchestratorLLMResponse for session '{session_id}'")
                return response
            
            self.logger.error(f"Enhanced parsing also failed for session '{session_id}'")
            return None
            
        except Exception as e:
            self.logger.error(f"Error in enhanced LLM response parsing for session '{session_id}': {str(e)}")
            return None

    async def run(self, user_goal: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[StreamData, None]:
        """
        Run the enhanced orchestrator with AI autonomy capabilities.
        
        Args:
            user_goal: The user's goal to accomplish
            context: Additional context for execution
            
        Yields:
            StreamData: Progress updates and results
        """
        # Store the current goal for context in tool execution
        self.current_goal = user_goal
        
        # Use the standard implementation, our enhancements are in the overridden methods
        async for data in super().run(user_goal, context):
            yield data
            
        # Log tool usage statistics at the end of the run
        self.autonomy_enhancer.log_tool_usage_statistics()
