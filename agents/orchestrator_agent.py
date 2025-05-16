# agents/orchestrator_agent.py
from typing import Any, Dict, List, Optional, Union, Type
import json
import time
from pydantic import ValidationError, BaseModel

from .base_agent import BaseAgent
from core.llm_interface import LLMInterface
from core.tool_registry import ToolRegistry
from core.memory_manager import MemoryManager
from core.schemas import LLMToolCall, OrchestratorLLMResponse, OrchestratorThought, OrchestratorAction

class OrchestratorAgent(BaseAgent):
    """
    The OrchestratorAgent implements the Model-Chosen Parameters (MCP) ReAct pattern.
    
    It builds a prompt for the LLM that includes:
    1. Available tools (with their JSON schemas)
    2. Current context and goal
    3. Instructions for the LLM to respond with structured thought + action JSON
    
    Then it executes the chosen action (tool call, final answer, clarification, etc.)
    and continues the loop until completion.
    """
    
    def __init__(self, agent_name: str, config: Any,
                 llm_interface: LLMInterface, 
                 memory_manager: MemoryManager, 
                 tool_registry: ToolRegistry):
        """Initialize the OrchestratorAgent."""
        super().__init__(agent_name, config, llm_interface, memory_manager)
        self.tool_registry = tool_registry
        self.max_iterations = self.config_full.orchestrator_max_iterations
        
        print(f"[OrchestratorAgent] Initialized with {len(self.tool_registry.get_all_tools())} tools and "
              f"max {self.max_iterations} iterations per goal.")

    async def run(self, user_goal: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Run the MCP ReAct loop to achieve the user's goal.
        
        Args:
            user_goal: The goal provided by the user
            context: Optional additional context
            
        Returns:
            str: The final result or error message
        """
        context = context or {}
        start_time = time.time()
        print(f"[{self.agent_name}] Received Goal: {user_goal}")
        
        # Add the user goal to memory
        self.memory.add_segment(
            segment_type="USER_GOAL", 
            content_text=user_goal, 
            source="USER"
        )
        
        # Initialize conversation history for the LLM
        conversation_history_for_llm = [] 
        
        # Main ReAct loop
        for iteration in range(self.max_iterations):
            print(f"--- Orchestrator Iteration {iteration + 1}/{self.max_iterations} ---")
            
            # 1. Prepare context and prompt
            current_context_str = self.memory.get_formatted_history(limit=10)
            conversation_history_for_llm.append(f"Context from Memory:\n{current_context_str}")
            
            # Get summary of available tools for the LLM prompt
            tools_summary_for_llm = self.tool_registry.list_tools_for_llm()
            
            # Build the prompt for the LLM
            prompt = self._build_llm_prompt(user_goal, conversation_history_for_llm, tools_summary_for_llm)
            
            # 2. Call LLM for structured thought and action
            llm_response_obj: Union[OrchestratorLLMResponse, str] = await self.llm.generate_structured_orchestrator_response(
                prompt=prompt,
                model_name=self.agent_config.get('model_name')
            )
            
            # Handle error from LLM (non-parseable response)
            if isinstance(llm_response_obj, str):
                error_message = f"LLM generation failed: {llm_response_obj}"
                self.memory.add_segment(
                    segment_type="ORCHESTRATOR_ERROR", 
                    content_text=error_message, 
                    source=self.agent_name
                )
                return error_message
            
            # Successfully parsed response
            llm_thought = llm_response_obj.thought_process
            llm_action = llm_response_obj.chosen_action
            
            # Log the thought to console and memory
            print(f"[Orchestrator Thought] {llm_thought.thought}")
            thought_content = f"Thought: {llm_thought.thought}"
            if llm_thought.reasoning:
                thought_content += f"\nReasoning: {llm_thought.reasoning}"
            if llm_thought.plan:
                thought_content += f"\nPlan: {', '.join(llm_thought.plan)}"
                
            self.memory.add_segment(
                segment_type="LLM_THOUGHT", 
                content_text=thought_content,
                source=self.agent_name + "_LLM"
            )
            
            # Add to conversation history for the next iteration
            conversation_history_for_llm.append(f"My Thought: {llm_thought.thought}")
            
            # 3. Process the chosen action
            if llm_action.action_type == "tool_call" and llm_action.tool_call:
                # Handle tool call action
                tool_name = llm_action.tool_call.tool_name
                tool_args_dict = llm_action.tool_call.arguments
                
                # Look up the tool by name
                tool_to_execute = self.tool_registry.get_tool(tool_name)
                
                if not tool_to_execute:
                    # Tool not found
                    observation = f"Error: Tool '{tool_name}' not found in registry."
                    print(f"[{self.agent_name}_ERROR] {observation}")
                    
                    self.memory.add_segment(
                        segment_type="TOOL_CALL_ERROR", 
                        content_text=observation,
                        source=self.agent_name
                    )
                else:
                    try:
                        # Validate arguments with Pydantic
                        args_schema = tool_to_execute.args_schema
                        validated_args = args_schema(**tool_args_dict)
                        print(f"[Tool Execution] {tool_name} with args: {validated_args.model_dump_json(indent=2)}")
                        
                        # TODO: Ethics check before execution if needed
                        # if config_full.ethics_enabled:
                        #     ethics_approval = ethics_manager.approve_action(...)
                        
                        # Execute the tool with validated arguments
                        tool_raw_output = await tool_to_execute.execute(args=validated_args)
                        
                        # Format the output
                        if isinstance(tool_raw_output, BaseModel):
                            observation = tool_raw_output.model_dump_json(indent=2)
                        elif isinstance(tool_raw_output, dict):
                            observation = json.dumps(tool_raw_output, indent=2)
                        else:
                            observation = str(tool_raw_output)
                        
                        # Log a preview of the observation
                        print(f"[Tool Result] '{tool_name}' observation (first 200 chars): {observation[:200]}...")
                        
                    except ValidationError as ve:
                        # Validation error with arguments
                        observation = f"Error: Invalid arguments for tool '{tool_name}'. Details: {ve}"
                        print(f"[{self.agent_name}_ERROR] {observation}")
                    except Exception as e_tool:
                        # General execution error
                        observation = f"Error executing tool '{tool_name}': {str(e_tool)}"
                        print(f"[{self.agent_name}_ERROR] {observation}")
                
                # Record the tool call and result in memory
                self.memory.add_segment(
                    segment_type="TOOL_CALL_AND_RESULT",
                    tool_name=tool_name,
                    tool_args=tool_args_dict,
                    tool_output=observation,
                    source=self.agent_name
                )
                
                # Add to conversation history for the next iteration
                conversation_history_for_llm.append(f"Action: Called tool {tool_name} with {json.dumps(tool_args_dict)}.\nObservation: {observation}")
                
            elif llm_action.action_type == "delegate_to_agent" and llm_action.delegate_to_agent_key:
                # Handle delegation to another agent
                agent_key = llm_action.delegate_to_agent_key
                task_desc = llm_action.delegated_task_description or "No task description provided"
                
                # This is a placeholder - delegation will be implemented fully later
                observation = f"Delegation to '{agent_key}' not yet fully implemented. Task: {task_desc}"
                print(f"[{self.agent_name}_INFO] {observation}")
                
                self.memory.add_segment(
                    segment_type="DELEGATION_ATTEMPT",
                    content_text=f"Attempted to delegate to {agent_key}: {task_desc}",
                    source=self.agent_name
                )
                
                conversation_history_for_llm.append(f"Action: Delegate to {agent_key}.\nObservation: {observation}")
                
            elif llm_action.action_type == "final_answer" and llm_action.final_answer:
                # Handle final answer - task is complete
                final_response = llm_action.final_answer
                print(f"[{self.agent_name} Final Answer] {final_response}")
                
                # Add to memory
                self.memory.add_segment(
                    segment_type="FINAL_ANSWER", 
                    content_text=final_response, 
                    source=self.agent_name
                )
                
                # TODO: Ethics check on final answer if needed
                # if self.config_full.ethics_enabled:
                #     ethics_manager.check_text_content(final_response)
                
                # Calculate and add execution time
                execution_time = time.time() - start_time
                print(f"[{self.agent_name}] Task completed in {execution_time:.2f} seconds after {iteration + 1} iterations.")
                
                return final_response
            
            elif llm_action.action_type == "clarification_request" and llm_action.clarification_question:
                # Handle clarification request - need more info from user
                clarification = llm_action.clarification_question
                print(f"[{self.agent_name} Clarification Request] {clarification}")
                
                # Add to memory
                self.memory.add_segment(
                    segment_type="CLARIFICATION_REQUEST", 
                    content_text=clarification, 
                    source=self.agent_name
                )
                
                return f"I need some clarification: {clarification}"
            
            else:
                # Invalid or incomplete action
                observation = "Error: LLM chose an invalid or incomplete action."
                print(f"[{self.agent_name}_ERROR] {observation}")
                
                self.memory.add_segment(
                    segment_type="INVALID_ACTION", 
                    content_text=observation,
                    source=self.agent_name
                )
                
                conversation_history_for_llm.append(f"Action: Invalid LLM Action.\nObservation: {observation}")
        
        # Max iterations reached without completion
        timeout_message = f"Orchestrator reached maximum iterations ({self.max_iterations}) for goal: '{user_goal}'. Unable to complete."
        self.memory.add_segment(
            segment_type="ORCHESTRATOR_TIMEOUT", 
            content_text=timeout_message, 
            source=self.agent_name
        )
        print(f"[{self.agent_name}] {timeout_message}")
        return timeout_message

    def _build_llm_prompt(self, goal: str, history_list: List[str], tools_summary: str) -> str:
        """
        Build a prompt for the LLM that includes instructions, context, and expected output format.
        
        Args:
            goal: The user's goal
            history_list: List of conversation history strings
            tools_summary: Summary of available tools
            
        Returns:
            str: The formatted prompt for the LLM
        """
        # Join history items with newlines
        history_str = "\n".join(history_list)
        
        # Instructions for the LLM output format
        output_format_instruction = """
You MUST respond with a single JSON object matching the following structure:
{
  "thought_process": {
    "thought": "Your brief primary thought or reasoning about the current step.",
    "reasoning": "(Optional) More detailed step-by-step reasoning or analysis if needed.",
    "plan": ["(Optional) A short list of sub-steps you're currently considering."]
  },
  "chosen_action": {
    "action_type": "Must be one of: 'tool_call', 'delegate_to_agent', 'final_answer', 'clarification_request'",
    "tool_call": { "(Only if action_type is 'tool_call')"
      "tool_name": "The name of the tool to use from the 'Available Tools' list.",
      "arguments": { "arg_name": "arg_value", ... }
    },
    "delegate_to_agent_key": "(Only if action_type is 'delegate_to_agent') The key of the specialized agent.",
    "delegated_task_description": "(Only if action_type is 'delegate_to_agent') The specific task for that agent.",
    "final_answer": "(Only if action_type is 'final_answer') Your comprehensive final answer to the user's goal.",
    "clarification_question": "(Only if action_type is 'clarification_request') A question to ask the user for more information."
  }
}

Ensure your entire response is a single, valid JSON object that can be parsed directly.
DO NOT include any text outside the JSON structure. Do not use ```json blocks, just output the raw JSON.
"""

        # Construct the full prompt
        prompt = f"""
As the WITS-NEXUS v2 Orchestrator, your task is to achieve the user's goal by thinking step-by-step and then deciding on a single, precise action.

USER'S OVERALL GOAL:
{goal}

AVAILABLE TOOLS:
{tools_summary}

CONVERSATION HISTORY & OBSERVATIONS (most recent last):
---
{history_str}
---

YOUR TASK (Current Iteration):
1. **Analyze:** Review the overall goal, history, and any recent observations.
2. **Reason:** Determine the single most logical next step. This forms your 'thought_process'.
3. **Act:** Based on your reasoning, decide on ONE 'chosen_action'. This could be:
   - Calling a tool with appropriate parameters
   - Providing a final answer if the goal is met
   - Asking for clarification if essential information is missing
   - Delegating to another specialized agent (if available and appropriate)

{output_format_instruction}

Current Thought & Action Plan (as JSON):
"""
        return prompt
