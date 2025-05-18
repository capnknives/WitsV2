# agents/orchestrator_agent.py
from typing import Any, Dict, List, Optional, Union, Type, AsyncGenerator
import json
import time
import logging
from pydantic import ValidationError, BaseModel, Field

from .base_agent import BaseAgent
from core.llm_interface import LLMInterface
from core.tool_registry import ToolRegistry
from core.memory_manager import MemoryManager
from core.debug_utils import log_async_execution_time # Assuming this decorator is defined
from core.schemas import (
    LLMToolCall,
    OrchestratorLLMResponse,
    OrchestratorThought,
    OrchestratorAction,
    MemorySegment # Assuming MemorySegment is used for self.memory.add_segment
)

class AgentDelegationRequest(BaseModel):
    """Model for agent delegation requests."""
    agent_key: str = Field(..., description="Identifier of the agent to delegate to.")
    task_description: str = Field(..., description="Description of the task to delegate.")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context for the task.")

class AgentDelegationResponse(BaseModel):
    """Model for agent delegation responses."""
    agent_key: str
    result: str
    completed: bool = True
    error: Optional[str] = None

class StreamData(BaseModel): # Define it here or import if it's in a shared schemas file
    type: str
    content: Any
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    iteration: Optional[int] = None
    max_iterations: Optional[int] = None
    # For thoughts
    reasoning: Optional[str] = None
    plan: Optional[List[str]] = None
    # For errors
    error_details: Optional[str] = None

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
                 tool_registry: ToolRegistry,
                 specialized_agents: Optional[Dict[str, BaseAgent]] = None):
        """Initialize the OrchestratorAgent."""
        super().__init__(agent_name, config, llm_interface, memory_manager)
        self.tool_registry = tool_registry

        # Check for max_iterations in different config structures
        if hasattr(self.config_full, "orchestrator_max_iterations"):
            # AppConfig case
            self.max_iterations = self.config_full.orchestrator_max_iterations
        elif hasattr(self.config_full, "max_iterations"):
            # AgentProfileConfig case (if config passed is a profile)
            self.max_iterations = self.config_full.max_iterations
        else:
            # Default fallback
            self.max_iterations = 10 # Default or from a global config setting

        self.specialized_agents = specialized_agents or {}
        self.logger = logging.getLogger(f"WITS.Agents.{agent_name}")

        self.logger.info(f"OrchestratorAgent initialized with {len(self.tool_registry.get_all_tools())} tools, "
                         f"{len(self.specialized_agents)} specialized agents, and "
                         f"max {self.max_iterations} iterations per goal.")

    @log_async_execution_time # Make sure this decorator correctly handles async generators
    async def run(self, user_goal: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[StreamData, None]:
        """
        Run the MCP ReAct loop to achieve the user's goal, yielding streamable updates.

        Args:
            user_goal: The goal provided by the user
            context: Optional additional context

        Yields:
            StreamData: Structured data packets for streaming to the client.
        """
        context = context or {}
        start_time = time.time()
        yield StreamData(type="info", content=f"Orchestrator received goal: {user_goal}")

        # Assuming self.memory.add_segment is an async method or can be awaited
        await self.memory.add_segment(
            segment_type="USER_GOAL",
            content_text=user_goal, # Ensure content_text is the correct field name
            source="USER"
        )

        conversation_history_for_llm = []

        # Main ReAct loop
        for iteration_num in range(self.max_iterations):
            yield StreamData(type="info", iteration=iteration_num + 1, max_iterations=self.max_iterations, content=f"Starting iteration {iteration_num + 1}/{self.max_iterations}")

            # Assuming self.memory.get_formatted_history is async
            current_context_str = await self.memory.get_formatted_history(limit=10)
            conversation_history_for_llm.append(f"Context from Memory:\n{current_context_str}")

            tools_summary_for_llm = self.tool_registry.list_tools_for_llm()
            prompt = self._build_llm_prompt(user_goal, conversation_history_for_llm, tools_summary_for_llm)

            llm_response_obj: Union[OrchestratorLLMResponse, str] = await self.llm.generate_structured_orchestrator_response(
                prompt=prompt,
                model_name=self.agent_config.get("model_name") # agent_config is from BaseAgent
            )

            if isinstance(llm_response_obj, str):
                error_message = f"LLM generation failed: {llm_response_obj}"
                await self.memory.add_segment(
                    segment_type="ORCHESTRATOR_ERROR",
                    content_text=error_message,
                    source=self.agent_name
                )
                yield StreamData(type="error", content="LLM generation failed.", error_details=llm_response_obj)
                return

            llm_thought = llm_response_obj.thought_process
            llm_action = llm_response_obj.chosen_action

            thought_content_dict = {
                "thought": llm_thought.thought,
                "reasoning": llm_thought.reasoning,
                "plan": llm_thought.plan
            }
            yield StreamData(type="llm_thought", content=thought_content_dict, iteration=iteration_num+1, reasoning=llm_thought.reasoning, plan=llm_thought.plan)

            thought_content_str = f"Thought: {llm_thought.thought}"
            if llm_thought.reasoning: thought_content_str += f"\nReasoning: {llm_thought.reasoning}" # Use \n for newlines
            if llm_thought.plan: thought_content_str += f"\nPlan: {', '.join(llm_thought.plan)}" # Use \n for newlines

            await self.memory.add_segment(
                segment_type="LLM_THOUGHT",
                content_text=thought_content_str,
                source=self.agent_name + "_LLM"
            )
            conversation_history_for_llm.append(f"My Thought: {llm_thought.thought}")

            # Process the chosen action
            observation: str = "" # Ensure observation is always defined
            if llm_action.action_type == "tool_call" and llm_action.tool_call:
                tool_name = llm_action.tool_call.tool_name
                tool_args_dict = llm_action.tool_call.arguments
                yield StreamData(type="tool_call", tool_name=tool_name, tool_args=tool_args_dict, content=f"Attempting to call tool: {tool_name}", iteration=iteration_num+1)

                tool_to_execute = self.tool_registry.get_tool(tool_name)
                if not tool_to_execute:
                    observation = f"Error: Tool '{tool_name}' not found in registry."
                    yield StreamData(type="tool_error", tool_name=tool_name, content=observation, iteration=iteration_num+1)
                    await self.memory.add_segment(segment_type="TOOL_CALL_ERROR", content_text=observation, source=self.agent_name)
                else:
                    try:
                        args_schema = tool_to_execute.args_schema
                        validated_args = args_schema(**tool_args_dict)

                        tool_raw_output = await tool_to_execute.execute(args=validated_args)
                        if isinstance(tool_raw_output, BaseModel): observation = tool_raw_output.model_dump_json(indent=2)
                        elif isinstance(tool_raw_output, dict): observation = json.dumps(tool_raw_output, indent=2)
                        else: observation = str(tool_raw_output)

                        yield StreamData(type="tool_result", tool_name=tool_name, content=observation, iteration=iteration_num+1)
                    except ValidationError as ve:
                        observation = f"Error: Invalid arguments for tool '{tool_name}'. Details: {ve}"
                        yield StreamData(type="tool_error", tool_name=tool_name, content=observation, error_details=str(ve), iteration=iteration_num+1)
                    except Exception as e_tool:
                        observation = f"Error executing tool '{tool_name}': {str(e_tool)}"
                        yield StreamData(type="tool_error", tool_name=tool_name, content=observation, error_details=str(e_tool), iteration=iteration_num+1)

                # Ensure MemorySegment content structure is correct, e.g. it expects content_text
                # If add_segment takes separate tool_name, tool_args, tool_output, use that.
                # Based on the file, it looks like MemorySegment now has a `content` field of type `MemorySegmentContent`.
                # And `add_segment` takes individual fields. Let's use the existing MemoryManager's add_segment signature.
                await self.memory.add_segment(
                    segment_type="TOOL_CALL_AND_RESULT",
                    tool_name=tool_name if llm_action.tool_call else None, # Pass tool_name
                    tool_args=tool_args_dict if llm_action.tool_call else None, # Pass tool_args
                    tool_output=observation, # Pass tool_output
                    source=self.agent_name
                )
                conversation_history_for_llm.append(f"Action: Called tool {tool_name} with {json.dumps(tool_args_dict)}.\nObservation: {observation}") # Use \n

            elif llm_action.action_type == "delegate_to_agent" and llm_action.delegate_to_agent_key:
                agent_key = llm_action.delegate_to_agent_key
                task_desc = llm_action.delegated_task_description or "No task description provided"
                yield StreamData(type="delegation_start", content=f"Delegating task to agent: {agent_key}", tool_name=agent_key, tool_args={"task_description": task_desc}, iteration=iteration_num+1)

                delegation_result = await self._handle_agent_delegation(agent_key, task_desc, context)
                observation = f"Agent {agent_key} delegation result: {delegation_result.result}"

                if delegation_result.error:
                    self.logger.error(f"Delegation to {agent_key} failed: {delegation_result.error}")
                    yield StreamData(type="delegation_error", content=f"Delegation to {agent_key} failed.", error_details=delegation_result.error, tool_name=agent_key, iteration=iteration_num+1)
                else:
                    self.logger.info(f"Delegation to {agent_key} completed successfully")
                    yield StreamData(type="delegation_result", content=observation, tool_name=agent_key, iteration=iteration_num+1)

                await self.memory.add_segment(
                    segment_type="DELEGATION_ATTEMPT",
                    content_text=f"Attempted to delegate to {agent_key}: {task_desc}\nResult: {observation}", # Use \n
                    source=self.agent_name
                )
                conversation_history_for_llm.append(f"Action: Delegate to {agent_key}.\nObservation: {observation}") # Use \n

            elif llm_action.action_type == "final_answer" and llm_action.final_answer:
                final_response = llm_action.final_answer
                yield StreamData(type="final_answer", content=final_response, iteration=iteration_num+1)

                await self.memory.add_segment(segment_type="FINAL_ANSWER", content_text=final_response, source=self.agent_name)
                execution_time = time.time() - start_time
                yield StreamData(type="info", content=f"Task completed in {execution_time:.2f}s after {iteration_num + 1} iterations.")
                return

            elif llm_action.action_type == "clarification_request" and llm_action.clarification_question:
                clarification = llm_action.clarification_question
                yield StreamData(type="clarification_request", content=clarification, iteration=iteration_num+1)
                await self.memory.add_segment(segment_type="CLARIFICATION_REQUEST", content_text=clarification, source=self.agent_name)
                return

            else: # Invalid or incomplete action
                observation = "Error: LLM chose an invalid or incomplete action."
                yield StreamData(type="error", content=observation, error_details="Invalid LLM action type or missing fields.", iteration=iteration_num+1)
                await self.memory.add_segment(segment_type="INVALID_ACTION", content_text=observation, source=self.agent_name)
                conversation_history_for_llm.append(f"Action: Invalid LLM Action.\nObservation: {observation}") # Use \n

        timeout_message = f"Orchestrator reached maximum iterations ({self.max_iterations}) for goal: '{user_goal}'. Unable to complete."
        await self.memory.add_segment(segment_type="ORCHESTRATOR_TIMEOUT", content_text=timeout_message, source=self.agent_name)
        yield StreamData(type="error", content=timeout_message, error_details="Max iterations reached.")
        return

    async def _handle_agent_delegation(self, agent_key: str, task_description: str, context: Dict[str, Any]) -> AgentDelegationResponse:
        """
        Delegate a task to a specialized agent.

        Args:
            agent_key: The key of the agent to delegate to
            task_description: Description of the task to perform
            context: Additional context for the task

        Returns:
            AgentDelegationResponse: The result of the delegation
        """
        if agent_key not in self.specialized_agents:
            return AgentDelegationResponse(
                agent_key=agent_key,
                result=f"Agent '{agent_key}' not found in specialized agents registry",
                completed=False,
                error=f"Unknown agent key: {agent_key}"
            )

        self.logger.info(f"Delegating task to {agent_key}: {task_description}")

        try:
            agent = self.specialized_agents[agent_key]
            # Adapt based on how specialized agents' run methods work (return string or stream)
            # If specialized_agent.run() is also an async generator:
            final_delegated_result = ""
            delegated_agent_run_method = agent.run(task_description, context)

            if isinstance(delegated_agent_run_method, AsyncGenerator):
                async for stream_item in delegated_agent_run_method:
                    # You might want to yield these intermediate steps to the main stream too
                    # For now, just capture the final_answer or error from the delegate
                    if stream_item.type == "final_answer":
                        final_delegated_result = str(stream_item.content)
                        break
                    elif stream_item.type == "error":
                        self.logger.error(f"Delegated agent {agent_key} encountered an error: {stream_item.error_details}")
                        final_delegated_result = f"Error from {agent_key}: {stream_item.content}"
                        # Potentially raise an exception or mark as completed=False
                        return AgentDelegationResponse(
                            agent_key=agent_key,
                            result=final_delegated_result,
                            completed=False, # Or True if error is part of the "result"
                            error=f"Delegated agent error: {stream_item.error_details or stream_item.content}"
                        )
                if not final_delegated_result: # If generator finished without a final_answer
                    final_delegated_result = "Delegated agent completed without a conclusive final answer."

            else: # If agent.run() is a coroutine returning a string or a non-generator result
                final_delegated_result_raw = await delegated_agent_run_method
                final_delegated_result = str(final_delegated_result_raw)


            return AgentDelegationResponse(
                agent_key=agent_key,
                result=final_delegated_result,
                completed=True
            )

        except Exception as e:
            error_msg = f"Error during delegation to {agent_key}: {str(e)}"
            self.logger.error(error_msg, exc_info=True) # Log with traceback
            return AgentDelegationResponse(
                agent_key=agent_key,
                result=f"Failed to complete task: {error_msg}",
                completed=False,
                error=error_msg
            )

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
        # Use actual newlines for joining history items
        history_str = "\n".join(history_list)

        # Instructions for the LLM output format - corrected the tool_call part
        # and other newline representations.
        output_format_instruction = """\
You MUST respond with a single JSON object matching the following structure:
{
  "thought_process": {
    "thought": "Your brief primary thought or reasoning about the current step.",
    "reasoning": "(Optional) More detailed step-by-step reasoning or analysis if needed.",
    "plan": ["(Optional) A short list of sub-steps you're currently considering."]
  },
  "chosen_action": {
    "action_type": "Must be one of: 'tool_call', 'delegate_to_agent', 'final_answer', 'clarification_request'",
    "tool_call": {
      "tool_name": "The name of the tool to use from the 'Available Tools' list.",
      "arguments": { /* Populated with key-value pairs specific to the chosen tool, e.g., "query": "value" */ }
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

        # Use actual newlines for agent info
        specialized_agents_info = ""
        if self.specialized_agents:
            specialized_agents_info = "\nSPECIALIZED AGENTS YOU CAN DELEGATE TO:\n"
            for key, agent_instance in self.specialized_agents.items():
                # Try to get a description from the agent itself if available, or use a generic one
                # This assumes specialized agents might have a 'description' attribute or method.
                # For now, using the logic from your previous corrected snippet.
                agent_description = "Specialized agent with custom capabilities" # Default
                if "scribe" in key.lower(): # Assuming key is like 'scribe_agent'
                    agent_description = "Creates documentation, summaries, reports, and transforms text content."
                elif "analyst" in key.lower():
                    agent_description = "Analyzes data, identifies patterns, and provides insights."
                elif "engineer" in key.lower():
                    agent_description = "Analyzes and modifies code, implements features, and manages code quality."
                elif "researcher" in key.lower():
                    agent_description = "Researches topics, gathers information, and synthesizes findings."
                # You could potentially get this from agent_instance.description if it exists
                specialized_agents_info += f"- {key}: {agent_description}\n"


        # Construct the full prompt
        prompt = f"""\
As the WITS-NEXUS v2 Orchestrator, your task is to achieve the user's goal by thinking step-by-step and then deciding on a single, precise action.

USER'S OVERALL GOAL:
{goal}

AVAILABLE TOOLS:
{tools_summary}{specialized_agents_info}

CONVERSATION HISTORY & OBSERVATIONS (most recent last):
---
{history_str}
---

YOUR TASK (Current Iteration):
1. **Analyze:** Review the overall goal, history, and any recent observations.
2. **Reason:** Determine the single most logical next step. This forms your 'thought_process'.
3. **Act:** Based on your reasoning, decide on ONE 'chosen_action'. This could be:
   - Calling a tool with appropriate parameters (ensure arguments match the tool's specific needs).
   - Providing a final answer if the goal is met.
   - Asking for clarification if essential information is missing.
   - Delegating to another specialized agent when their expertise is clearly needed for the current sub-task.

GUIDELINES FOR DELEGATION:
- Delegate to the ScribeAgent for documentation, content creation, and text transformation tasks.
- Delegate to the AnalystAgent for data analysis, pattern recognition, and insight generation.
- Delegate to the EngineerAgent for code analysis, modification, and generation tasks.
- Delegate to the ResearcherAgent for in-depth information gathering and synthesis tasks.
- Provide clear, specific, and self-contained instructions when delegating.

{output_format_instruction}

Current Thought & Action Plan (as JSON):
"""
        return prompt