# agents/base_orchestrator_agent.py
from typing import Any, Dict, List, Optional, AsyncGenerator, Union
import json
import logging
import asyncio
import uuid
from pydantic import BaseModel
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager
from core.schemas import StreamData
from core.json_utils import safe_json_loads
from utils.logging_utils import log_async_execution_time
from agents.base_agent import BaseAgent

class LLMToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]

class OrchestratorAction(BaseModel):
    action_type: str
    tool_call: Optional[LLMToolCall] = None
    final_answer: Optional[str] = None

class OrchestratorThought(BaseModel):
    thought: str
    reasoning: Optional[str] = None
    plan: Optional[str] = None

class OrchestratorLLMResponse(BaseModel):
    thought_process: OrchestratorThought
    chosen_action: OrchestratorAction

class BaseOrchestratorAgent(BaseAgent):
    """Base class for all orchestrator agents. Handles core orchestration logic."""
    
    def __init__(self, 
                 agent_name: str, 
                 config: Dict[str, Any], 
                 llm_interface: LLMInterface, 
                 memory_manager: MemoryManager, 
                 tool_registry: Optional[Any] = None, 
                 delegation_targets: Optional[Union[Dict[str, BaseAgent], List[str]]] = None, 
                 max_iterations: int = 5):
        super().__init__(agent_name, config, llm_interface, memory_manager, tool_registry)
        
        # Handle delegation targets (gotta have friends to delegate to! >.>)
        if isinstance(delegation_targets, dict):
            self.agents_registry = delegation_targets
            self.delegation_targets = list(delegation_targets.keys())
        else:
            self.agents_registry = {}
            self.delegation_targets = list(delegation_targets) if delegation_targets else []
            
        # Let's get this party started! Configure all the things xD
        self.orchestrator_model_name = getattr(config, 'llm_model_name', llm_interface.model_name)  # Brain power: activated!
        self.max_iterations = max_iterations  # How many tries before we give up? x.x
        self.final_answer_tool = 'FinalAnswerTool'  # For when we finally figure it out =D

        self.logger.info(f"OrchestratorAgent is alive! Using brain model: {llm_interface.model_name} ^_^ Max tries: {self.max_iterations}")
        self.logger.info(f"My minions... I mean, delegation targets: {self.delegation_targets} =D")
        self.logger.info(f"When I know the answer I'll use: {self.final_answer_tool} \\o/")

    async def _handle_agent_delegation(self, agent_name: str, goal: str, context: Dict[str, Any]) -> AsyncGenerator[StreamData, None]:
        """Handle delegation to another agent."""
        self.logger.info(f"Delegating to {agent_name}: {goal}")

        if not self.agents_registry or agent_name not in self.agents_registry:
            self.logger.error(f"No agents_registry or {agent_name} not found in registry.")
            yield StreamData(type="error", content=f"Agent {agent_name} not found in registry")
            return

        delegate_agent = self.agents_registry[agent_name]
        self.logger.info(f"Found agent {delegate_agent}")

        try:
            # Prepare delegation context 
            delegation_context = context.copy() if context else {}

            # Execute delegated task and handle response
            async for data in delegate_agent.run(task_description=goal, context=delegation_context):
                if isinstance(data, StreamData):
                    yield data
                else:
                    yield StreamData(type="tool_response", content=str(data))

        except Exception as e:
            error_msg = f"Error during delegation to agent '{agent_name}' for session '{context.get('session_id', 'unknown')}': {str(e)}"
            self.logger.error(error_msg)
            yield StreamData(type="error", content=error_msg)

    def _parse_llm_response(self, response_str: str, session_id: str) -> Optional[OrchestratorLLMResponse]:
        """Parse the LLM response into a structured format."""
        try:
            # Try to parse the raw JSON
            response_data = safe_json_loads(response_str)
            if not response_data:
                self.logger.error(f"Empty or invalid JSON response for session '{session_id}'")
                return None

            # Validate and convert to our response model
            parsed_response = OrchestratorLLMResponse(**response_data)
            return parsed_response

        except Exception as e:
            self.logger.error(f"Failed to parse LLM response for session '{session_id}': {e}")
            return None

    def _build_llm_prompt(self, goal: str, conversation_history: List[str], previous_steps: List[Dict[str, Any]]) -> str:
        """Build the prompt for the LLM with current context."""
        # Start with the goal
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
        
        # Add available tools/agents info
        if self.delegation_targets:
            prompt += "Available Agents:\n"
            prompt += "\n".join([f"- {agent}" for agent in self.delegation_targets]) + "\n"
        
        prompt += "\nWhat would you like to do next?"
        return prompt

    @log_async_execution_time(logging.getLogger("WITS.orchestrator"))
    async def run(self, user_goal: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[StreamData, None]:
        """Run a goal through the orchestrator."""
        context = context or {}
        session_id = context.get("session_id", str(uuid.uuid4()))
        conversation_history = context.get("conversation_history", [])

        self.logger.info(f"Orchestrator starting for session '{session_id}'. Goal: {user_goal}")

        try:
            previous_steps: List[Dict[str, Any]] = []

            # ReAct loop
            for i in range(self.max_iterations):
                yield StreamData(
                    type="info", 
                    content=f"Orchestrator iteration {i+1}/{self.max_iterations} for session '{session_id}'.",
                    iteration=i+1, 
                    max_iterations=self.max_iterations
                )

                # Get LLM response
                prompt = self._build_llm_prompt(user_goal, conversation_history, previous_steps)
                llm_response = await self.llm.chat_completion_async(
                    model_name=self.orchestrator_model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                llm_response_content = llm_response.get('response', '')
                self.logger.info(f"Got LLM response for session '{session_id}' \\o/")

                # Parse response and take action
                parsed_response = self._parse_llm_response(llm_response_content, session_id)
                if not parsed_response:
                    yield StreamData(type="error", content="Failed to parse LLM response")
                    continue

                # Record this step
                current_step = {
                    "thought": parsed_response.thought_process,
                    "action": parsed_response.chosen_action,
                    "observation": None  # Will be set after executing action
                }

                # Execute the chosen action
                if parsed_response.chosen_action.action_type == "final_answer":
                    yield StreamData(
                        type="final_answer",
                        content=parsed_response.chosen_action.final_answer
                    )
                    return
                elif parsed_response.chosen_action.action_type == "tool_call":
                    tool_call = parsed_response.chosen_action.tool_call
                    if not tool_call or not tool_call.tool_name:
                        yield StreamData(type="error", content="Invalid tool call")
                        continue

                    # Handle delegation
                    async for delegation_result in self._handle_agent_delegation(
                        tool_call.tool_name,
                        tool_call.arguments.get("goal", ""),
                        context
                    ):
                        yield delegation_result
                        if isinstance(delegation_result, StreamData):
                            current_step["observation"] = delegation_result.content

                previous_steps.append(current_step)

            # Max iterations reached
            yield StreamData(
                type="error",
                content=f"Hit maximum iterations ({self.max_iterations}) without completing the task"
            )

        except Exception as e:
            self.logger.exception(f"Critical error in run method: {e}")
            yield StreamData(
                type="error",
                content=f"A critical error occurred: {str(e)}",
                error_details=str(e)
            )
