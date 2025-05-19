# agents/orchestrator_agent.py
import asyncio
import json
import logging
import time
from typing import AsyncGenerator, List, Dict, Any, Optional, Union

from agents.base_agent import BaseAgent
from core.config import AppConfig
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager
from core.schemas import LLMToolCall, OrchestratorLLMResponse, OrchestratorThought, OrchestratorAction, MemorySegment, MemorySegmentContent, StreamData
from core.debug_utils import log_async_execution_time

class OrchestratorAgent(BaseAgent):
    def __init__(self,
                 agent_name: str,
                 config: AppConfig,
                 llm_interface: LLMInterface,
                 memory_manager: MemoryManager,
                 delegation_targets: Dict[str, BaseAgent],
                 max_iterations: int = 10):
        super().__init__(agent_name, config, llm_interface, memory_manager)
        self.delegation_targets = delegation_targets
        self.max_iterations = max_iterations
        self.logger = logging.getLogger(f"WITS.{self.__class__.__name__}")
        self.final_answer_tool_name = "FinalAnswerTool"
        self.orchestrator_model_name = getattr(config.models, 'orchestrator', config.models.default)
        self.logger.info(f"OrchestratorAgent initialized. Model: {self.orchestrator_model_name}, Max iterations: {self.max_iterations}")
        self.logger.info(f"Delegation targets: {list(self.delegation_targets.keys())}")
        self.logger.info(f"Final answer tool configured as: {self.final_answer_tool_name}")

    def _get_agent_descriptions_for_prompt(self) -> str:
        descriptions = []
        for name, agent in self.delegation_targets.items():
            description = agent.get_description() if hasattr(agent, 'get_description') and callable(agent.get_description) else f"No specific description available for {name}."
            descriptions.append(f'- Agent Name: "{name}"\\n  Description: {description}')
        
        descriptions.append(f'- Tool Name: "{self.final_answer_tool_name}"\\n  Description: Use this tool to provide the final answer directly to the user when the goal has been fully achieved and no more actions are needed.')
        return "\\n".join(descriptions)

    def _build_llm_prompt(self, user_goal: str, conversation_history: List[Dict[str, str]], previous_steps: List[Dict[str, Any]]) -> str:
        history_str = "\\n".join([f"{turn['role']}: {turn['content']}" for turn in conversation_history])
        
        previous_steps_str = "No previous actions taken in this orchestration cycle."
        if previous_steps:
            formatted_steps = []
            for i, step in enumerate(previous_steps):
                thought_obj = step.get('thought') # OrchestratorThought object
                action_obj = step.get('action')   # OrchestratorAction object
                observation_text = step.get('observation', "N/A")

                thought_text = thought_obj.thought if thought_obj and hasattr(thought_obj, 'thought') else "N/A"
                action_details = "N/A"

                if action_obj and hasattr(action_obj, 'action_type'):
                    if action_obj.action_type == 'tool_call' and action_obj.tool_call:
                        action_details = f"Tool \'{action_obj.tool_call.tool_name}\' with args: {json.dumps(action_obj.tool_call.arguments)}"
                    elif action_obj.action_type == 'final_answer':
                        action_details = f"Final Answer: {action_obj.final_answer}"
                    elif action_obj.action_type == 'delegate_to_agent' and action_obj.delegate_to_agent_key:
                        action_details = f"Delegate to \'{action_obj.delegate_to_agent_key}\': {action_obj.delegated_task_description}"

                formatted_steps.append(f"Step {i+1}:")
                formatted_steps.append(f"  Thought: {thought_text}")
                formatted_steps.append(f"  Action Taken: {action_details}")
                formatted_steps.append(f"  Observation: {observation_text}")
            previous_steps_str = "\\n".join(formatted_steps)

        agent_descriptions = self._get_agent_descriptions_for_prompt()
        output_format_instruction = f"""\\
You MUST respond in a single, valid JSON object. Do not add any text before or after the JSON object.
The JSON object should have two main keys: "thought_process" and "chosen_action".

1.  "thought_process": This key should contain a JSON object detailing your thought process. This object MUST have a "thought" string field. Optionally, include "reasoning" (string) and "plan" (list of strings).
    Example: {{"thought": "User needs X. ResearchAgent is best.", "reasoning": "ResearchAgent has access to web.", "plan": ["Call ResearchAgent", "Summarize"]}}

2.  "chosen_action": This key should contain a JSON object representing the action you\'ve decided to take. This action object MUST have an "action_type" string field. Based on "action_type", other fields are required:
    *   If "action_type" is "tool_call":
        It MUST include a "tool_call" object with "tool_name" (string) and "arguments" (object).
        The "tool_name" must be one of the specialized agents (e.g., "ResearchAgent", "CodingAgent").
        The "arguments" for these agents MUST contain a "goal" key with a string value.
        Example: {{"action_type": "tool_call", "tool_call": {{"tool_name": "ResearchAgent", "arguments": {{"goal": "Find recent studies on X."}}}}}}
    *   If "action_type" is "final_answer":
        It MUST include a "final_answer" string field with the comprehensive response.
        Example: {{"action_type": "final_answer", "final_answer": "Based on studies, X is Y."}}

Example of the full JSON output for delegating to an agent (as a tool_call):
```json
{{
  "thought_process": {{
    "thought": "The user wants to know X. The ResearchAgent is best suited for this. I will formulate a specific goal for it.",
    "reasoning": "ResearchAgent can access external information.",
    "plan": ["Delegate to ResearchAgent", "Format the response", "Provide final answer"]
  }},
  "chosen_action": {{
    "action_type": "tool_call",
    "tool_call": {{
      "tool_name": "ResearchAgent",
      "arguments": {{
        "goal": "Find and summarize recent studies on topic X published in the last year."
      }}
    }}
  }}
}}
```

Example of the full JSON output for providing a final answer:
```json
{{
  "thought_process": {{
    "thought": "I have gathered all necessary information and can now provide a complete answer.",
    "reasoning": "All sub-tasks completed successfully."
  }},
  "chosen_action": {{
    "action_type": "final_answer",
    "final_answer": "Based on recent studies, the findings on topic X are Y and Z."
  }}
}}
```
Ensure your entire response is ONLY the single JSON object described.
"""
        prompt = f"""\\
You are the Orchestrator Agent. Your primary role is to understand a user\'s goal and achieve it by strategically delegating tasks to specialized agents or by providing a final answer once the goal is met. You operate in a cycle of thought, action, and observation.

Your available tools (specialized agents and the FinalAnswerTool) are:
{agent_descriptions}

User\'s Goal:
{user_goal}

Conversation History (User and Assistant interactions before this orchestration process):
{history_str if history_str else "No prior conversation history provided for this goal."}

Previous Steps in this Orchestration Cycle (Your previous thoughts, actions, and their outcomes):
{previous_steps_str}

Current Task:
Based on the User\'s Goal, Conversation History, and Previous Steps, decide the next action.
If the goal can be fully addressed with the information from previous steps, use the "{self.final_answer_tool_name}" by setting action_type to "final_answer".
Otherwise, delegate to the most appropriate agent by setting action_type to "tool_call", specifying its name as tool_name, and a clear, actionable goal in arguments.
If you are unsure or stuck, you can also use the "{self.final_answer_tool_name}" (action_type: "final_answer") to explain the situation.

Response Format Instructions:
{output_format_instruction}

JSON Response:
"""
        return prompt

    def _parse_llm_response(self, llm_response_str: str, session_id: str) -> Optional[OrchestratorLLMResponse]:
        try:
            self.logger.debug(f"Attempting to parse LLM response for session \'{session_id}\': {llm_response_str}")
            
            if llm_response_str.startswith("```json"):
                llm_response_str = llm_response_str[len("```json"):].strip()
            elif llm_response_str.startswith("```"):
                llm_response_str = llm_response_str[len("```"):].strip()
            
            if llm_response_str.endswith("```"):
                llm_response_str = llm_response_str[:-len("```")].strip()

            llm_json = json.loads(llm_response_str)
            
            thought_data = llm_json.get("thought_process")
            action_data = llm_json.get("chosen_action")

            if not thought_data or not isinstance(thought_data, dict) or "thought" not in thought_data:
                self.logger.error(f"Invalid or missing \'thought_process\' structure in LLM response for session \'{session_id}\': {thought_data}")
                return None
            
            thought = OrchestratorThought(
                thought=thought_data["thought"],
                reasoning=thought_data.get("reasoning"),
                plan=thought_data.get("plan")
            )

            if not action_data or not isinstance(action_data, dict) or "action_type" not in action_data:
                self.logger.error(f"Invalid or missing \'chosen_action\' structure in LLM response for session \'{session_id}\': {action_data}")
                return None

            action_type = action_data["action_type"]
            action_obj = None # Renamed to avoid conflict with outer scope variable name 'action'

            if action_type == "tool_call":
                tool_call_data = action_data.get("tool_call")
                if not tool_call_data or "tool_name" not in tool_call_data or "arguments" not in tool_call_data:
                    self.logger.error(f"Invalid \'tool_call\' data for action_type \'tool_call\' in session \'{session_id}\': {tool_call_data}")
                    return None
                
                tool_name = tool_call_data["tool_name"]
                tool_args = tool_call_data["arguments"]

                if tool_name != self.final_answer_tool_name and tool_name not in self.delegation_targets:
                    self.logger.error(f"LLM chose an unknown tool/agent for session \'{session_id}\': {tool_name}")
                    return None
                if tool_name in self.delegation_targets and "goal" not in tool_args:
                     self.logger.error(f"Agent \'{tool_name}\' chosen by LLM for session \'{session_id}\', but \'goal\' missing in args: {tool_args}")
                     return None

                action_obj = OrchestratorAction(
                    action_type=action_type,
                    tool_call=LLMToolCall(tool_name=tool_name, arguments=tool_args)
                )
            elif action_type == "final_answer":
                final_answer_text = action_data.get("final_answer")
                if final_answer_text is None:
                    self.logger.error(f"\'final_answer\' action_type chosen by LLM for session \'{session_id}\', but \'final_answer\' text missing: {action_data}")
                    return None
                action_obj = OrchestratorAction(action_type=action_type, final_answer=final_answer_text)
            else:
                self.logger.error(f"Unknown action_type \'{action_type}\' in LLM response for session \'{session_id}\'.")
                return None
            
            parsed_response = OrchestratorLLMResponse(thought_process=thought, chosen_action=action_obj)
            self.logger.info(f"LLM response successfully parsed for session \'{session_id}\'. Thought: \'{thought.thought[:100]}...\', Action Type: {action_obj.action_type}")
            return parsed_response

        except json.JSONDecodeError as e:
            self.logger.error(f"JSONDecodeError parsing LLM response for session \'{session_id}\': {e}. Response: {llm_response_str}")
            return None
        except Exception as e:
            self.logger.exception(f"Unexpected error parsing LLM response for session \'{session_id}\': {e}. Response: {llm_response_str}")
            return None

    @log_async_execution_time(logging.getLogger(f"WITS.OrchestratorAgent._handle_agent_delegation"))
    async def _handle_agent_delegation(self, agent_name: str, goal: str, context: Dict[str, Any]) -> AsyncGenerator[StreamData, None]:
        session_id = context.get("session_id", f"orch_delegate_fallback_{time.time_ns()}")
        if agent_name in self.delegation_targets:
            delegate_agent = self.delegation_targets[agent_name]
            self.logger.info(f"Delegating goal to agent \\'{agent_name}\\' for session \\'{session_id}\\\': {goal}")
            yield StreamData(type="info", content=f"Delegating to {agent_name} with goal: {goal}", tool_name=agent_name, tool_args={"goal": goal}, iteration=context.get("delegator_iteration"))
            
            delegation_context = context.copy()
            delegation_context["session_id"] = session_id 
            delegation_context["delegator_agent"] = self.agent_name

            full_response_content = ""
            try:
                # First, await the agent's run method to get its result.
                # The result might be an AsyncGenerator or a direct value (e.g., StreamData, string).
                agent_run_output = await delegate_agent.run(user_input_or_task=goal, context=delegation_context)
                
                last_yielded_stream_data: Optional[StreamData] = None

                if isinstance(agent_run_output, AsyncGenerator):
                    async for response_chunk in agent_run_output:
                        # Ensure iteration is passed through if available from sub-agent
                        if response_chunk.iteration is None:
                            response_chunk.iteration = context.get("delegator_iteration")
                        yield response_chunk 
                        last_yielded_stream_data = response_chunk
                        if response_chunk.type == "final_answer" or response_chunk.type == "content" or response_chunk.type == "tool_response":
                            if isinstance(response_chunk.content, str):
                                full_response_content += response_chunk.content + "\\\\n"
                            elif isinstance(response_chunk.content, dict):
                                 full_response_content += json.dumps(response_chunk.content) + "\\\\n"
                # Handle cases where the agent returns a single StreamData or other direct value
                elif isinstance(agent_run_output, StreamData):
                    yield agent_run_output
                    last_yielded_stream_data = agent_run_output
                    if agent_run_output.content and isinstance(agent_run_output.content, str):
                        full_response_content = agent_run_output.content
                    elif agent_run_output.content:
                        full_response_content = json.dumps(agent_run_output.content)
                elif isinstance(agent_run_output, str):
                    full_response_content = agent_run_output
                    # Add reasoning and plan if available from context (though unlikely for direct string return)
                    stream_data_out = StreamData(type="content", content=agent_run_output, tool_name=agent_name, iteration=context.get("delegator_iteration"))
                    yield stream_data_out
                    last_yielded_stream_data = stream_data_out
                elif agent_run_output is not None: # Catch other non-None, non-AsyncGenerator, non-StreamData results
                    full_response_content = json.dumps(agent_run_output)
                    stream_data_out = StreamData(type="content", content=full_response_content, tool_name=agent_name, iteration=context.get("delegator_iteration"))
                    yield stream_data_out
                    last_yielded_stream_data = stream_data_out

                trimmed_response = full_response_content.strip()
                if not trimmed_response:
                    trimmed_response = f"Agent {agent_name} completed its task but returned no explicit content."
                    self.logger.warning(f"Agent {agent_name} returned no explicit content for session \\'{session_id}\\' on goal: {goal}")
                
                # Ensure a tool_response is yielded to signal completion of delegation to the run loop
                # This might be redundant if the sub-agent already yielded a final_answer or tool_response type StreamData
                # However, it's a safeguard.
                # Check if the last yielded item from an async generator was already a tool_response or final_answer.
                # This check is tricky here, so we might yield an extra "tool_response" in some cases.
                # For simplicity, we'll yield it. The run loop can filter if needed.
                if not last_yielded_stream_data or last_yielded_stream_data.type not in ["tool_response", "final_answer"]:
                    yield StreamData(type="tool_response", content=trimmed_response, tool_name=agent_name, iteration=context.get("delegator_iteration"))
                self.logger.info(f"Agent \\'{agent_name}\\' completed for session \\'{session_id}\\\'. Accumulated response length: {len(trimmed_response)}")

            except Exception as e:
                self.logger.exception(f"Error during delegation to agent \\'{agent_name}\\' for session \\'{session_id}\\\': {e}")
                yield StreamData(type="tool_response", content=f"Error executing {agent_name}: {str(e)}", tool_name=agent_name, error_details=str(e), iteration=context.get("delegator_iteration"))
        else:
            self.logger.error(f"Attempted to delegate to unknown agent \\'{agent_name}\\' for session \\'{session_id}\\\'.")
            yield StreamData(type="tool_response", content=f"Error: Agent {agent_name} not found.", tool_name=agent_name, error_details=f"Agent {agent_name} not found.", iteration=context.get("delegator_iteration"))

    @log_async_execution_time(logging.getLogger(f"WITS.OrchestratorAgent.run"))
    async def run(self, user_goal: str, context: Dict[str, Any]) -> AsyncGenerator[StreamData, None]:
        session_id = context.get("session_id", f"orch_run_fallback_{time.time_ns()}")
        self.logger.info(f"Orchestrator starting for session \'{session_id}\'. Goal: {user_goal}")

        init_content = MemorySegmentContent(text=f"Orchestrator initialized with goal: {user_goal}")
        await self.memory.add_memory_segment(MemorySegment(
            type="ORCHESTRATOR_INIT",
            source=self.agent_name,
            content=init_content,
            metadata={"session_id": session_id, "user_goal": user_goal, "timestamp": time.time()}
        ))

        conversation_history = context.get("conversation_history", [])
        previous_steps: List[Dict[str, Any]] = []

        for i in range(self.max_iterations):
            yield StreamData(type="info", content=f"Orchestrator iteration {i+1}/{self.max_iterations} for session \'{session_id}\'.", iteration=i+1, max_iterations=self.max_iterations)
            self.logger.info(f"Orchestrator iteration {i+1}/{self.max_iterations} for session \'{session_id}\'.")

            prompt = self._build_llm_prompt(user_goal, conversation_history, previous_steps)
            
            prompt_log_content = MemorySegmentContent(text=f"LLM Prompt for iteration {i+1} (see debug for full prompt)")
            await self.memory.add_memory_segment(MemorySegment(
                type="ORCHESTRATOR_LLM_PROMPT",
                source=self.agent_name,
                content=prompt_log_content,
                metadata={"session_id": session_id, "iteration": i + 1, "timestamp": time.time()}
            ))
            self.logger.debug(f"Orchestrator LLM Prompt for session \'{session_id}\', iteration {i+1}:\\n{prompt}")

            try:
                llm_response_str = await self.llm.chat_completion_async(
                    model_name=self.orchestrator_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config_full.default_temperature,
                )
            except Exception as e:
                self.logger.exception(f"LLM call failed for session \\'{session_id}\\' on iteration {i+1}: {e}")
                yield StreamData(type="error", content=f"Orchestrator failed to get a response from LLM: {str(e)}", error_details=str(e), iteration=i+1)
                error_content = MemorySegmentContent(text=f"LLM call failed: {str(e)}", tool_output=f"Prompt was: {prompt[:200]}...")
                await self.memory.add_memory_segment(MemorySegment(
                    type="ORCHESTRATOR_LLM_ERROR",
                    source=self.agent_name,
                    content=error_content,
                    metadata={"session_id": session_id, "iteration": i + 1, "timestamp": time.time()}
                ))
                return

            self.logger.debug(f"Orchestrator LLM Raw Response for session \'{session_id}\', iteration {i+1}: {llm_response_str}")

            llm_response_log_content = MemorySegmentContent(
                text=f"LLM Raw Response: {llm_response_str[:200]}...",
                tool_output=llm_response_str
            )
            await self.memory.add_memory_segment(MemorySegment(
                type="ORCHESTRATOR_LLM_RESPONSE_RAW",
                source=self.agent_name,
                content=llm_response_log_content,
                metadata={"session_id": session_id, "iteration": i + 1, "timestamp": time.time(), "llm_model": self.orchestrator_model_name}
            ))

            parsed_llm_response = self._parse_llm_response(llm_response_str, session_id)

            if not parsed_llm_response:
                yield StreamData(type="error", content="Orchestrator failed to parse LLM response. Ending current attempt.", error_details=f"Raw response: {llm_response_str[:200]}", iteration=i+1) # Added iteration
                self.logger.error(f"Failed to parse LLM response for session \\'{session_id}\\' on iteration {i+1}. Raw response: {llm_response_str}")
                parse_error_content = MemorySegmentContent(
                    text="Failed to parse LLM response.", 
                    tool_output=llm_response_str
                )
                await self.memory.add_memory_segment(MemorySegment(
                    type="ORCHESTRATOR_PARSE_ERROR",
                    source=self.agent_name,
                    content=parse_error_content,
                    metadata={"session_id": session_id, "iteration": i + 1, "timestamp": time.time()}
                ))
                yield StreamData(type="final_answer", content="I encountered an issue processing the information. Please try rephrasing your request or try again later.", iteration=i+1)
                return

            current_thought = parsed_llm_response.thought_process
            current_action = parsed_llm_response.chosen_action
            
            yield StreamData(type="orchestrator_thought", content=current_thought.thought, reasoning=current_thought.reasoning, plan=current_thought.plan, iteration=i+1) # Added iteration
            self.logger.info(f"Orchestrator Thought for session \\'{session_id}\\\', iteration {i+1}: {current_thought.thought}")

            thought_action_text = f"Thought: {current_thought.thought}\\\\nAction Type: {current_action.action_type}"
            log_tool_name = None
            log_tool_args = None
            if current_action.action_type == "tool_call" and current_action.tool_call:
                log_tool_name = current_action.tool_call.tool_name
                log_tool_args = current_action.tool_call.arguments
                thought_action_text += f"\\nTool: {log_tool_name}\\nArgs: {json.dumps(log_tool_args)}"
            elif current_action.action_type == "final_answer":
                thought_action_text += f"\\nFinal Answer: {current_action.final_answer}"
                log_tool_name = self.final_answer_tool_name

            thought_action_content = MemorySegmentContent(
                text=thought_action_text,
                tool_name=log_tool_name, 
                tool_args=log_tool_args
            )
            await self.memory.add_memory_segment(MemorySegment(
                type="ORCHESTRATOR_THOUGHT_ACTION",
                source=self.agent_name,
                content=thought_action_content,
                metadata={"session_id": session_id, "iteration": i + 1, "timestamp": time.time()}
            ))

            observation = ""
            if current_action.action_type == "final_answer":
                answer = current_action.final_answer if current_action.final_answer is not None else "No answer provided."
                self.logger.info(f"Orchestrator decided for final_answer for session \\'{session_id}\\'. Answer: {answer}")
                yield StreamData(type="final_answer", content=answer, iteration=i+1)
                
                final_answer_log_content = MemorySegmentContent(
                    text=f"Final Answer provided: {answer}",
                    tool_name=self.final_answer_tool_name,
                    tool_args={"answer": answer}
                )
                await self.memory.add_memory_segment(MemorySegment(
                    type="ORCHESTRATOR_FINAL_ANSWER",
                    source=self.agent_name,
                    content=final_answer_log_content,
                    metadata={"session_id": session_id, "iteration": i + 1, "timestamp": time.time()}
                ))
                return 

            active_tool_name = None # For logging observation
            if current_action.action_type == "tool_call" and current_action.tool_call:
                active_tool_name = current_action.tool_call.tool_name
                tool_args_dict = current_action.tool_call.arguments
                sub_goal = tool_args_dict.get("goal", "")

                if not sub_goal:
                    self.logger.error(f"Orchestrator attempted to delegate to {active_tool_name} for session \'{session_id}\' but no \'goal\' was specified in args: {tool_args_dict}")
                    observation = f"Error: No goal specified for agent {active_tool_name}."
                    yield StreamData(type="error", content=observation, error_details=observation, tool_name=active_tool_name, iteration=i+1) # Added iteration and tool_name
                else:
                    yield StreamData(type="info", content=f"Delegating to {active_tool_name}...", tool_name=active_tool_name, tool_args=tool_args_dict, iteration=i+1)
                    
                    delegation_log_content = MemorySegmentContent(
                        text=f"Delegating to agent: {active_tool_name} with goal: {sub_goal}",
                        tool_name=active_tool_name,
                        tool_args=tool_args_dict
                    )
                    await self.memory.add_memory_segment(MemorySegment(
                        type="ORCHESTRATOR_DELEGATION_OUT",
                        source=self.agent_name,
                        content=delegation_log_content,
                        metadata={"session_id": session_id, "iteration": i + 1, "timestamp": time.time()}
                    ))

                    delegation_context = context.copy() 
                    delegation_context["delegator_iteration"] = i + 1

                    accumulated_agent_response = ""
                    async for agent_response_chunk in self._handle_agent_delegation(active_tool_name, sub_goal, delegation_context):
                        yield agent_response_chunk
                        if agent_response_chunk.type == "tool_response": 
                            observation = agent_response_chunk.content if isinstance(agent_response_chunk.content, str) else json.dumps(agent_response_chunk.content)
                            self.logger.info(f"Observation from {active_tool_name} for session \'{session_id}\' (iteration {i+1}): {observation[:200]}...")
                        elif agent_response_chunk.type == "content" or agent_response_chunk.type == "final_answer":
                             if isinstance(agent_response_chunk.content, str):
                                accumulated_agent_response += agent_response_chunk.content + "\\n"
                             elif isinstance(agent_response_chunk.content, dict):
                                accumulated_agent_response += json.dumps(agent_response_chunk.content) + "\\n"
                    
                    if not observation and accumulated_agent_response:
                        observation = accumulated_agent_response.strip()
                    elif not observation and not accumulated_agent_response:
                        observation = f"Agent {active_tool_name} completed but provided no specific output."
                        self.logger.warning(f"Agent {active_tool_name} provided no specific output for session \'{session_id}\' on iteration {i+1}")
            else: 
                observation = f"Error: Invalid action type \\'{current_action.action_type}\\' received for execution."
                self.logger.error(f"Invalid action type \\'{current_action.action_type}\\' in main loop for session \\'{session_id}\\'. This indicates a logic error in parsing or action definition.")
                yield StreamData(type="error", content=observation, error_details=observation, iteration=i+1) # Added iteration

            # Corrected conditional access for active_tool_name in log content
            observation_log_text = f"Observation from {active_tool_name if active_tool_name else 'N/A'}: {observation[:500]}..."
            observation_log_content = MemorySegmentContent(
                text=observation_log_text,
                tool_name=active_tool_name, # Will be None if not a tool_call
                tool_output=observation
            )
            await self.memory.add_memory_segment(MemorySegment(
                type="ORCHESTRATOR_OBSERVATION",
                source=self.agent_name, 
                content=observation_log_content,
                metadata={"session_id": session_id, "iteration": i + 1, "tool_used": active_tool_name, "timestamp": time.time()}
            ))
            
            previous_steps.append({
                "thought": current_thought,
                "action": current_action,
                "observation": observation
            })
            yield StreamData(type="orchestrator_observation", content=observation, tool_name=active_tool_name, iteration=i+1) # Added iteration

        self.logger.warning(f"Orchestrator reached max iterations ({self.max_iterations}) for session \\'{session_id}\\' without reaching a final answer for goal: {user_goal}")
        yield StreamData(type="info", content="Orchestrator reached maximum iterations.", iteration=self.max_iterations) # Added iteration
        max_iter_content = MemorySegmentContent(text=f"Reached max iterations ({self.max_iterations}) for goal: {user_goal}")
        await self.memory.add_memory_segment(MemorySegment(
            type="ORCHESTRATOR_MAX_ITERATIONS",
            source=self.agent_name,
            content=max_iter_content,
            metadata={"session_id": session_id, "user_goal": user_goal, "timestamp": time.time()}
        ))
        yield StreamData(type="final_answer", content="I\\'ve made several attempts but couldn\\'t complete your request. You might want to try rephrasing or breaking it down.", iteration=self.max_iterations) # Added iteration

    def get_description(self) -> str:
        return "This agent orchestrates tasks by delegating to specialized agents or providing a final answer. It manages the overall goal achievement."

    def get_available_tools(self) -> List[Dict[str, Any]]:
        tools = []
        for name, agent in self.delegation_targets.items():
            tools.append({
                "tool_name": name,
                "description": agent.get_description(), 
                "args_schema": { 
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": f"The specific goal for the {name} agent."}
                    },
                    "required": ["goal"]
                }
            })
        tools.append({
            "tool_name": self.final_answer_tool_name,
            "description": "Provides the final answer to the user when the goal is achieved.",
            "args_schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The final answer to the user\'s goal."}
                },
                "required": ["answer"]
            }
        })
        return tools