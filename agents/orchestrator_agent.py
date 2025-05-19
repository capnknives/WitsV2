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
        self.is_book_writing_mode = False # Added for book writing mode
        self.book_writing_state: Dict[str, Any] = {} # Added to store book writing state

    def _get_agent_descriptions_for_prompt(self) -> str:
        descriptions = []
        for name, agent in self.delegation_targets.items():
            description = agent.get_description() if hasattr(agent, 'get_description') and callable(agent.get_description) else f"No specific description available for {name}."
            descriptions.append(f'- Agent Name: "{name}"\\n  Description: {description}')
        
        descriptions.append(f'- Tool Name: "{self.final_answer_tool_name}"\\n  Description: Use this tool to provide the final answer directly to the user when the goal has been fully achieved and no more actions are needed.')
        return "\\n".join(descriptions)

    def _build_llm_prompt(self, user_goal: str, conversation_history: List[Dict[str, str]], previous_steps: List[Dict[str, Any]]) -> str:
        history_str = "\\n".join([f"{turn['role']}: {turn['content']}" for turn in conversation_history])
        
        # Determine if we are in book writing mode
        # Simple check based on keywords, can be made more sophisticated
        if "book" in user_goal.lower() or "novel" in user_goal.lower() or "story" in user_goal.lower():
            self.is_book_writing_mode = True
            self.logger.info(f"Detected book writing mode for goal: {user_goal}")
        else:
            # Reset if not a book-writing goal, or manage separate states per goal
            self.is_book_writing_mode = False


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
        
        book_writing_prompt_injection = ""
        if self.is_book_writing_mode:
            book_writing_prompt_injection = """\
You are currently in 'Book Writing Mode'.
The goal is to write a book. Decompose this goal into logical steps suitable for the following specialized book-writing agents:
- "book_plotter": For generating plot outlines, chapter structures, and story arcs.
- "book_character_dev": For developing character profiles, backstories, and motivations.
- "book_worldbuilder": For creating the setting, lore, and rules of the fictional world.
- "book_prose_generator": For writing the actual narrative, dialogue, and descriptions.
- "book_editor": For reviewing, editing, and refining the generated prose.

When delegating to these agents, provide highly specific task descriptions. For example:
- To PlotterAgent: 'Generate a 3-act plot outline for a sci-fi mystery novel set on Mars.'
- To CharacterDevelopmentAgent: 'Develop a character sheet for the protagonist, a cynical detective, including their fears, desires, and a key childhood trauma.'
- To WorldBuilderAgent: 'Describe the political system and major factions of the elven kingdom of Eldoria.'
- To ProseGenerationAgent: 'Write Chapter 1 (approx. 1500 words) based on the approved outline, focusing on introducing the main character and the inciting incident. Current outline: [Outline Snippet]. Character sheet: [Character Snippet].'
- To EditorAgent: 'Review Chapter 5 for plot consistency with the overall outline and check for repetitive phrasing. Current book state: [Relevant Book State].'

Remember to manage the overall state of the book (outline, character sheets, generated chapters) by retrieving and updating this information via the MemoryManager. You can instruct other agents to save their outputs to memory.
Current Book State (from MemoryManager, if available):
Outline: {self.book_writing_state.get('outline', 'Not yet defined.')}
Characters: {json.dumps(self.book_writing_state.get('characters', {}), indent=2)}
World Details: {json.dumps(self.book_writing_state.get('world_details', {}), indent=2)}
Generated Chapters: {list(self.book_writing_state.get('chapters', {}).keys())}
"""

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
You are the Orchestrator Agent. Your primary role is to understand a user's goal and achieve it by strategically delegating tasks to specialized agents or by providing a final answer once the goal is met. You operate in a cycle of thought, action, and observation.

{book_writing_prompt_injection}

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
            self.logger.info(f"Delegating goal to agent '{agent_name}' for session '{session_id}': {goal}")
            
            # Pass book writing state to specialized agents if in book writing mode
            delegation_context = context.copy()
            if self.is_book_writing_mode:
                delegation_context["book_writing_state"] = self.book_writing_state
                self.logger.info(f"Passing book_writing_state to {agent_name}: {list(self.book_writing_state.keys())}")

            yield StreamData(type="info", content=f"Delegating to {agent_name} with goal: {goal}", tool_name=agent_name, tool_args={"goal": goal}, iteration=context.get("delegator_iteration"))
            
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
        self.logger.info(f"Orchestrator starting for session '{session_id}'. Goal: {user_goal}")

        # Initialize or load book writing state if applicable
        if "book" in user_goal.lower() or "novel" in user_goal.lower() or "story" in user_goal.lower(): # Basic check
            self.is_book_writing_mode = True
            book_state_memory_key = f"book_state_{session_id}_{user_goal[:50].replace(' ', '_')}"
            self.logger.info(f"Book writing mode enabled. Attempting to load state with key: {book_state_memory_key}")
            try:
                # Assuming retrieve_memory_segment_by_key searches for a segment with metadata['key'] == book_state_memory_key
                # and type == "BOOK_WRITING_STATE", returning the most recent one.
                # This method was mentioned as "await self.memory.retrieve_memory_segment_by_key(memory_key)" in summary.
                if hasattr(self.memory, "retrieve_memory_segment_by_key"):
                    loaded_segment = await self.memory.retrieve_memory_segment_by_key(book_state_memory_key)
                else:
                    self.logger.warning("MemoryManager does not have 'retrieve_memory_segment_by_key' method. Attempting search_memory.")
                    # Fallback to search_memory if retrieve_memory_segment_by_key is not available
                    # This requires knowing how search_memory works and what query to use.
                    # For now, assume it might return a list of segments or None.
                    # This is a placeholder for a more robust search query.
                    search_results = await self.memory.search_memory(
                        query_text=f"Retrieve book writing state for key {book_state_memory_key}",
                        limit=1,
                        filter_metadata={"key": book_state_memory_key, "type": "BOOK_WRITING_STATE"} # Hypothetical filter
                    )
                    loaded_segment = search_results[0] if search_results else None

                if loaded_segment and loaded_segment.content and loaded_segment.content.text:
                    self.book_writing_state = json.loads(loaded_segment.content.text)
                    self.logger.info(f"Loaded existing book writing state for key '{book_state_memory_key}': {list(self.book_writing_state.keys())}")
                else:
                    self.book_writing_state = {} # Initialize if not found
                    self.logger.info(f"No existing book state found for key '{book_state_memory_key}' (or content.text was empty), initializing empty state.")
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse loaded book state for key '{book_state_memory_key}': {e}. Initializing empty state.")
                self.book_writing_state = {}
            except AttributeError as e: # Handles if retrieve_memory_segment_by_key or search_memory is not found or behaves unexpectedly
                self.logger.error(f"Error accessing memory methods for book state loading (key '{book_state_memory_key}'): {e}. Initializing empty state.")
                self.book_writing_state = {}
            except Exception as e: 
                self.logger.error(f"Unexpected error loading book state for key '{book_state_memory_key}': {e}. Initializing empty state.")
                self.book_writing_state = {}
        else:
            self.is_book_writing_mode = False
            self.book_writing_state = {}


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
                    delegation_context["user_goal"] = user_goal # Pass user_goal for consistent memory key usage

                    accumulated_agent_response = ""
                    async for agent_response_chunk in self._handle_agent_delegation(active_tool_name, sub_goal, delegation_context):
                        yield agent_response_chunk
                        if agent_response_chunk.type == "tool_response": 
                            observation = agent_response_chunk.content if isinstance(agent_response_chunk.content, str) else json.dumps(agent_response_chunk.content)
                            self.logger.info(f"Observation from {active_tool_name} for session '{session_id}' (iteration {i+1}): {observation[:200]}...")
                        
                            # If in book writing mode, attempt to update book_writing_state from agent's output
                            if self.is_book_writing_mode and active_tool_name in ["book_plotter", "book_character_dev", "book_worldbuilder", "book_prose_generator", "book_editor"]:
                                try:
                                    agent_content_str = observation # observation is the string form of agent_response_chunk.content
                                    update_data_dict = None
                                    
                                    try:
                                        parsed_content = json.loads(agent_content_str)
                                        if isinstance(parsed_content, dict):
                                            update_data_dict = parsed_content
                                    except json.JSONDecodeError:
                                        self.logger.warning(f"Observation from {active_tool_name} is a string but not valid JSON: {agent_content_str[:200]}...")
                                    
                                    if update_data_dict and "update_book_state" in update_data_dict:
                                        update_payload = update_data_dict["update_book_state"]
                                        if isinstance(update_payload, dict):
                                            # Retrieve user_goal from delegation_context for consistent key generation
                                            current_user_goal_for_key = delegation_context.get("user_goal", "unknown_user_goal")
                                            
                                            for key, value in update_payload.items():
                                                if key in self.book_writing_state and isinstance(self.book_writing_state[key], dict) and isinstance(value, dict):
                                                    self.book_writing_state[key].update(value) # Merge dictionaries
                                                elif key in self.book_writing_state and isinstance(self.book_writing_state[key], list) and isinstance(value, list):
                                                    # For lists, decide on extend vs replace. Extend is often safer.
                                                    self.book_writing_state[key].extend(value) 
                                                else:
                                                    self.book_writing_state[key] = value # Replace or add new key
                                            self.logger.info(f"Book writing state updated by {active_tool_name} with keys: {list(update_payload.keys())}")
                                            
                                            # Persist updated book state to memory
                                            book_state_memory_key_for_save = f"book_state_{session_id}_{current_user_goal_for_key[:50].replace(' ', '_')}"
                                            state_json_string = json.dumps(self.book_writing_state)
                                            
                                            state_content_obj = MemorySegmentContent(
                                                text=state_json_string, 
                                                tool_output=f"Book state updated by {active_tool_name}." # Brief description for the segment
                                            )
                                            await self.memory.add_memory_segment(MemorySegment(
                                                type="BOOK_WRITING_STATE", 
                                                source=self.agent_name,
                                                content=state_content_obj,
                                                metadata={
                                                    "session_id": session_id, 
                                                    "user_goal_summary": current_user_goal_for_key[:50], 
                                                    "timestamp": time.time(), 
                                                    "key": book_state_memory_key_for_save # Store the key for retrieval
                                                }
                                            ))
                                            self.logger.info(f"Persisted updated book_writing_state to memory with key '{book_state_memory_key_for_save}'.")
                                        else:
                                            self.logger.warning(f"'update_book_state' payload from {active_tool_name} is not a dictionary: {update_payload}")
                                    # else: # Optional: log if no update_book_state key or not a dict
                                    #    self.logger.debug(f"No 'update_book_state' key found in {active_tool_name} response, or content not a parsable dict with it. Observation: {agent_content_str[:100]}")

                                except Exception as e:
                                    # Log the full exception for better debugging
                                    self.logger.exception(f"Error updating/persisting book writing state from {active_tool_name} response: {e}. Observation was: {agent_content_str[:200]}")
