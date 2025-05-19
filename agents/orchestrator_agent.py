# agents/orchestrator_agent.py
import asyncio
import json
import logging
import time
import re
from typing import AsyncGenerator, List, Dict, Any, Optional, Union
from pydantic import ValidationError # Added for handling Pydantic validation errors

from agents.base_agent import BaseAgent
from core.config import AppConfig
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager, MemorySegment
from core.schemas import LLMToolCall, OrchestratorLLMResponse, OrchestratorThought, OrchestratorAction, MemorySegmentContent, StreamData
from agents.book_writing_schemas import BookWritingState, ChapterOutlineSchema, CharacterProfileSchema, WorldAnvilSchema, ChapterProseSchema
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
        self.is_book_writing_mode = False
        self.book_writing_state: Optional[BookWritingState] = None
        self.current_project_name: Optional[str] = None

    def _extract_project_name(self, user_goal: str, context: Dict[str, Any]) -> Optional[str]:
        if "project_name" in context and context["project_name"]:
            self.logger.debug(f"Extracted project_name '{context['project_name']}' from context.")
            return str(context["project_name"])

        patterns = [
            r"(?:project named|book named|novel named|story named|project|book|novel|story)\\s+['\\\"]([^'\\\"]+)['\\\"]",
            r"create a new book titled\\s+['\\\"]([^'\\\"]+)['\\\"]",
            r"start book\\s+['\\\"]([^'\\\"]+)['\\\"]"
        ]
        for pattern in patterns:
            match = re.search(pattern, user_goal, re.IGNORECASE)
            if match:
                project_name = match.group(1).strip()
                if project_name:
                    self.logger.debug(f"Extracted project_name '{project_name}' from user_goal using pattern: {pattern}")
                    return project_name
        
        self.logger.debug(f"Could not extract project_name from user_goal: {user_goal} or context.")
        return None

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
                thought_obj = step.get('thought') 
                action_obj = step.get('action')   
                observation_text = step.get('observation', "N/A")

                thought_text = thought_obj.thought if thought_obj and hasattr(thought_obj, 'thought') else "N/A"
                action_details = "N/A"

                if action_obj and hasattr(action_obj, 'action_type'):
                    if action_obj.action_type == 'tool_call' and action_obj.tool_call:
                        action_details = f"Tool \\'{action_obj.tool_call.tool_name}\\' with args: {json.dumps(action_obj.tool_call.arguments)}"
                    elif action_obj.action_type == 'final_answer':
                        action_details = f"Final Answer: {action_obj.final_answer}"
                    elif action_obj.action_type == 'delegate_to_agent' and action_obj.delegate_to_agent_key:
                        action_details = f"Delegate to \\'{action_obj.delegate_to_agent_key}\\': {action_obj.delegated_task_description}"

                formatted_steps.append(f"Step {i+1}:")
                formatted_steps.append(f"  Thought: {thought_text}")
                formatted_steps.append(f"  Action Taken: {action_details}")
                formatted_steps.append(f"  Observation: {observation_text}")
            previous_steps_str = "\\n".join(formatted_steps)

        agent_descriptions = self._get_agent_descriptions_for_prompt()
        
        book_writing_prompt_injection = ""
        if self.is_book_writing_mode and self.book_writing_state:
            plot_summary_str = self.book_writing_state.overall_plot_summary or "Not yet defined."
            chapters_outlined_str = ", ".join([f"Ch {c.chapter_number}: {c.title or 'Untitled'}" for c in self.book_writing_state.detailed_chapter_outlines]) or "None"
            characters_str = ", ".join([p.name for p in self.book_writing_state.character_profiles]) or "None"
            world_notes_present = "Yes" if self.book_writing_state.world_building_notes and (self.book_writing_state.world_building_notes.locations or self.book_writing_state.world_building_notes.lore or self.book_writing_state.world_building_notes.rules) else "No"
            prose_generated_chapters = ", ".join(self.book_writing_state.generated_prose.keys()) or "None"
            revision_notes_str = self.book_writing_state.revision_notes or "None"
            style_guide_str = self.book_writing_state.writing_style_guide or "Not set."
            tone_guide_str = self.book_writing_state.tone_guide or "Not set."


            book_writing_prompt_injection = f"""\\
You are currently in 'Book Writing Mode' for the project '{self.book_writing_state.project_name}'.
The goal is to collaboratively write a book. You must manage and update a structured BookWritingState.
When delegating to specialized book-writing agents, provide them with relevant slices of the BookWritingState and instruct them to return structured JSON output that can be used to update specific fields in the BookWritingState.

Current BookWritingState Summary for project '{self.book_writing_state.project_name}':
- Overall Plot Summary: {plot_summary_str}
- Detailed Chapter Outlines: {chapters_outlined_str}
- Character Profiles: {characters_str}
- World Building Notes Present: {world_notes_present}
- Generated Prose for Chapters: {prose_generated_chapters}
- Revision Notes: {revision_notes_str}
- Writing Style Guide: {style_guide_str}
- Tone Guide: {tone_guide_str}

Specialized Agents and Expected Output for BookWritingState:
- "book_plotter": Manages plot and chapter outlines.
  - Input: May receive current plot summary, character profiles, existing outlines.
  - Output: Should return JSON like {{"detailed_chapter_outlines": [ChapterOutlineSchema_dict, ...]}} or {{"overall_plot_summary": "new summary"}}.
- "book_character_dev": Manages character profiles.
  - Input: May receive existing profiles, plot context.
  - Output: Should return JSON like {{"character_profiles": [CharacterProfileSchema_dict, ...]}}.
- "book_worldbuilder": Manages world-building notes.
  - Input: May receive existing world notes.
  - Output: Should return JSON like {{"world_building_notes": WorldAnvilSchema_dict}}.
- "book_prose_generator": Writes narrative content.
  - Input: Relevant chapter outlines, character profiles, world notes for the specific task.
  - Output: Should return JSON like {{"generated_prose_update": {{"chapter_X": ChapterProseSchema_dict}}}}. (chapter_X is the chapter number string key e.g. "1", "2")
- "book_editor": Reviews and refines prose or other content.
  - Input: Prose, outlines, style guides, tone guides.
  - Output: Can return revised prose (similar to prose_generator output, e.g. {{"generated_prose_update": {{"chapter_X": ChapterProseSchema_dict}}}}) or updated revision_notes {{"revision_notes": "new notes"}}.

Example delegation:
- To PlotterAgent: 'Based on the overall plot summary, generate detailed outlines for the first three chapters. Return as {{"detailed_chapter_outlines": [...]}}.'
- To CharacterDevelopmentAgent: 'Flesh out the antagonist profile based on the plot. Current antagonist profile: [relevant snippet from character_profiles]. Return as {{"character_profiles": [...]}}.'

Your task is to decide the next step to advance the book project based on the user's goal and the current BookWritingState.
"""
        elif self.is_book_writing_mode and not self.book_writing_state:
             book_writing_prompt_injection = """\\
You are currently in 'Book Writing Mode'. However, the BookWritingState could not be initialized (likely missing a project name).
Please ask the user to specify a project name (e.g., "Create a new book project named 'My Novel'") or load an existing project.
If the user provides a project name, your next step should be to re-initialize with that project name.
You can use the FinalAnswerTool to communicate this to the user.
"""

        output_format_instruction = f"""\\
You MUST respond in a single, valid JSON object. Do not add any text before or after the JSON object.
The JSON object should have two main keys: "thought_process" and "chosen_action".

1.  "thought_process": This key should contain a JSON object detailing your thought process. This object MUST have a "thought" string field. Optionally, include "reasoning" (string) and "plan" (list of strings).
    Example: {{"thought": "User needs X. ResearchAgent is best.", "reasoning": "ResearchAgent has access to web.", "plan": ["Call ResearchAgent", "Summarize"]}}

2.  "chosen_action": This key should contain a JSON object representing the action you\\'ve decided to take. This action object MUST have an "action_type" string field. Based on "action_type", other fields are required:
    *   If "action_type" is "tool_call":
        It MUST include a "tool_call" object with "tool_name" (string) and "arguments" (object).
        The "tool_name" must be one of the specialized agents (e.g., "ResearchAgent", "CodingAgent", "book_plotter").
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

User\\'s Goal:
{user_goal}

Conversation History (User and Assistant interactions before this orchestration process):
{history_str if history_str else "No prior conversation history provided for this goal."}

Previous Steps in this Orchestration Cycle (Your previous thoughts, actions, and their outcomes):
{previous_steps_str}

Current Task:
Based on the User\\'s Goal, Conversation History, and Previous Steps, decide the next action.
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
            self.logger.debug(f"Attempting to parse LLM response for session \\'{session_id}\\': {llm_response_str}")
            
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
                self.logger.error(f"Invalid or missing \\'thought_process\\' structure in LLM response for session \\'{session_id}\\': {thought_data}")
                return None
            
            thought = OrchestratorThought(
                thought=thought_data["thought"],
                reasoning=thought_data.get("reasoning"),
                plan=thought_data.get("plan")
            )

            if not action_data or not isinstance(action_data, dict) or "action_type" not in action_data:
                self.logger.error(f"Invalid or missing \\'chosen_action\\' structure in LLM response for session \\'{session_id}\\': {action_data}")
                return None

            action_type = action_data["action_type"]
            action_obj = None 

            if action_type == "tool_call":
                tool_call_data = action_data.get("tool_call")
                if not tool_call_data or "tool_name" not in tool_call_data or "arguments" not in tool_call_data:
                    self.logger.error(f"Invalid \\'tool_call\\' data for action_type \\'tool_call\\' in session \\'{session_id}\\': {tool_call_data}")
                    return None
                
                tool_name = tool_call_data["tool_name"]
                tool_args = tool_call_data["arguments"]

                if tool_name != self.final_answer_tool_name and tool_name not in self.delegation_targets:
                    self.logger.error(f"LLM chose an unknown tool/agent for session \\'{session_id}\\': {tool_name}")
                    return None
                if tool_name in self.delegation_targets and "goal" not in tool_args: # Specialized agents expect a 'goal'
                     self.logger.error(f"Agent \\'{tool_name}\\' chosen by LLM for session \\'{session_id}\\', but \\'goal\\' missing in args: {tool_args}")
                     # Allow this for now, but specialized agents might fail if 'goal' is strictly needed by their run method.
                     # Consider if this should be a hard return None or just a warning.
                     # For book writing agents, the 'goal' will be the specific task.

                action_obj = OrchestratorAction(
                    action_type=action_type,
                    tool_call=LLMToolCall(tool_name=tool_name, arguments=tool_args)
                )
            elif action_type == "final_answer":
                final_answer_text = action_data.get("final_answer")
                if final_answer_text is None: # Check for None, empty string is a valid answer
                    self.logger.error(f"\\'final_answer\\' action_type chosen by LLM for session \\'{session_id}\\', but \\'final_answer\\' text missing: {action_data}")
                    return None
                action_obj = OrchestratorAction(action_type=action_type, final_answer=final_answer_text)
            else:
                self.logger.error(f"Unknown action_type \\'{action_type}\\' in LLM response for session \\'{session_id}\\'.")
                return None
            
            parsed_response = OrchestratorLLMResponse(thought_process=thought, chosen_action=action_obj)
            self.logger.info(f"LLM response successfully parsed for session \\'{session_id}\\'. Thought: \\'{thought.thought[:100]}...\\', Action Type: {action_obj.action_type}")
            return parsed_response

        except json.JSONDecodeError as e:
            self.logger.error(f"JSONDecodeError parsing LLM response for session \\'{session_id}\\': {e}. Response: {llm_response_str}")
            return None
        except Exception as e:
            self.logger.exception(f"Unexpected error parsing LLM response for session \\'{session_id}\\': {e}. Response: {llm_response_str}")
            return None

    @log_async_execution_time(logging.getLogger(f"WITS.OrchestratorAgent._handle_agent_delegation"))
    async def _handle_agent_delegation(self, agent_name: str, goal: str, context: Dict[str, Any]) -> AsyncGenerator[StreamData, None]:
        session_id = context.get("session_id", f"orch_delegate_fallback_{time.time_ns()}")
        if agent_name in self.delegation_targets:
            delegate_agent = self.delegation_targets[agent_name]
            self.logger.info(f"Delegating goal to agent '{agent_name}' for session '{session_id}': {goal}")
            
            delegation_context = context.copy()
            if self.is_book_writing_mode and self.book_writing_state and agent_name.startswith("book_"):
                agent_specific_state_slice = {}
                if agent_name == "book_plotter":
                    agent_specific_state_slice["overall_plot_summary"] = self.book_writing_state.overall_plot_summary
                    agent_specific_state_slice["character_profiles"] = [p.model_dump() for p in self.book_writing_state.character_profiles]
                    agent_specific_state_slice["detailed_chapter_outlines"] = [o.model_dump() for o in self.book_writing_state.detailed_chapter_outlines]
                elif agent_name == "book_character_dev":
                    agent_specific_state_slice["overall_plot_summary"] = self.book_writing_state.overall_plot_summary
                    agent_specific_state_slice["character_profiles"] = [p.model_dump() for p in self.book_writing_state.character_profiles]
                elif agent_name == "book_worldbuilder":
                    if self.book_writing_state.world_building_notes:
                        agent_specific_state_slice["world_building_notes"] = self.book_writing_state.world_building_notes.model_dump()
                    else:
                        agent_specific_state_slice["world_building_notes"] = WorldAnvilSchema().model_dump() # Pass empty structure
                elif agent_name == "book_prose_generator":
                    agent_specific_state_slice["overall_plot_summary"] = self.book_writing_state.overall_plot_summary
                    agent_specific_state_slice["detailed_chapter_outlines"] = [o.model_dump() for o in self.book_writing_state.detailed_chapter_outlines]
                    agent_specific_state_slice["character_profiles"] = [p.model_dump() for p in self.book_writing_state.character_profiles]
                    if self.book_writing_state.world_building_notes:
                        agent_specific_state_slice["world_building_notes"] = self.book_writing_state.world_building_notes.model_dump()
                    if self.book_writing_state.writing_style_guide:
                         agent_specific_state_slice["writing_style_guide"] = self.book_writing_state.writing_style_guide
                    if self.book_writing_state.tone_guide:
                         agent_specific_state_slice["tone_guide"] = self.book_writing_state.tone_guide
                elif agent_name == "book_editor":
                    agent_specific_state_slice["generated_prose"] = {k: v.model_dump() for k, v in self.book_writing_state.generated_prose.items()}
                    agent_specific_state_slice["detailed_chapter_outlines"] = [o.model_dump() for o in self.book_writing_state.detailed_chapter_outlines]
                    agent_specific_state_slice["character_profiles"] = [p.model_dump() for p in self.book_writing_state.character_profiles]
                    if self.book_writing_state.world_building_notes:
                        agent_specific_state_slice["world_building_notes"] = self.book_writing_state.world_building_notes.model_dump()
                    if self.book_writing_state.writing_style_guide:
                         agent_specific_state_slice["writing_style_guide"] = self.book_writing_state.writing_style_guide
                    if self.book_writing_state.tone_guide:
                         agent_specific_state_slice["tone_guide"] = self.book_writing_state.tone_guide
                    agent_specific_state_slice["revision_notes"] = self.book_writing_state.revision_notes

                if agent_specific_state_slice: # Only add if not empty
                     delegation_context["book_writing_state_slice"] = agent_specific_state_slice # This is the key agents should look for
                     self.logger.info(f"Passing book_writing_state_slice to {agent_name} with keys: {list(agent_specific_state_slice.keys())}")
                else:
                    self.logger.info(f"No specific book_writing_state_slice prepared for {agent_name}, or agent is not a book agent.")

            yield StreamData(type="info", content=f"Delegating to {agent_name} with goal: {goal}", tool_name=agent_name, tool_args={"goal": goal}, iteration=context.get("delegator_iteration"))
            
            delegation_context["session_id"] = session_id 
            delegation_context["delegator_agent"] = self.agent_name

            full_response_content = ""
            try:
                agent_run_output = await delegate_agent.run(user_input_or_task=goal, context=delegation_context)
                
                last_yielded_stream_data: Optional[StreamData] = None

                if isinstance(agent_run_output, AsyncGenerator):
                    async for response_chunk in agent_run_output:
                        if response_chunk.iteration is None:
                            response_chunk.iteration = context.get("delegator_iteration")
                        yield response_chunk 
                        last_yielded_stream_data = response_chunk
                        if response_chunk.type == "final_answer" or response_chunk.type == "content" or response_chunk.type == "tool_response":
                            if isinstance(response_chunk.content, str):
                                full_response_content += response_chunk.content + "\\n"
                            elif isinstance(response_chunk.content, dict): # Agent might return structured JSON directly
                                 full_response_content += json.dumps(response_chunk.content) + "\\n"
                elif isinstance(agent_run_output, StreamData):
                    yield agent_run_output
                    last_yielded_stream_data = agent_run_output
                    if agent_run_output.content and isinstance(agent_run_output.content, str):
                        full_response_content = agent_run_output.content
                    elif agent_run_output.content: # Could be dict
                        full_response_content = json.dumps(agent_run_output.content)
                elif isinstance(agent_run_output, str): # Direct string return
                    full_response_content = agent_run_output
                    stream_data_out = StreamData(type="content", content=agent_run_output, tool_name=agent_name, iteration=context.get("delegator_iteration"))
                    yield stream_data_out
                    last_yielded_stream_data = stream_data_out
                elif agent_run_output is not None: 
                    full_response_content = json.dumps(agent_run_output) # Assume it's JSON serializable if not str/StreamData
                    stream_data_out = StreamData(type="content", content=full_response_content, tool_name=agent_name, iteration=context.get("delegator_iteration"))
                    yield stream_data_out
                    last_yielded_stream_data = stream_data_out

                trimmed_response = full_response_content.strip()
                if not trimmed_response:
                    trimmed_response = f"Agent {agent_name} completed its task but returned no explicit content."
                    self.logger.warning(f"Agent {agent_name} returned no explicit content for session \\'{session_id}\\' on goal: {goal}")
                
                if not last_yielded_stream_data or last_yielded_stream_data.type not in ["tool_response", "final_answer"]:
                    yield StreamData(type="tool_response", content=trimmed_response, tool_name=agent_name, iteration=context.get("delegator_iteration"))
                self.logger.info(f"Agent \\'{agent_name}\\' completed for session \\'{session_id}\\'. Accumulated response length: {len(trimmed_response)}")

            except Exception as e:
                self.logger.exception(f"Error during delegation to agent \\'{agent_name}\\' for session \\'{session_id}\\': {e}")
                yield StreamData(type="tool_response", content=f"Error executing {agent_name}: {str(e)}", tool_name=agent_name, error_details=str(e), iteration=context.get("delegator_iteration"))
        else:
            self.logger.error(f"Attempted to delegate to unknown agent \\'{agent_name}\\' for session \\'{session_id}\\'.")
            yield StreamData(type="tool_response", content=f"Error: Agent {agent_name} not found.", tool_name=agent_name, error_details=f"Agent {agent_name} not found.", iteration=context.get("delegator_iteration"))

    @log_async_execution_time(logging.getLogger(f"WITS.OrchestratorAgent.run"))
    async def run(self, user_goal: str, context: Dict[str, Any]) -> AsyncGenerator[StreamData, None]:
        session_id = context.get("session_id", f"orch_run_fallback_{time.time_ns()}")
        self.logger.info(f"Orchestrator starting for session '{session_id}'. Goal: {user_goal}")

        # Determine book writing mode and project name early
        if "book" in user_goal.lower() or "novel" in user_goal.lower() or "story" in user_goal.lower() or ("project_name" in context and context.get("is_book_project")):
            self.is_book_writing_mode = True
            # Try to get project_name from context first (e.g., if UI sends it for an existing project)
            self.current_project_name = context.get("project_name")
            if not self.current_project_name:
                 self.current_project_name = self._extract_project_name(user_goal, context) # Fallback to goal extraction
            
            if not self.current_project_name:
                 self.logger.warning(f"Book writing mode detected, but no project name found in goal or context: {user_goal}")
                 yield StreamData(type="final_answer", content="To work on a book, please specify a project name. For example: \\\"Create a new book project named 'My Awesome Novel'\\\" or \\\"Load book project 'My Existing Work'\\\".", iteration=0)
                 return 

            book_state_memory_key = f"book_project_{self.current_project_name.replace(' ', '_').lower()}"
            self.logger.info(f"Book writing mode enabled. Project: '{self.current_project_name}'. Memory key: {book_state_memory_key}")
            
            loaded_successfully = False
            try:
                # Use MemoryManager's search_memory with metadata filter
                # Assuming search_memory can filter by a unique 'key' in metadata
                retrieved_segments = self.memory.search_memory(
                    query=book_state_memory_key, # Query using the specific key
                    k=5, # Retrieve a few candidates in case of similar keys or semantic search behavior
                    # type_filter="BOOK_WRITING_STATE" # Assuming search_memory might have a type_filter
                                                      # If not, we filter manually below.
                )
                
                found_segment = None
                if retrieved_segments:
                    for segment in retrieved_segments:
                        if not (hasattr(segment, 'metadata') and isinstance(segment.metadata, dict) and \
                                hasattr(segment, 'content')):
                            self.logger.warning(f"Retrieved memory segment has unexpected structure: {segment}")
                            continue

                        segment_type = segment.metadata.get("type")
                        segment_key = segment.metadata.get("key")

                        if segment_type == "BOOK_WRITING_STATE" and segment_key == book_state_memory_key:
                            found_segment = segment
                            break
                
                if found_segment:
                    self.book_writing_state = BookWritingState.parse_raw(found_segment.content)
                    self.logger.info(f"Successfully loaded BookWritingState for project: {self.current_project_name}")
                else:
                    self.logger.info(f"No existing BookWritingState found for project: {self.current_project_name}. Initializing new state.")
                    self.book_writing_state = BookWritingState(project_name=self.current_project_name)
                    self.memory.add_memory(
                        content=self.book_writing_state.json(),
                        type="BOOK_WRITING_STATE",
                        metadata={"key": book_state_memory_key, "project_name": self.current_project_name, "type": "BOOK_WRITING_STATE"}
                    )
                    self.logger.info(f"Initialized and saved new BookWritingState for project: {self.current_project_name}")

            except ValidationError as ve:
                self.logger.error(f"Pydantic ValidationError parsing BookWritingState for '{self.current_project_name}': {ve}")
                self.book_writing_state = BookWritingState(project_name=self.current_project_name)
                self.memory.add_memory(
                    content=self.book_writing_state.json(),
                    type="BOOK_WRITING_STATE",
                    metadata={"key": book_state_memory_key, "project_name": self.current_project_name, "type": "BOOK_WRITING_STATE"}
                )
                self.logger.info(f"Initialized and saved new BookWritingState for '{self.current_project_name}' due to parsing error of existing state.")
            except Exception as e:
                self.logger.error(f"Unexpected error loading or initializing BookWritingState for project '{self.current_project_name}': {e}", exc_info=True)
                self.book_writing_state = BookWritingState(project_name=self.current_project_name)
                # Potentially save this new state as well, depending on desired behavior for unexpected errors
                self.logger.info(f"Initialized new BookWritingState for '{self.current_project_name}' due to unexpected error.")
        
        elif self.is_book_writing_mode and not self.current_project_name:
            # Project name was requested, orchestrator will return the request.
            # self.book_writing_state remains None.
            # The LLM prompt will be adjusted accordingly.
            self.logger.debug("Book writing mode active, but project name is missing. Awaiting user input for project name.")

        init_content = MemorySegmentContent(text=f"Orchestrator initialized with goal: {user_goal}")
        await self.memory.add_memory_segment(MemorySegment(
            type="ORCHESTRATOR_INIT",
            source=self.agent_name,
            content=init_content,
            metadata={"session_id": session_id, "user_goal": user_goal, "timestamp": time.time()}
        ))

        conversation_history = context.get("conversation_history", [])
        previous_steps: List[Dict[str, Any]] = []

        # --- Start of the iteration loop ---
        for i in range(self.max_iterations):
            yield StreamData(type="info", content=f"Orchestrator iteration {i+1}/{self.max_iterations} for session \\'{session_id}\\'.", iteration=i+1, max_iterations=self.max_iterations)
            self.logger.info(f"Orchestrator iteration {i+1}/{self.max_iterations} for session \\'{session_id}\\'.")

            prompt = self._build_llm_prompt(user_goal, conversation_history, previous_steps)
            
            prompt_log_content = MemorySegmentContent(text=f"LLM Prompt for iteration {i+1} (see debug for full prompt)")
            await self.memory.add_memory_segment(MemorySegment(
                type="ORCHESTRATOR_LLM_PROMPT",
                source=self.agent_name,
                content=prompt_log_content,
                metadata={"session_id": session_id, "iteration": i + 1, "timestamp": time.time()}
            ))
            self.logger.debug(f"Orchestrator LLM Prompt for session \\'{session_id}\\', iteration {i+1}:\\n{prompt}")

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

            self.logger.debug(f"Orchestrator LLM Raw Response for session \\'{session_id}\\', iteration {i+1}: {llm_response_str}")

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
            self.logger.info(f"Orchestrator Thought for session \\'{session_id}\\', iteration {i+1}: {current_thought.thought}")

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

            active_tool_name = None 
            if current_action.action_type == "tool_call" and current_action.tool_call:
                active_tool_name = current_action.tool_call.tool_name
                tool_args_dict = current_action.tool_call.arguments
                sub_goal = tool_args_dict.get("goal", "") # Specialized agents expect 'goal'

                if not sub_goal and active_tool_name.startswith("book_"): # Stricter for book agents
                    self.logger.error(f"Orchestrator attempted to delegate to book agent {active_tool_name} for session \\'{session_id}\\' but no \\'goal\\' was specified in args: {tool_args_dict}")
                    observation = f"Error: No goal specified for agent {active_tool_name}."
                    yield StreamData(type="error", content=observation, error_details=observation, tool_name=active_tool_name, iteration=i+1)
                else:
                    yield StreamData(type="info", content=f"Delegating to {active_tool_name}...", tool_name=active_tool_name, tool_args=tool_args_dict, iteration=i+1)
                    delegation_context = context.copy() 
                    delegation_context["delegator_iteration"] = i + 1
                    delegation_context["user_goal"] = user_goal 
                    if self.current_project_name: # Pass project name to sub-agents if available
                        delegation_context["project_name"] = self.current_project_name


                    async for agent_response_chunk in self._handle_agent_delegation(active_tool_name, sub_goal, delegation_context):
                        yield agent_response_chunk
                        if agent_response_chunk.type == "tool_response": 
                            observation = agent_response_chunk.content if isinstance(agent_response_chunk.content, str) else json.dumps(agent_response_chunk.content)
                            self.logger.info(f"Observation from {active_tool_name} for session '{session_id}' (iteration {i+1}): {observation[:200]}...")
                        
                            # --- New BookWritingState Update Logic ---
                            if self.is_book_writing_mode and self.book_writing_state and self.current_project_name and active_tool_name and active_tool_name.startswith("book_") and observation:
                                try:
                                    agent_output_data = json.loads(observation) # Agent is expected to return JSON string
                                    self.logger.info(f"Attempting to update BookWritingState for project '{self.current_project_name}' from {active_tool_name} output.")
                                    updated_state_field = False

                                    # Helper to reduce redundancy for list updates (characters, outlines)
                                    def update_list_items(current_list, new_item_data_list, item_schema, id_field_name="name"):
                                        made_change = False
                                        validated_new_items = [item_schema(**item_data) for item_data in new_item_data_list]
                                        
                                        # Create a dictionary of new items for quick lookup
                                        new_items_dict = {getattr(item, id_field_name): item for item in validated_new_items if hasattr(item, id_field_name)}

                                        temp_list = []
                                        existing_item_ids = set()

                                        for existing_item in current_list:
                                            item_id = getattr(existing_item, id_field_name)
                                            existing_item_ids.add(item_id)
                                            if item_id in new_items_dict:
                                                temp_list.append(new_items_dict[item_id]) # Replace with new version
                                                made_change = True
                                            else:
                                                temp_list.append(existing_item) # Keep old version
                                        
                                        # Add genuinely new items (those whose ID wasn't in existing_item_ids)
                                        for new_item_id, new_item in new_items_dict.items():
                                            if new_item_id not in existing_item_ids:
                                                temp_list.append(new_item)
                                                made_change = True
                                        
                                        if made_change:
                                            current_list[:] = temp_list # Modify list in-place
                                        return made_change

                                    if active_tool_name == "book_plotter":
                                        if "detailed_chapter_outlines" in agent_output_data:
                                            new_outlines_data = agent_output_data["detailed_chapter_outlines"]
                                            if isinstance(new_outlines_data, list):
                                                # Replace entire list for outlines, or implement smarter merge if needed
                                                self.book_writing_state.detailed_chapter_outlines = [ChapterOutlineSchema(**outline_data) for outline_data in new_outlines_data]
                                                self.logger.info(f"Updated detailed_chapter_outlines for '{self.current_project_name}'. Count: {len(self.book_writing_state.detailed_chapter_outlines)}")
                                                updated_state_field = True
                                        if "overall_plot_summary" in agent_output_data and isinstance(agent_output_data["overall_plot_summary"], str):
                                            self.book_writing_state.overall_plot_summary = agent_output_data["overall_plot_summary"]
                                            self.logger.info(f"Updated overall_plot_summary for '{self.current_project_name}'.")
                                            updated_state_field = True

                                    elif active_tool_name == "book_character_dev" and "character_profiles" in agent_output_data:
                                        new_profiles_data = agent_output_data["character_profiles"]
                                        if isinstance(new_profiles_data, list):
                                            if update_list_items(self.book_writing_state.character_profiles, new_profiles_data, CharacterProfileSchema, "name"):
                                                self.logger.info(f"Updated character_profiles for '{self.current_project_name}'. Current count: {len(self.book_writing_state.character_profiles)}")
                                                updated_state_field = True
                                    
                                    elif active_tool_name == "book_worldbuilder" and "world_building_notes" in agent_output_data:
                                        notes_data = agent_output_data["world_building_notes"]
                                        if isinstance(notes_data, dict):
                                            self.book_writing_state.world_building_notes = WorldAnvilSchema(**notes_data)
                                            self.logger.info(f"Updated world_building_notes for '{self.current_project_name}'.")
                                            updated_state_field = True

                                    elif active_tool_name == "book_prose_generator" and "generated_prose_update" in agent_output_data:
                                        prose_update = agent_output_data["generated_prose_update"] 
                                        if isinstance(prose_update, dict):
                                            for chapter_key, prose_data_dict in prose_update.items():
                                                if isinstance(prose_data_dict, dict):
                                                    # For Phase 1, ChapterProseSchema is simple. Phase 3 will add versioning.
                                                    self.book_writing_state.generated_prose[str(chapter_key)] = ChapterProseSchema(**prose_data_dict)
                                                    self.logger.info(f"Updated generated_prose for chapter '{chapter_key}' in project '{self.current_project_name}'.")
                                                    updated_state_field = True
                                                else:
                                                     self.logger.warning(f"Prose data for chapter '{chapter_key}' is not a dict: {prose_data_dict}")
                                        else:
                                            self.logger.warning(f"generated_prose_update from {active_tool_name} is not a dict: {prose_update}")
                                    
                                    elif active_tool_name == "book_editor":
                                        if "generated_prose_update" in agent_output_data:
                                            prose_update = agent_output_data["generated_prose_update"]
                                            if isinstance(prose_update, dict):
                                                for chapter_key, prose_data_dict in prose_update.items():
                                                    if isinstance(prose_data_dict, dict):
                                                        self.book_writing_state.generated_prose[str(chapter_key)] = ChapterProseSchema(**prose_data_dict)
                                                        self.logger.info(f"Editor updated generated_prose for chapter '{chapter_key}' in project '{self.current_project_name}'.")
                                                        updated_state_field = True
                                        if "revision_notes" in agent_output_data and isinstance(agent_output_data["revision_notes"], str):
                                            self.book_writing_state.revision_notes = agent_output_data["revision_notes"]
                                            self.logger.info(f"Updated revision_notes for '{self.current_project_name}'.")
                                            updated_state_field = True
                                    
                                    if updated_state_field:
                                        book_state_memory_key_for_save = f"book_project_{self.current_project_name.replace(' ', '_').lower()}"
                                        state_json_string = self.book_writing_state.model_dump_json(indent=2)
                                        
                                        # Use self.memory.add_memory, consistent with initialization
                                        # Assuming add_memory is synchronous. If it's async, it would need 'await'.
                                        self.memory.add_memory(
                                            content=state_json_string,
                                            type="BOOK_WRITING_STATE",
                                            metadata={
                                                "key": book_state_memory_key_for_save, 
                                                "project_name": self.current_project_name, 
                                                "type": "BOOK_WRITING_STATE", # Ensure type is part of metadata for filtering
                                                "timestamp": time.time(),
                                                "updated_by_agent": active_tool_name,
                                                "session_id": session_id
                                            }
                                        )
                                        self.logger.info(f"Persisted updated BookWritingState to memory for '{self.current_project_name}' with key '{book_state_memory_key_for_save}'.")
                                    elif not agent_output_data: # Agent returned empty JSON string "{}"
                                        self.logger.info(f"{active_tool_name} returned empty JSON, no BookWritingState updates applied.")
                                    else: # Agent returned JSON but no recognized keys for update
                                         self.logger.info(f"No specific BookWritingState updates applied from {active_tool_name} output (keys not recognized or data invalid): {observation[:200]}")

                                except json.JSONDecodeError:
                                    self.logger.warning(f"Observation from {active_tool_name} was not valid JSON. Cannot update BookWritingState. Observation: {observation[:200]}")
                                except Exception as e: 
                                    self.logger.exception(f"Error updating/persisting BookWritingState from {active_tool_name} response for project '{self.current_project_name}': {e}. Observation: {observation[:200]}")
                            # --- End of New BookWritingState Update Logic ---
            
            previous_steps.append({"thought": current_thought, "action": current_action, "observation": observation})
            if len(previous_steps) >= self.max_iterations:
                self.logger.warning(f"Max iterations ({self.max_iterations}) reached for session \\'{session_id}\\'.")
                yield StreamData(type="final_answer", content="I have reached the maximum number of steps for this task. If the goal is not yet met, please try rephrasing or breaking it down.", iteration=i+1)
                return

        self.logger.info(f"Orchestrator finished for session \\'{session_id}\\' without reaching a final answer within max_iterations.")
        # Fallback if loop finishes (should ideally be handled by max_iterations check or final_answer)
        yield StreamData(type="final_answer", content="The process completed, but a definitive answer was not reached within the allocated steps.", iteration=self.max_iterations)

    def _update_list_items(self, existing_items: List[Any], new_item_data_list: List[Dict], item_schema: Any, key_field: str):
        """
        Helper to update a list of Pydantic models.
        Adds new items or updates existing ones based on a key_field.
        """
        if not isinstance(new_item_data_list, list):
            self.logger.warning(f"new_item_data_list is not a list: {new_item_data_list}")
            return

        updated_keys = set()
        for i in range(len(existing_items) -1, -1, -1): # Iterate backwards for safe removal/update
            existing_item = existing_items[i]
            if not hasattr(existing_item, key_field):
                continue
            
            existing_key_value = getattr(existing_item, key_field)
            for new_data in new_item_data_list:
                if not isinstance(new_data, dict): continue
                new_key_value = new_data.get(key_field)
                if new_key_value == existing_key_value:
                    try:
                        updated_item = item_schema(**new_data)
                        existing_items[i] = updated_item # Replace existing
                        updated_keys.add(new_key_value)
                        self.logger.debug(f"Updated item with {key_field}={new_key_value}")
                    except ValidationError as e:
                        self.logger.error(f"Validation error updating item with {key_field}={new_key_value}: {e}")
                    break 
 
        # Add items that weren't updates of existing ones
        for new_data in new_item_data_list:
            if not isinstance(new_data, dict): continue
            new_key_value = new_data.get(key_field)
            if new_key_value not in updated_keys:
                try:
                    new_item = item_schema(**new_data)
                    existing_items.append(new_item)
                    self.logger.debug(f"Added new item with {key_field}={new_key_value}")
                except ValidationError as e:
                    self.logger.error(f"Validation error adding new item with {key_field}={new_key_value}: {e}")
