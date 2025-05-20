# agents/orchestrator_agent.py
import asyncio
import json
import logging
import time
import re
import uuid
from typing import AsyncGenerator, List, Dict, Any, Optional, Union
from pydantic import ValidationError # Added for handling Pydantic validation errors
from datetime import datetime # Added for consistent timestamping
from core.json_utils import safe_json_loads, balance_json_braces # Import JSON utilities

from agents.base_agent import BaseAgent
from core.config import AppConfig
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager
from core.schemas import LLMToolCall, OrchestratorLLMResponse, OrchestratorThought, OrchestratorAction, MemorySegmentContent, StreamData
from core.tool_registry import ToolRegistry # Import ToolRegistry
from .book_writing_schemas import BookWritingState # Simplified import
from core.schemas import StreamData # Ensure StreamData is imported

from agents.book_writing_schemas import BookWritingState, ChapterOutlineSchema, CharacterProfileSchema, WorldAnvilSchema, ChapterProseSchema
from core.debug_utils import log_async_execution_time

class OrchestratorAgent(BaseAgent):
    def __init__(self,
                 agent_name: str,
                 config: Any,  # Changed from AppConfig to Any to accept AgentProfileConfig
                 llm_interface: LLMInterface,
                 memory_manager: MemoryManager,
                 delegation_targets: Dict[str, BaseAgent],
                 tool_registry: Optional[ToolRegistry] = None,
                 max_iterations: int = 10):
        super().__init__(agent_name, config, llm_interface, memory_manager, tool_registry=tool_registry)
        self.delegation_targets = delegation_targets if delegation_targets else {}
        self.max_iterations = max_iterations
        self.logger = logging.getLogger(f"WITS.{self.__class__.__name__}")
        self.final_answer_tool_name = "FinalAnswerTool"
        self.is_book_writing_mode = False
        self.book_writing_state = None
        self.current_project_name = None
        
        # Use the model name from llm_interface
        self.orchestrator_model_name = llm_interface.model_name        # Initialize the orchestrator with the model from llm_interface
        self.logger.info(f"OrchestratorAgent initialized. It will use LLM model: {self.orchestrator_model_name} from its LLMInterface. Max iterations: {self.max_iterations}")
        self.logger.info(f"Delegation targets: {list(self.delegation_targets.keys())}")
        self.logger.info(f"Final answer tool configured as: {self.final_answer_tool_name}")

    def _extract_project_name(self, user_goal: str, context: Dict[str, Any]) -> Optional[str]:
        if "project_name" in context and context["project_name"]:
            self.logger.debug(f"Extracted project_name '{context['project_name']}' from context.")
            return str(context["project_name"])

        patterns = [
            r"(?:project named|book named|novel named|story named|project|book|novel|story)\s+(?:\*\*)?['\"]([^'\"\*]+)['\"](?:\*\*)?",
            r"create a new book titled\s+(?:\*\*)?['\"]([^'\"\*]+)['\"](?:\*\*)?",
            r"start book\s+(?:\*\*)?['\"]([^'\"\*]+)['\"](?:\*\*)?",
            r"create a new book project titled\s+(?:\*\*)?['\"]([^'\"\*]+)['\"](?:\*\*)?",
            r"create a new book project named\s+(?:\*\*)?['\"]([^'\"\*]+)['\"](?:\*\*)?"
        ]
        for pattern in patterns:
            match = re.search(pattern, user_goal, re.IGNORECASE)
            if match:
                project_name = match.group(1).strip()
                if project_name.endswith('**'):
                    project_name = project_name[:-2].strip()
                if project_name:
                    self.logger.debug(f"Extracted project_name '{project_name}' from user_goal using pattern: {pattern}")
                    return project_name
        
        self.logger.debug(f"Could not extract project_name from user_goal: {user_goal} or context.")
        return None

    def _get_agent_descriptions_for_prompt(self) -> str:
        descriptions = []
        for name, agent in self.delegation_targets.items():
            # Try to get description from agent's config first
            config = getattr(agent, 'config_full', None)
            if config and hasattr(config, 'display_name') and hasattr(config, 'description'):
                display_name = config.display_name
                description = config.description
            else:
                # Fallback to get_description or default
                display_name = name
                description = agent.get_description() if hasattr(agent, 'get_description') and callable(agent.get_description) else f"No specific description available for {name}."
            
            descriptions.append(f'- Tool Name: "{name}"\n  Display Name: "{display_name}"\n  Description: {description}')
        
        if self.is_book_writing_mode:
            descriptions.extend([
                '- Tool Name: "book_plotter"\n  Description: Manages plot and chapter outlines for book writing projects.',
                '- Tool Name: "book_character_dev"\n  Description: Handles character development and profiles.',
                '- Tool Name: "book_worldbuilder"\n  Description: Creates the setting, lore, and rules of fictional worlds.',
                '- Tool Name: "book_prose_generator"\n  Description: Writes the actual narrative, dialogue, and descriptions.',
                '- Tool Name: "book_editor"\n  Description: Reviews and refines generated book content.'
            ])

        descriptions.append(f'- Tool Name: "{self.final_answer_tool_name}"\n  Description: Use this tool to provide the final answer directly to the user when the goal has been fully achieved and no more actions are needed.')
        return "\n".join(descriptions)

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

Example of the full JSON response for delegating to an agent (as a tool_call):
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

Example of the full JSON response for providing a final answer:
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
              # Use the safe_json_loads function from our json_utils module
            try:
                llm_json = safe_json_loads(llm_response_str, session_id)
            except json.JSONDecodeError as json_err:
                # If still failing, log the error and return None to trigger a retry
                self.logger.error(f"JSON parsing error in session '{session_id}': {json_err}")
                return None
            
            # Check for project_name_extracted in LLM response
            if "project_name_extracted" in llm_json and llm_json["project_name_extracted"]:
                extracted_project_name = llm_json["project_name_extracted"]
                self.logger.info(f"Found project_name_extracted in LLM response: '{extracted_project_name}'")
                
                # Update agent state if we're in book writing mode but project name isn't set
                if self.is_book_writing_mode and not self.current_project_name:
                    self.current_project_name = extracted_project_name
                    # Only create BookWritingState if we have a valid project name
                    if extracted_project_name and not self.book_writing_state:
                        self.book_writing_state = BookWritingState(project_name=extracted_project_name)
                        self.logger.info(f"Initialized BookWritingState with project name from LLM response: '{extracted_project_name}'")
            
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
            agent_specific_state_slice = {} # Initialize here

            if self.is_book_writing_mode and self.book_writing_state and agent_name.startswith("book_"):
                # Populate agent_specific_state_slice based on agent_name
                if agent_name == "book_plotter":
                    agent_specific_state_slice = {
                        "overall_plot_summary": self.book_writing_state.overall_plot_summary,
                        "character_profiles": [p.model_dump() for p in self.book_writing_state.character_profiles],
                        "detailed_chapter_outlines": [o.model_dump() for o in self.book_writing_state.detailed_chapter_outlines]
                    }
                elif agent_name == "book_character_dev":
                    agent_specific_state_slice = {
                        "overall_plot_summary": self.book_writing_state.overall_plot_summary,
                        "character_profiles": [p.model_dump() for p in self.book_writing_state.character_profiles]
                    }
                elif agent_name == "book_worldbuilder":
                    if self.book_writing_state.world_building_notes:
                        agent_specific_state_slice["world_building_notes"] = self.book_writing_state.world_building_notes.model_dump()
                    else:
                        agent_specific_state_slice["world_building_notes"] = WorldAnvilSchema().model_dump() # Ensure it's always a dict
                elif agent_name == "book_prose_generator":
                    agent_specific_state_slice = {
                        "overall_plot_summary": self.book_writing_state.overall_plot_summary,
                        "detailed_chapter_outlines": [o.model_dump() for o in self.book_writing_state.detailed_chapter_outlines],
                        "character_profiles": [p.model_dump() for p in self.book_writing_state.character_profiles]
                    }
                    if self.book_writing_state.world_building_notes:
                        agent_specific_state_slice["world_building_notes"] = self.book_writing_state.world_building_notes.model_dump()
                    if self.book_writing_state.writing_style_guide:
                        agent_specific_state_slice["writing_style_guide"] = self.book_writing_state.writing_style_guide
                    if self.book_writing_state.tone_guide:
                        agent_specific_state_slice["tone_guide"] = self.book_writing_state.tone_guide
                elif agent_name == "book_editor":
                    agent_specific_state_slice = {
                        "generated_prose": {k: v.model_dump() for k, v in self.book_writing_state.generated_prose.items()},
                        "detailed_chapter_outlines": [o.model_dump() for o in self.book_writing_state.detailed_chapter_outlines],
                        "character_profiles": [p.model_dump() for p in self.book_writing_state.character_profiles]
                    }
                    if self.book_writing_state.world_building_notes:
                        agent_specific_state_slice["world_building_notes"] = self.book_writing_state.world_building_notes.model_dump()
                    if self.book_writing_state.writing_style_guide:
                        agent_specific_state_slice["writing_style_guide"] = self.book_writing_state.writing_style_guide
                    if self.book_writing_state.tone_guide:
                        agent_specific_state_slice["tone_guide"] = self.book_writing_state.tone_guide
                    agent_specific_state_slice["revision_notes"] = self.book_writing_state.revision_notes

                if agent_specific_state_slice: # Check if it was populated
                    delegation_context["book_writing_state_slice"] = agent_specific_state_slice
                    self.logger.info(f"Passing book_writing_state_slice to {agent_name} with keys: {list(agent_specific_state_slice.keys())}")
                else:
                    self.logger.info(f"No specific book_writing_state_slice prepared for {agent_name} (it might not be a book agent or no specific slice needed).")
            else:
                self.logger.info(f"Not in book writing mode or book_writing_state is None, or agent is not a book agent. No state slice passed for {agent_name}.")


            yield StreamData(type="info", content=f"Delegating to {agent_name} with goal: {goal}", tool_name=agent_name, tool_args={"goal": goal}, iteration=context.get("delegator_iteration"))

            delegation_context["session_id"] = session_id
            delegation_context["delegator_agent"] = self.agent_name

            full_response_content = ""
            last_yielded_stream_data: Optional[StreamData] = None

            try:
                agent_response = await delegate_agent.run(task_description=goal, context=delegation_context)

                # If agent_response is already an async generator (has __aiter__)
                if hasattr(agent_response, '__aiter__'):
                    try:
                        async for response_chunk in agent_response:
                            if not isinstance(response_chunk, StreamData):
                                self.logger.warning(f"Agent {agent_name} yielded non-StreamData object: {type(response_chunk)}. Converting to StreamData.")
                                response_chunk = StreamData(
                                    type="content",
                                    content=str(response_chunk),
                                    tool_name=agent_name,
                                    iteration=context.get("delegator_iteration")
                                )

                            if response_chunk.iteration is None:
                                response_chunk.iteration = context.get("delegator_iteration")

                            yield response_chunk
                            last_yielded_stream_data = response_chunk

                            if response_chunk.type == "update_state" and isinstance(response_chunk.content, dict) and "book_writing_state_slice" in response_chunk.content:
                                state_slice_update = response_chunk.content.get("book_writing_state_slice", {})
                                if self.is_book_writing_mode and self.book_writing_state and state_slice_update:
                                    self.logger.info(f"Received book_writing_state_slice update from {agent_name} with keys: {list(state_slice_update.keys())}")
                                    self._update_book_writing_state_from_slice(state_slice_update, agent_name, session_id)
                                    update_summary = f"Updated BookWritingState with data from {agent_name}."
                                    if full_response_content: full_response_content += "\n" + update_summary
                                    else: full_response_content = update_summary
                            elif response_chunk.type in ["final_answer", "content", "tool_response"]:
                                chunk_content_str = ""
                                if isinstance(response_chunk.content, str):
                                    chunk_content_str = response_chunk.content
                                elif isinstance(response_chunk.content, dict):
                                    try:
                                        chunk_content_str = json.dumps(response_chunk.content)
                                    except TypeError:
                                        chunk_content_str = str(response_chunk.content)
                                
                                if chunk_content_str:
                                    if full_response_content: full_response_content += "\n" + chunk_content_str
                                    else: full_response_content = chunk_content_str

                    except Exception as stream_error:
                        error_msg = f"Error processing stream from agent {agent_name}: {str(stream_error)}"
                        self.logger.exception(error_msg)
                        yield StreamData(
                            type="error",
                            content=error_msg,
                            tool_name=agent_name,
                            error_details=str(stream_error),
                            iteration=context.get("delegator_iteration")
                        )
                        return

                # If agent_response is a StreamData object directly
                elif isinstance(agent_response, StreamData):
                    if agent_response.iteration is None:
                        agent_response.iteration = context.get("delegator_iteration")
                    yield agent_response
                    last_yielded_stream_data = agent_response
                    if agent_response.content:
                        if isinstance(agent_response.content, str):
                            full_response_content = agent_response.content
                        elif isinstance(agent_response.content, dict):
                            try:
                                full_response_content = json.dumps(agent_response.content)
                            except TypeError:
                                full_response_content = str(agent_response.content)

                # If agent_response is some other value, convert it to StreamData
                elif agent_response is not None:
                    response_content_str = str(agent_response)
                    stream_data = StreamData(
                        type="content",
                        content=response_content_str,
                        tool_name=agent_name,
                        iteration=context.get("delegator_iteration")
                    )
                    yield stream_data
                    last_yielded_stream_data = stream_data
                    full_response_content = response_content_str
                else:
                    # Agent returned None
                    full_response_content = f"Agent {agent_name} completed task but returned no explicit content."
                    self.logger.info(full_response_content)

            except Exception as e:
                error_msg = f"Error during delegation to agent '{agent_name}' for session '{session_id}': {str(e)}"
                self.logger.exception(error_msg)
                yield StreamData(
                    type="error",
                    content=error_msg,
                    tool_name=agent_name,
                    error_details=str(e),
                    iteration=context.get("delegator_iteration")
                )
                return

            trimmed_response = full_response_content.strip()
            if not trimmed_response and not (last_yielded_stream_data and last_yielded_stream_data.type == "update_state"):
                trimmed_response = f"Agent {agent_name} completed its task but returned no explicit textual content."
                self.logger.warning(f"Agent {agent_name} returned no explicit textual content for session '{session_id}' on goal: {goal}")

            # Ensure a final tool_response is sent if the agent didn't send a terminal signal
            if not last_yielded_stream_data or last_yielded_stream_data.type not in ["tool_response", "final_answer", "error"]:
                final_tool_response_content = trimmed_response if trimmed_response else f"Agent {agent_name} completed."
                yield StreamData(
                    type="tool_response",
                    content=final_tool_response_content,
                    tool_name=agent_name,
                    iteration=context.get("delegator_iteration")
                )
            elif last_yielded_stream_data.type == "update_state" and not trimmed_response:
                yield StreamData(
                    type="tool_response",
                    content=f"Agent {agent_name} completed with state update.",
                    tool_name=agent_name,
                    iteration=context.get("delegator_iteration")
                )

            self.logger.info(f"Agent '{agent_name}' completed for session '{session_id}'. Final response summary length: {len(trimmed_response)}")

        else:
            error_msg = f"Attempted to delegate to unknown agent '{agent_name}' for session '{session_id}'."
            self.logger.error(error_msg)
            yield StreamData(
                type="error",
                content=error_msg,
                tool_name=agent_name,
                error_details=f"Agent {agent_name} not found.",
                iteration=context.get("delegator_iteration")
            )
    
    async def _save_current_book_state_to_memory(self): # Ensure this is correctly unindented
        if not self.book_writing_state or not self.current_project_name:
            self.logger.warning("Attempted to save book state, but state or project name is missing.")
            return
        
        if not hasattr(self, 'memory') or self.memory is None:
            self.logger.warning("Memory manager is not available. Cannot save book writing state.")
            return

        book_state_memory_key = f"book_project_{self.current_project_name.replace(' ', '_').lower()}"
        
        try:
            # Using add_segment as corrected previously
            await self.memory.add_segment(
                segment_type="BOOK_WRITING_STATE",
                source=self.agent_name, # Orchestrator is saving it
                content_text=f"Book state for {self.current_project_name} automatically saved.",
                tool_args=self.book_writing_state.model_dump(), # Save the full state here
                meta={"key": book_state_memory_key, 
                      "type": "BOOK_WRITING_STATE", 
                      "project_name": self.current_project_name,
                      "timestamp": datetime.now().isoformat() # Use ISO format timestamp
                     },
            )
            self.logger.info(f"Successfully saved book state for project '{self.current_project_name}' to memory with key '{book_state_memory_key}'.")
        except Exception as e:
            self.logger.error(f"Error saving book state for project '{self.current_project_name}': {e}", exc_info=True)

    def _update_book_writing_state_from_slice(self, state_slice: Dict[str, Any], agent_name: str, session_id: str) -> None:
        """
        Update the OrchestratorAgent's BookWritingState with data from specialized agents.
        
        Args:
            state_slice: The state slice containing updates from the specialized agent
            agent_name: The name of the agent that provided the update
            session_id: The current session ID for logging purposes
        """
        if not self.is_book_writing_mode or not self.book_writing_state:
            self.logger.warning(f"Received book_writing_state_slice update but not in book writing mode or state is None. Agent: {agent_name}")
            return
            
        try:
            # Handle overall_plot_summary update
            if "overall_plot_summary" in state_slice:
                new_plot_summary = state_slice.get("overall_plot_summary")
                if new_plot_summary and isinstance(new_plot_summary, str):
                    old_summary = self.book_writing_state.overall_plot_summary or ""
                    self.book_writing_state.overall_plot_summary = new_plot_summary
                    self.logger.info(f"Updated overall_plot_summary from {agent_name}. Old length: {len(old_summary)}, New length: {len(new_plot_summary)}")
            
            # Handle detailed_chapter_outlines update
            if "detailed_chapter_outlines" in state_slice:
                new_outlines_data = state_slice.get("detailed_chapter_outlines", [])
                if isinstance(new_outlines_data, list) and len(new_outlines_data) > 0:
                    # Process and validate each chapter outline before adding to the state
                    validated_outlines = []
                    for outline_dict in new_outlines_data:
                        try:
                            # Validate against ChapterOutlineSchema
                            validated_outline = ChapterOutlineSchema.model_validate(outline_dict)
                            validated_outlines.append(validated_outline)
                        except Exception as e:
                            self.logger.warning(f"Invalid chapter outline from {agent_name} in session {session_id}: {e}. Outline: {outline_dict}")
                    
                    # Update outlines based on agent's updates
                    if validated_outlines:
                        # Create a map of existing outlines by chapter number for efficient update
                        existing_outlines_map = {o.chapter_number: o for o in self.book_writing_state.detailed_chapter_outlines}
                        
                        # Update existing or add new chapter outlines
                        for new_outline in validated_outlines:
                            existing_outlines_map[new_outline.chapter_number] = new_outline
                            
                        # Reconstruct the full outline list in sorted chapter order
                        self.book_writing_state.detailed_chapter_outlines = [
                            existing_outlines_map[ch_num] for ch_num in sorted(existing_outlines_map.keys())
                        ]
                        
                        self.logger.info(f"Updated detailed_chapter_outlines from {agent_name}. Now have {len(self.book_writing_state.detailed_chapter_outlines)} chapters.")
            
            # Handle character_profiles update
            if "character_profiles" in state_slice:
                new_profiles_data = state_slice.get("character_profiles", [])
                if isinstance(new_profiles_data, list) and len(new_profiles_data) > 0:
                    # Process and validate each character profile
                    validated_profiles = []
                    for profile_dict in new_profiles_data:
                        try:
                            # Validate against CharacterProfileSchema
                            validated_profile = CharacterProfileSchema.model_validate(profile_dict)
                            validated_profiles.append(validated_profile)
                        except Exception as e:
                            self.logger.warning(f"Invalid character profile from {agent_name} in session {session_id}: {e}. Profile: {profile_dict}")
                    
                    # Update character profiles
                    if validated_profiles:
                        # Create a map of existing profiles by name
                        existing_profiles_map = {p.name: p for p in self.book_writing_state.character_profiles}
                        
                        # Update existing or add new character profiles
                        for new_profile in validated_profiles:
                            existing_profiles_map[new_profile.name] = new_profile
                            
                        # Reconstruct the profiles list
                        self.book_writing_state.character_profiles = list(existing_profiles_map.values())
                        self.logger.info(f"Updated character_profiles from {agent_name}. Now have {len(self.book_writing_state.character_profiles)} character profiles.")
            
            # Handle world_building_notes update
            if "world_building_notes" in state_slice:
                world_notes_data = state_slice.get("world_building_notes")
                if isinstance(world_notes_data, dict):
                    try:
                        # Validate against WorldAnvilSchema
                        validated_world_notes = WorldAnvilSchema.model_validate(world_notes_data)
                        self.book_writing_state.world_building_notes = validated_world_notes
                        self.logger.info(f"Updated world_building_notes from {agent_name}.")
                    except Exception as e:
                        self.logger.warning(f"Invalid world building notes from {agent_name} in session {session_id}: {e}")
            
            # Handle other potential updates (revision_notes, writing_style_guide, tone_guide)
            simple_string_fields = ["revision_notes", "writing_style_guide", "tone_guide"]
            for field in simple_string_fields:
                if field in state_slice and isinstance(state_slice[field], str):
                    setattr(self.book_writing_state, field, state_slice[field])
                    self.logger.info(f"Updated {field} from {agent_name}.")
            
            # After updating, save the state to memory
            asyncio.create_task(self._save_current_book_state_to_memory())
            
        except Exception as e:
            self.logger.error(f"Error updating BookWritingState from {agent_name} in session {session_id}: {e}", exc_info=True)

    @log_async_execution_time(logging.getLogger(f"WITS.OrchestratorAgent.run"))
    async def run(self, user_goal: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[StreamData, None]:
        """Run a goal through the orchestrator. Let's get this party started! \\o/"""
        # Initialize context and get our session ID (gotta keep track of things! ^_^)
        context = context or {}
        session_id = context.get("session_id", str(uuid.uuid4()))
        conversation_history = context.get("conversation_history", [])

        self.logger.info(f"Orchestrator starting for session '{session_id}'. Goal: {user_goal}")

        try:
            # Check if this is a book writing project =D
            if (project_name := self._extract_project_name(user_goal, context)) is not None:
                self.current_project_name = project_name
                self.is_book_writing_mode = True
                if not self.book_writing_state:
                    self.book_writing_state = BookWritingState(project_name=project_name)
                self.logger.info(f"Book writing mode active for project: {project_name} \\o/")
            elif "book" in user_goal.lower():
                self.logger.warning(f"Book mode detected but no project name found? O.o Goal: {user_goal}")

            previous_steps: List[Dict[str, Any]] = []

            # Time for the ReAct loop! Here we go! =D
            for i in range(self.max_iterations):
                yield StreamData(
                    type="info", 
                    content=f"Orchestrator iteration {i+1}/{self.max_iterations} for session '{session_id}'.",
                    iteration=i+1, 
                    max_iterations=self.max_iterations
                )

                # Build the prompt and get LLM's thoughts
                prompt = self._build_llm_prompt(user_goal, conversation_history, previous_steps)
                
                # Get creative with the temperature! (but not too creative x.x)
                options = {}
                if self.config and hasattr(self.config, 'temperature') and self.config.temperature is not None:
                    options["temperature"] = self.config.temperature

                # Ask our LLM friend what to do next =D
                llm_response = await self.llm.chat_completion_async(
                    model_name=self.orchestrator_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options=options
                )
                
                llm_response_content = llm_response.get('response', '')
                self.logger.info(f"Got LLM response for session '{session_id}' \\o/")

                try:
                    # Try to understand what the LLM wants us to do o.O
                    llm_response_parsed = self._parse_llm_response(llm_response_content, session_id)
                    if llm_response_parsed is None:
                        self.logger.error(f"Failed to parse LLM response x.x Session: '{session_id}'")
                        yield StreamData(
                            type="error", 
                            content="Oops! I got confused. Can we try that again?", 
                            iteration=i+1
                        )
                        return

                    # Get the thought process and action
                    thought = llm_response_parsed.thought_process
                    action = llm_response_parsed.chosen_action

                    # Remember what we're doing (for science! And debugging XD)
                    previous_steps.append({
                        "thought": thought.model_dump(),
                        "action": action.model_dump(),
                        "observation": None
                    })

                    # Time to do something! But what? o.O
                    if action.action_type == "tool_call":
                        if action.tool_call is None:
                            self.logger.error(
                                f"Tool call is None?! What happened?! x.x\\n"
                                f"Response: {llm_response_content}"
                            )
                            yield StreamData(
                                type="error", 
                                content="I got confused about what tool to use...", 
                                iteration=i+1
                            )
                            continue

                        tool_name = action.tool_call.tool_name
                        tool_args = action.tool_call.arguments

                        # Make sure we know this tool/agent =D
                        if (tool_name != self.final_answer_tool_name and 
                            tool_name not in self.delegation_targets):
                            self.logger.error(
                                f"Unknown tool/agent: '{tool_name}' o.O\\n"
                                f"Response: {llm_response_content}"
                            )
                            yield StreamData(
                                type="error", 
                                content=f"I don't know how to use '{tool_name}'...", 
                                iteration=i+1
                            )
                            continue

                        # Tool args should be a dict! No exceptions! >.<
                        if not isinstance(tool_args, dict):
                            self.logger.error(
                                f"Invalid args for {tool_name}: {tool_args}\\n"
                                f"Response: {llm_response_content}"
                            )
                            yield StreamData(
                                type="error", 
                                content=f"Got confused about how to use {tool_name}...", 
                                iteration=i+1
                            )
                            continue

                        # If we're delegating to an agent, they need a goal! ^_^
                        if tool_name in self.delegation_targets:
                            delegation_goal = tool_args.get("goal")
                            if not delegation_goal:
                                self.logger.error(
                                    f"No goal for agent '{tool_name}'?! x.x\\n"
                                    f"Args: {tool_args}\\n"
                                    f"Response: {llm_response_content}"
                                )
                                yield StreamData(
                                    type="error", 
                                    content=f"I forgot what to tell {tool_name} to do...", 
                                    iteration=i+1
                                )
                                continue

                            # Time to delegate! Let's do this! \\o/
                            self.logger.info(f"Delegating to {tool_name}: {delegation_goal}")
                            delegation_context = {
                                "delegator_iteration": i+1,
                                "session_id": session_id
                            }

                            async for response in self._handle_agent_delegation(
                                tool_name, delegation_goal, delegation_context
                            ):
                                yield response

                    # The LLM thinks we're done! But are we really? =D
                    elif action.action_type == "final_answer":
                        final_answer_text = action.final_answer
                        if not isinstance(final_answer_text, str):
                            self.logger.error(
                                f"Invalid final answer o.O: {final_answer_text}\\n"
                                f"Response: {llm_response_content}"
                            )
                            yield StreamData(
                                type="error", 
                                content="I got confused about my final answer...", 
                                iteration=i+1
                            )
                            continue

                        # Victory! We did it! \\o/
                        yield StreamData(
                            type="final_answer", 
                            content=final_answer_text, 
                            iteration=i+1
                        )
                        return

                    # What kind of action is this?! o.O
                    else:
                        self.logger.error(
                            f"Unknown action type: '{action.action_type}'\\n"
                            f"Response: {llm_response_content}"
                        )
                        yield StreamData(
                            type="error", 
                            content=f"I don't know how to '{action.action_type}'...", 
                            iteration=i+1
                        )
                        continue

                except Exception as e:
                    # Something broke! Time to panic! x.x
                    self.logger.exception(
                        f"Error in iteration {i+1}: {e}\\n"
                        f"Response: {llm_response_content}"
                    )
                    yield StreamData(
                        type="error", 
                        content=f"Oops! Something went wrong: {str(e)}", 
                        iteration=i+1
                    )
                    return

            # We hit max iterations?! How did we get here? o.O
            yield StreamData(
                type="error",
                content=(
                    f"I hit my limit of {self.max_iterations} steps "
                    "without finishing the task... Sorry! x.x"
                ),
                iteration=self.max_iterations
            )

        except Exception as e:
            # Something REALLY broke! ABANDON SHIP! x.x
            self.logger.exception(f"Critical error in run method: {e}")
            yield StreamData(
                type="error",
                content=f"A critical error occurred: {str(e)}",
                error_details=str(e)
            )
