import json
from typing import Any, Dict, Optional, AsyncGenerator

from agents.base_agent import BaseAgent
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager
from core.schemas import MemorySegment, MemorySegmentContent, StreamData
from tools.story_bible_tool import StoryBibleToolArgsInput # For potential direct use or context

class EditorAgent(BaseAgent):
    def __init__(self, agent_name: str, config: Dict[str, Any], llm_interface: LLMInterface, memory_manager: MemoryManager, tool_registry: Optional[Any] = None):
        super().__init__(agent_name, config, llm_interface, memory_manager, tool_registry)
        self.llm_model_name = self.config_full.models.get('editor', self.config_full.models.default)
        self.logger.info(f"'{self.agent_name}' initialized with LLM model: {self.llm_model_name}.")

    async def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[StreamData, None]:
        effective_context = context if context is not None else {}
        session_id = effective_context.get("session_id", "unknown_session")
        user_goal_summary = effective_context.get("user_goal_summary", "unknown_goal")

        # Attempt to identify the target of editing, e.g., "chapter_5" or "scene_2a"
        # This is a simplified parser. A more robust solution might involve structured task input.
        # Example task: "Edit Chapter 5 for plot consistency and tone."
        # Example task: "Review scene 'The Confrontation' (scene_confrontation) for dialogue pacing."
        
        target_prose_identifier = None
        task_lower = task_description.lower()
        
        # Try to find a direct identifier like "chapter_x" or "scene_y"
        # This could be passed explicitly by the orchestrator in a structured way in the future.
        # For now, we'll assume it might be in the task_description or context.
        if "target_prose_id" in effective_context:
            target_prose_identifier = effective_context["target_prose_id"]
        else: # Fallback to parsing from task description
            parts = task_lower.split()
            try:
                if "chapter" in parts:
                    idx = parts.index("chapter")
                    if idx + 1 < len(parts):
                        target_prose_identifier = f"chapter_{parts[idx+1].replace('.', '').replace(',', '')}"
                elif "scene" in parts: # Could be more specific if scenes have unique names/IDs
                    idx = parts.index("scene")
                    # This is very basic, assumes scene name/id follows "scene" keyword
                    if idx + 1 < len(parts): 
                        # Remove surrounding quotes if any, make it a simple ID
                        scene_name_candidate = parts[idx+1].strip("\"'").replace(" ", "_")
                        target_prose_identifier = f"scene_{scene_name_candidate}"

            except ValueError:
                pass # Keyword not found

        if not target_prose_identifier:
            # If no specific identifier, the task might be general (e.g., "Review overall tone")
            # For now, we'll assume tasks are targeted. If not, this agent would need different logic.
            self.logger.warning(f"'{self.agent_name}' (session: {session_id}) could not identify a specific prose target from task: {task_description}. Editing might be general or fail.")
            # Defaulting to a generic identifier if none found, though this is not ideal for targeted editing.
            target_prose_identifier = "general_edit_task"


        self.logger.info(f"'{self.agent_name}' (session: {session_id}) received task: {task_description} (Target ID: {target_prose_identifier})")
        yield StreamData(type="info", content=f"'{self.agent_name}' starting edit task for '{target_prose_identifier}': {task_description[:100]}...")

        book_state = effective_context.get("book_writing_state", {})
        generated_prose_map = book_state.get("generated_prose", {})
        
        prose_to_edit = None
        if target_prose_identifier != "general_edit_task" and target_prose_identifier in generated_prose_map:
            prose_to_edit = generated_prose_map[target_prose_identifier]
        elif target_prose_identifier == "general_edit_task":
             # For general tasks, maybe concatenate all prose or use full book state?
             # This part needs more definition if general, non-targeted edits are common.
             # For now, we'll assume it means the LLM should be aware of the whole book state.
             prose_to_edit = json.dumps(book_state, indent=2) # Pass the whole book state as "prose"
             self.logger.info(f"'{self.agent_name}' (session: {session_id}) performing general review. Prose_to_edit is the full book state.")
        
        if not prose_to_edit and target_prose_identifier != "general_edit_task":
            self.logger.error(f"'{self.agent_name}' (session: {session_id}) could not find prose for identifier '{target_prose_identifier}' in book_state.generated_prose.")
            yield StreamData(type="error", content=f"Prose for '{target_prose_identifier}' not found for editing.")
            yield StreamData(type="tool_response", content={"message": f"Prose for '{target_prose_identifier}' not found.", "update_book_state": {}})
            return

        # Fetch other relevant context for the editor
        plot_outline = book_state.get("plot_outline", "Not available.")
        character_profiles = book_state.get("character_profiles", {})
        world_details = book_state.get("world_details", {})
        
        # Consider using StoryBibleTool if available and task requires it
        # For example, "Check consistency of character X\'s actions in Chapter 5 against their profile in the story bible."
        # story_bible_info = await self.tool_registry.get_tool("StoryBibleTool").execute(...) 

        prompt = f"""
You are the Editor Agent. Your task is to review and edit a piece of prose based on specific criteria.

Editing Task: {task_description}
Target Prose Identifier: {target_prose_identifier}

Prose to Edit:
---
{prose_to_edit if prose_to_edit else "No specific prose provided for this identifier, this might be a general review task."}
---

Supporting Context from the Book State:
Overall Plot Outline:
{json.dumps(plot_outline, indent=2) if isinstance(plot_outline, dict) else plot_outline}

Character Profiles:
{json.dumps(character_profiles, indent=2) if character_profiles else "No character profiles provided."}

World Details:
{json.dumps(world_details, indent=2) if world_details else "No world details provided."}

Please perform the editing task as described. 
If the task is to provide feedback, structure your feedback clearly.
If the task is to make direct edits, provide the FULL REVISED PROSE for \'{target_prose_identifier}\'. 
Do NOT just provide a list of changes or a diff. Return the complete, edited text.
If the task is general (e.g., "review overall tone") and no specific prose was targeted, provide your feedback or analysis as a text response.

Edited Prose or Feedback:
"""
        yield StreamData(type="llm_prompt", content=prompt)
        self.logger.debug(f"EditorAgent LLM Prompt (session {session_id}, target: {target_prose_identifier}):\\n{prompt}")
        
        edited_content_str = ""
        try:
            edited_content_str = await self.llm.chat_completion_async(
                model_name=self.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config_full.agent_temperature, # Typically lower for editing
                max_tokens=self.config_full.agent_max_tokens, # May need to be large if returning full chapters
            )
            yield StreamData(type="llm_response", content=edited_content_str)
            self.logger.debug(f"EditorAgent LLM Raw Response (session {session_id}, target: {target_prose_identifier}): {edited_content_str[:300]}...")

            if not edited_content_str or not edited_content_str.strip():
                raise ValueError("LLM returned empty or whitespace-only response for editing.")

        except Exception as e:
            self.logger.error(f"'{self.agent_name}' (session: {session_id}, target: {target_prose_identifier}) Error during LLM call: {e}")
            yield StreamData(type="error", content=f"Error during editing task for {target_prose_identifier}: {e}", error_details=str(e))
            yield StreamData(type="tool_response", content={"message": f"Failed to edit/review {target_prose_identifier}: {e}", "update_book_state": {}})
            return

        # Save the edited prose (or feedback) to memory
        # If it was a targeted edit, update the specific prose segment.
        # If it was general feedback, it might be stored differently or just returned.
        
        update_for_orchestrator = {}
        memory_log_output = ""

        if target_prose_identifier != "general_edit_task":
            # Assume edited_content_str is the full revised prose for the target
            update_for_orchestrator = {
                "generated_prose": {
                    target_prose_identifier: edited_content_str
                },
                "edit_summary": { # Add a note about the edit
                     target_prose_identifier: f"Edited based on task: {task_description[:100]}"   
                }
            }
            memory_log_output = f"Edited prose for {target_prose_identifier}. Length: {len(edited_content_str)} chars."
            
            # The Orchestrator will handle saving this updated 'generated_prose' part of the book_state.
            # EditorAgent itself doesn't need to create a new memory segment for the prose text itself if it's just an update.
            # However, we can log the *result* of the edit operation.
            edit_op_content = MemorySegmentContent(
                text=f"Edit task: {task_description}\\nTarget: {target_prose_identifier}\\nResult: First 200 chars of edited content: {edited_content_str[:200]}...",
                tool_name=self.agent_name,
                tool_args={"task_description": task_description, "target_prose_identifier": target_prose_identifier},
                tool_output=memory_log_output
            )
            edit_op_segment = MemorySegment(
                type="EDITOR_OPERATION_LOG",
                source=self.agent_name,
                content=edit_op_content,
                importance=0.6,
                metadata={
                    "session_id": session_id,
                    "user_goal_summary": user_goal_summary,
                    "target_prose_identifier": target_prose_identifier,
                    "agent_name": self.agent_name
                }
            )
            await self.memory.add_memory_segment(edit_op_segment)
            self.logger.info(f"'{self.agent_name}' (session: {session_id}) logged edit operation for '{target_prose_identifier}'.")
            yield StreamData(type="memory_add", content=f"Logged edit operation for '{target_prose_identifier}'.")

        else: # General review task, edited_content_str is feedback
            update_for_orchestrator = {
                "general_edit_feedback": {
                     "last_feedback": edited_content_str,
                     "task": task_description
                }
            }
            memory_log_output = f"General feedback provided. Length: {len(edited_content_str)} chars."
            # Log this feedback as well
            feedback_content = MemorySegmentContent(
                text=edited_content_str,
                tool_name=self.agent_name,
                tool_args={"task_description": task_description},
                tool_output=memory_log_output
            )
            feedback_segment = MemorySegment(
                type="EDITOR_GENERAL_FEEDBACK",
                source=self.agent_name,
                content=feedback_content,
                importance=0.7,
                metadata={"session_id": session_id, "user_goal_summary": user_goal_summary, "agent_name": self.agent_name}
            )
            await self.memory.add_memory_segment(feedback_segment)
            self.logger.info(f"\'{self.agent_name}\' (session: {session_id}) saved general editing feedback to memory.")
            yield StreamData(type="memory_add", content="Saved general editing feedback to memory.")


        final_response_payload = {
            "message": f"'{self.agent_name}' successfully completed editing task for '{target_prose_identifier}'.",
            "edited_content_preview": edited_content_str[:200] + "..." if edited_content_str else "N/A",
            "update_book_state": update_for_orchestrator 
        }
        
        self.logger.info(f"'{self.agent_name}' (session: {session_id}) completed task for '{target_prose_identifier}'.")
        yield StreamData(type="tool_response", content=final_response_payload)

    def get_description(self) -> str:
        return "Reviews and edits generated prose for consistency, clarity, style, grammar, plot coherence, and other specified criteria. Can also provide general feedback on the text."

    def get_config_key(self) -> str:
        return "book_editor"
