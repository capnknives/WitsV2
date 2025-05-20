from typing import Any, Dict, Optional, AsyncGenerator
import json

from pydantic import ValidationError, parse_obj_as

from agents.base_agent import BaseAgent
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager # Kept for __init__ signature
from core.schemas import StreamData
from agents.book_writing_schemas import ChapterProseSchema, BookWritingState

class EditorAgent(BaseAgent):
    def __init__(self, agent_name: str, config: Any, llm_interface: LLMInterface, memory_manager: Optional[MemoryManager] = None, tool_registry: Optional[Any] = None):
        super().__init__(agent_name, config, llm_interface, memory_manager, tool_registry)
        self.llm_model_name = self.llm.model_name
        self.logger.info(f"'{self.agent_name}' initialized. It will use the LLM model: {self.llm_model_name} from its LLMInterface.")
    
    @staticmethod
    def get_description() -> str:
        return "I'm the grammar police and style guru! \\o/ I take your prose, make it shine, " + \
               "and return it all nicely packaged in JSON (because we're fancy like that =D). " + \
               "No typo is safe from my watchful eye! ^_^"

    @staticmethod
    def get_config_schema() -> Dict[str, Any]:
        return {
            "type": "object",  # Keep it structured, keep it clean! >.>
            "properties": {
                "description": {"type": "string"},  # Tell me what you want, what you really really want!
                "additional_instructions": {"type": "string"}  # Extra wisdom for the editing process =P
            }
        }

    async def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[StreamData, None]:
        effective_context = context if context is not None else {}
        session_id = effective_context.get("session_id", "unknown_session")
        
        book_writing_state_slice: Dict = effective_context.get("book_writing_state_slice", {})
        project_name = book_writing_state_slice.get("project_name", "Unknown Project")
        
        target_prose_id_str = effective_context.get("target_prose_id")

        if not target_prose_id_str:
            self.logger.error(f"'{self.agent_name}' (session: {session_id}) 'target_prose_id' (chapter number string) not found in context.")
            yield StreamData(type="error", content="No target chapter specified for editing")
            return

        self.logger.info(f"'{self.agent_name}' (session: {session_id}) received task: '{task_description}' for target chapter: '{target_prose_id_str}'.")

        all_generated_prose_dicts: Dict[str, Dict] = book_writing_state_slice.get("generated_prose", {})
        original_prose_dict = all_generated_prose_dicts.get(target_prose_id_str)
        
        if not original_prose_dict:
            self.logger.error(f"'{self.agent_name}' (session: {session_id}) could not find prose for chapter ID '{target_prose_id_str}'.")
            yield StreamData(type="error", content=f"Could not find prose for chapter {target_prose_id_str}")
            return

        try:
            original_prose_obj = ChapterProseSchema.parse_obj(original_prose_dict)
        except ValidationError as ve:
            self.logger.error(f"'{self.agent_name}' (session: {session_id}) failed to parse original prose for chapter ID '{target_prose_id_str}': {ve}")
            yield StreamData(type="error", content=f"Invalid original prose data for chapter {target_prose_id_str}")
            return

        chapter_outlines = book_writing_state_slice.get("detailed_chapter_outlines", [])
        character_profiles = book_writing_state_slice.get("character_profiles", [])
        world_anvil_notes = book_writing_state_slice.get("world_building_notes", {})

        prompt = f"""You are the Editor Agent for the project: "{project_name}".
Your task is to edit the provided chapter prose based on the following instructions:
Editing Task: {task_description}
Target Chapter Number: {original_prose_obj.chapter_number} (ID: {target_prose_id_str})

Original Chapter Prose (Scenes) to Edit:
---
Chapter Number: {original_prose_obj.chapter_number}
Scenes:
{json.dumps(original_prose_obj.scenes, indent=2)}
---

Supporting Context:
Chapter Outlines (relevant excerpts):
{json.dumps([o for o in chapter_outlines if o.get('chapter_number') == original_prose_obj.chapter_number or len(chapter_outlines) < 3][:2], indent=2)}
Character Profiles (names of first few):
{json.dumps([cp.get('name', 'N/A') for cp in character_profiles[:3]], indent=2)}
World Anvil Notes (keys of first few):
{json.dumps(list(world_anvil_notes.keys())[:5], indent=2)}

Please perform the editing task. Your response MUST be a single JSON object.
The JSON object can have two optional keys:

1.  "edited_prose_object": If you make changes to the chapter's scenes, include the *complete, updated* ChapterProseSchema object here.
    - The schema for ChapterProseSchema is: {ChapterProseSchema.schema_json(indent=2)}
    - IMPORTANT: The "chapter_number" in the "edited_prose_object" MUST be {original_prose_obj.chapter_number}.
2.  "revision_notes_additions": A list of strings, where each string is a specific note or comment about the revisions made or suggestions for further changes.

Example 1 (prose edited, notes added):
{{
  "edited_prose_object": {{
    "chapter_number": {original_prose_obj.chapter_number},
    "scenes": {{
      "Scene 1: The Revised Opening": "The fully revised text for scene 1...",
      "Scene 2: A New Development": "Text for a newly added or heavily modified scene 2..."
    }}
  }},
  "revision_notes_additions": ["Corrected narrative flow in Scene 1.", "Expanded on character X's reaction in Scene 2."]
}}

Example 2 (only notes added, no direct prose edits):
{{
  "revision_notes_additions": ["Suggest breaking down the long scene into two for better pacing.", "The dialogue in the existing Scene 1 feels a bit stilted."]
}}

Example 3 (only prose edited, no specific notes):
{{
  "edited_prose_object": {{
    "chapter_number": {original_prose_obj.chapter_number},
    "scenes": {{
      "Scene 1: The Opening": "The original scene 1 text with minor tweaks...",
      "Scene 2: Conflict": "The original scene 2 text, also with minor adjustments."
    }}
  }}
}}

If you do not make any direct edits to the prose, do not include the "edited_prose_object" key.
If you do not have any revision notes, do not include the "revision_notes_additions" key.
If no changes or notes are applicable, return an empty JSON object {{}}.
Ensure your output is valid JSON.
"""
        self.logger.debug(f"EditorAgent LLM Prompt (session {session_id}, target chapter: {target_prose_id_str}):\n{prompt[:1000]}...")
        
        yield StreamData(type="info", content=f"Editing chapter {target_prose_id_str}...")

        try:
            llm_response_str = await self.llm.chat_completion_async(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature if self.config and self.config.temperature is not None else 0.7,
                max_tokens=self.config.max_tokens if self.config and self.config.max_tokens is not None else 2048,
                json_mode=True
            )
            self.logger.debug(f"EditorAgent LLM Raw Response (session {session_id}, target chapter: {target_prose_id_str}): {llm_response_str[:300]}...")

            if not llm_response_str or not llm_response_str.strip():
                yield StreamData(type="error", content="Empty response from LLM")
                return
            
            response_data = json.loads(llm_response_str)

        except json.JSONDecodeError as jde:
            self.logger.error(f"'{self.agent_name}' (session: {session_id}, target chapter: {target_prose_id_str}) JSONDecodeError: {jde}. Response: {llm_response_str}")
            yield StreamData(type="error", content="LLM output was not valid JSON")
            return
        except Exception as e:
            self.logger.error(f"'{self.agent_name}' (session: {session_id}, target chapter: {target_prose_id_str}) Error during LLM call or parsing: {e}")
            yield StreamData(type="error", content=f"Error during editing task for chapter {target_prose_id_str}")
            return

        orchestrator_payload = {}
        
        edited_prose_data = response_data.get("edited_prose_object")
        if edited_prose_data:
            try:
                # Ensure chapter_number from LLM matches the original, as per instruction
                if edited_prose_data.get("chapter_number") != original_prose_obj.chapter_number:
                    self.logger.warning(f"LLM provided chapter_number {edited_prose_data.get('chapter_number')} for edited prose, "
                                    f"expected {original_prose_obj.chapter_number}. Overriding.")
                    edited_prose_data["chapter_number"] = original_prose_obj.chapter_number 
                                
                validated_edited_prose_obj = ChapterProseSchema.parse_obj(edited_prose_data)
                orchestrator_payload["edited_prose_object"] = validated_edited_prose_obj.dict(exclude_none=True)
                self.logger.info(f"'{self.agent_name}' (session: {session_id}) successfully validated edited prose for chapter '{target_prose_id_str}'.")
                yield StreamData(type="info", content=f"Successfully edited chapter {target_prose_id_str}")
            except ValidationError as ve:
                self.logger.error(f"'{self.agent_name}' (session: {session_id}, target chapter: {target_prose_id_str}) ValidationError for edited_prose_object: {ve}. Data: {edited_prose_data}")
                yield StreamData(type="error", content=f"Edited prose validation failed: {ve}")
            except Exception as ex:
                self.logger.error(f"'{self.agent_name}' (session: {session_id}, target chapter: {target_prose_id_str}) Error processing edited_prose_object: {ex}. Data: {edited_prose_data}")
                yield StreamData(type="error", content=f"Error processing edited prose: {ex}")

        revision_notes = response_data.get("revision_notes_additions")
        if revision_notes is not None:
            if isinstance(revision_notes, list) and all(isinstance(note, str) for note in revision_notes):
                orchestrator_payload["revision_notes_additions"] = revision_notes
                for note in revision_notes:
                    yield StreamData(type="info", content=f"Added revision note: {note}")
            else:
                self.logger.warning(f"'{self.agent_name}' (session: {session_id}) received invalid revision_notes_additions format")
                yield StreamData(type="warning", content="Invalid revision notes format")
        
        if not orchestrator_payload:
            yield StreamData(type="info", content="No changes or notes were needed")
            return

        self.logger.info(f"'{self.agent_name}' (session: {session_id}) completed task for chapter '{target_prose_id_str}'.")
        yield StreamData(type="tool_response", content=json.dumps(orchestrator_payload))
