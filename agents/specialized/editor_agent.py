import json
from typing import Any, Dict, Optional, List

from pydantic import ValidationError, parse_obj_as

from agents.base_agent import BaseAgent
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager # Kept for __init__ signature
from agents.book_writing_schemas import ChapterProseSchema, BookWritingState # Assuming BookWritingState might be useful for context type hint

class EditorAgent(BaseAgent):
    def __init__(self, agent_name: str, config: Any, llm_interface: LLMInterface, memory_manager: MemoryManager, tool_registry: Optional[Any] = None):
        super().__init__(agent_name, config, llm_interface, memory_manager, tool_registry)
        # config is an AgentProfileConfig object. It does not directly have a 'models' attribute.
        # The model name should be determined from config.llm_model_name or a global default.
        # The BaseAgent already sets self.agent_config which contains the model_name.
        # We need to ensure that the AppConfig (global) is accessible if we need its .models.default as a fallback.
        # For now, let's assume the specific model for the editor is in its profile or it uses the agent's default.
        
        # If the agent_profile (self.config_full which is AgentProfileConfig) has llm_model_name, use it.
        # Otherwise, this agent might need a way to access the global AppConfig.models.default.
        # The BaseAgent._get_agent_specific_config tries to set a model_name in self.agent_config.
        
        # Let's simplify: The LLMInterface passed to the agent is already configured with a model.
        # If this agent *must* use a *different* model than the one its LLMInterface is set up with,
        # then it needs to request a new LLMInterface or have the global AppConfig.
        # For now, we assume the provided llm_interface is adequate or its model is what we use.
        self.llm_model_name = self.llm.model_name # Use the model from the provided LLMInterface
        self.logger.info(f"'{self.agent_name}' initialized. It will use the LLM model: {self.llm_model_name} from its LLMInterface.")

    @staticmethod
    def get_description() -> str:
        return "An agent specialized in reviewing and editing chapter prose. It takes a specific chapter's prose, applies edits based on the task, and returns the updated chapter prose object and/or revision notes in a structured JSON format."

    @staticmethod
    def get_config_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "additional_instructions": {"type": "string"}
            }
        }

    def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> str:
        effective_context = context if context is not None else {}
        session_id = effective_context.get("session_id", "unknown_session")
        
        book_writing_state_slice: Dict = effective_context.get("book_writing_state_slice", {})
        project_name = book_writing_state_slice.get("project_name", "Unknown Project")
        
        # target_prose_id is expected to be the chapter_number as a string (e.g., "5")
        target_prose_id_str = effective_context.get("target_prose_id")

        if not target_prose_id_str:
            self.logger.error(f"'{self.agent_name}' (session: {session_id}) 'target_prose_id' (chapter number string) not found in context.")
            return json.dumps({"error": "'target_prose_id' not provided in context.", "details": "EditorAgent requires a specific chapter number string to identify the prose to edit."})

        self.logger.info(f"'{self.agent_name}' (session: {session_id}) received task: '{task_description}' for target chapter: '{target_prose_id_str}'.")

        all_generated_prose_dicts: Dict[str, Dict] = book_writing_state_slice.get("generated_prose", {})
        original_prose_dict = all_generated_prose_dicts.get(target_prose_id_str)
        
        if not original_prose_dict:
            self.logger.error(f"'{self.agent_name}' (session: {session_id}) could not find prose for chapter ID '{target_prose_id_str}'.")
            return json.dumps({"error": f"Prose for chapter ID '{target_prose_id_str}' not found for editing."})

        try:
            original_prose_obj = ChapterProseSchema.parse_obj(original_prose_dict)
        except ValidationError as ve:
            self.logger.error(f"'{self.agent_name}' (session: {session_id}) failed to parse original prose for chapter ID '{target_prose_id_str}': {ve}")
            return json.dumps({"error": f"Invalid original prose data for chapter ID '{target_prose_id_str}'.", "details": str(ve)})

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
        self.logger.debug(f"EditorAgent LLM Prompt (session {session_id}, target chapter: {target_prose_id_str}):\\n{prompt[:1000]}...")
        
        try:
            llm_response_str = self.llm.chat_completion(
                model_name=self.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config_full.agent_temperature,
                max_tokens=self.config_full.agent_max_tokens,
                json_mode=True
            )
            self.logger.debug(f"EditorAgent LLM Raw Response (session {session_id}, target chapter: {target_prose_id_str}): {llm_response_str[:300]}...")

            if not llm_response_str or not llm_response_str.strip():
                raise ValueError("LLM returned empty or whitespace-only response.")
            
            response_data = json.loads(llm_response_str)

        except json.JSONDecodeError as jde:
            self.logger.error(f"'{self.agent_name}' (session: {session_id}, target chapter: {target_prose_id_str}) JSONDecodeError: {jde}. Response: {llm_response_str}")
            return json.dumps({"error": "LLM output was not valid JSON.", "details": str(jde), "llm_response": llm_response_str})
        except Exception as e:
            self.logger.error(f"'{self.agent_name}' (session: {session_id}, target chapter: {target_prose_id_str}) Error during LLM call or parsing: {e}")
            return json.dumps({"error": f"Error during editing task for chapter '{target_prose_id_str}'.", "details": str(e)})

        orchestrator_payload = {}
        
        edited_prose_data = response_data.get("edited_prose_object")
        if edited_prose_data:
            try:
                # Ensure chapter_number from LLM matches the original, as per instruction
                if edited_prose_data.get("chapter_number") != original_prose_obj.chapter_number:
                    self.logger.warning(f"LLM provided chapter_number {edited_prose_data.get('chapter_number')} for edited prose, "
                                        f"expected {original_prose_obj.chapter_number}. Overriding or warning.")
                    # Option: Force correct chapter_number or reject if critical
                    edited_prose_data["chapter_number"] = original_prose_obj.chapter_number 
                                
                validated_edited_prose_obj = ChapterProseSchema.parse_obj(edited_prose_data)
                orchestrator_payload["edited_prose_object"] = validated_edited_prose_obj.dict(exclude_none=True)
                self.logger.info(f"'{self.agent_name}' (session: {session_id}) successfully validated edited prose for chapter '{target_prose_id_str}'.")
            except ValidationError as ve:
                self.logger.error(f"'{self.agent_name}' (session: {session_id}, target chapter: {target_prose_id_str}) ValidationError for edited_prose_object: {ve}. Data: {edited_prose_data}")
                orchestrator_payload["edited_prose_error"] = f"Edited prose validation failed: {str(ve)}"
            except Exception as ex:
                self.logger.error(f"'{self.agent_name}' (session: {session_id}, target chapter: {target_prose_id_str}) Error processing edited_prose_object: {ex}. Data: {edited_prose_data}")
                orchestrator_payload["edited_prose_error"] = f"Error processing edited prose: {str(ex)}"

        revision_notes = response_data.get("revision_notes_additions")
        if revision_notes is not None:
            if isinstance(revision_notes, list) and all(isinstance(note, str) for note in revision_notes):
                orchestrator_payload["revision_notes_additions"] = revision_notes
                self.logger.info(f"'{self.agent_name}' (session: {session_id}) successfully processed {len(revision_notes)} revision notes for chapter '{target_prose_id_str}'.")
            else:
                self.logger.warning(f"'{self.agent_name}' (session: {session_id}, target chapter: {target_prose_id_str}) 'revision_notes_additions' was not a list of strings. Received: {type(revision_notes)}")
                orchestrator_payload["revision_notes_error"] = "'revision_notes_additions' was not a list of strings."
        
        if not orchestrator_payload:
             self.logger.info(f"'{self.agent_name}' (session: {session_id}, target chapter: {target_prose_id_str}) LLM returned no actionable data (empty JSON or only errors not processed into payload).")
             # Return empty dict if no valid data, or specific error structure if only errors occurred
             if "edited_prose_error" in orchestrator_payload and not "revision_notes_additions" in orchestrator_payload:
                 return json.dumps({"error": "LLM response processing failed for prose.", "details": orchestrator_payload})
             return json.dumps({})


        self.logger.info(f"'{self.agent_name}' (session: {session_id}) completed task for chapter '{target_prose_id_str}'. Returning payload to orchestrator.")
        return json.dumps(orchestrator_payload)
