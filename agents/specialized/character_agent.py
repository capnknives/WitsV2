from typing import Any, Dict, Optional, List
import json
import logging # Though BaseAgent initializes logger, direct use might be needed if extending
from core.schemas import MemorySegment, MemorySegmentContent # Ensure this is the correct import path

from agents.base_agent import BaseAgent

class CharacterDevelopmentAgent(BaseAgent):
    # No __init__ method, inherits from BaseAgent

    async def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> str:
        self.logger.info(f"'{self.agent_name}' received task for character development: {task_description}")
        if context is None:
            context = {}

        session_id = context.get("session_id", "unknown_session")

        system_prompt = (
            "You are an expert character creator and storyteller. "
            "Your task is to generate detailed and compelling character profiles based on user requests. "
            "Ensure each profile is rich with information that can be used to drive a narrative."
        )

        example_json_structure = '''
[
  {
    "name": "Character Name",
    "role_in_story": "e.g., Protagonist, Antagonist, Mentor, Supporting",
    "appearance": "Brief but evocative description of physical appearance and typical attire.",
    "personality_traits": ["trait1 (e.g., cynical)", "trait2 (e.g., brave)", "trait3 (e.g., secretive)"],
    "backstory": "A detailed history of the character, including formative events, significant relationships, and past traumas or triumphs that shape their current self.",
    "motivations": ["primary motivation (e.g., seek revenge)", "secondary motivation (e.g., protect family)"],
    "flaws": ["major flaw (e.g., arrogance)", "minor flaw (e.g., impatient)"],
    "strengths": ["key strength (e.g., expert strategist)", "secondary strength (e.g., loyal friend)"],
    "potential_arc": "Describe how the character might change, grow, or devolve throughout a story based on their traits and potential plot interactions."
  }
  // ... more characters if requested
]
'''
        user_prompt = (
            f"Task: {task_description}\\n\\n"
            f"Please generate detailed character profiles based on this task. "
            f"Return your response as a single, valid JSON list, where each element is an object representing a character. "
            f"Use the following structure for each character object:\\n"
            f"{example_json_structure}\\n\\n"
            f"Ensure your entire output is ONLY the JSON list. Do not include any explanatory text before or after the JSON list."
        )

        try:
            self.logger.debug(f"CharacterDevelopmentAgent LLM System Prompt:\\n{system_prompt}")
            self.logger.debug(f"CharacterDevelopmentAgent LLM User Prompt:\\n{user_prompt}")

            llm_response_str = await self.llm.chat_completion_async(
                model_name=self.agent_config.get("model_name", self.config_full.models.default),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.agent_config.get("temperature", 0.75)
            )
            self.logger.debug(f"CharacterDevelopmentAgent LLM Raw Response:\\n{llm_response_str}")

            character_profiles_data: Any = None
            try:
                character_profiles_data = json.loads(llm_response_str)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse LLM response as JSON: {e}. Raw response: {llm_response_str}")
                character_profiles_data = {"error": "Failed to parse LLM response as JSON", "raw_response": llm_response_str}

            saved_character_names: List[str] = []
            processed_profiles_for_state = [] # Stores successfully processed dicts

            if isinstance(character_profiles_data, list):
                for profile_data in character_profiles_data:
                    if isinstance(profile_data, dict):
                        profile_json_str = json.dumps(profile_data, indent=2)
                        char_name = profile_data.get("name", f"UnknownCharacter_{len(saved_character_names) + 1}")
                        
                        try:
                            # Attempting add_memory_segment first as it's more common in the project
                            await self.memory.add_memory_segment(MemorySegment(
                                type="CHARACTER_PROFILE",
                                source=self.agent_name,
                                content=MemorySegmentContent(text=profile_json_str, tool_output=f"Profile for character: {char_name}"),
                                metadata={"character_name": char_name, "session_id": session_id, "task": task_description[:100]}
                            ))
                            saved_character_names.append(char_name)
                            processed_profiles_for_state.append(profile_data)
                            self.logger.info(f"Saved character profile for '{char_name}' to memory using add_memory_segment.")
                        except Exception as mem_e: # Catching generic exception if add_memory_segment fails
                            self.logger.error(f"Failed to save character profile for '{char_name}' to memory: {mem_e}")
                    else:
                        self.logger.warning(f"Skipping an item in character_profiles_data as it is not a dictionary: {profile_data}")
            
            elif isinstance(character_profiles_data, dict) and "error" in character_profiles_data:
                # This case is when JSON parsing itself failed.
                self.logger.error(f"LLM data parsing resulted in an error structure: {character_profiles_data.get('error')}")
                # processed_profiles_for_state remains empty, saved_character_names remains empty
            
            else:
                # LLM response was not a list, nor the error dict from parsing.
                self.logger.warning(f"LLM response was not a list of profiles or a parsing error dict. Type: {type(character_profiles_data)}, Data: {str(character_profiles_data)[:200]}")
                # Ensure character_profiles_data is an error structure for consistent return handling
                if not (isinstance(character_profiles_data, dict) and 'error' in character_profiles_data):
                    character_profiles_data = {"error": "Unexpected LLM response format", "raw_response": str(character_profiles_data)[:500]}


            message: str
            update_book_state: Dict[str, Any]

            if isinstance(character_profiles_data, dict) and "error" in character_profiles_data:
                message = f"CharacterDevelopmentAgent encountered an error: {character_profiles_data.get('error')}"
                update_book_state = {"character_development_error": character_profiles_data}
            elif saved_character_names: # Successfully processed and saved some profiles
                message = f"CharacterDevelopmentAgent processed task: '{task_description[:50]}...'. Developed {len(saved_character_names)} characters: {', '.join(saved_character_names)}."
                update_book_state = {"character_profiles_developed": processed_profiles_for_state} # Use the list of dicts
            else: # No errors in parsing, but no characters saved (e.g., LLM returned empty list or non-profile data that was skipped)
                message = f"CharacterDevelopmentAgent processed task: '{task_description[:50]}...'. No new character profiles were successfully generated or saved from the LLM response."
                # If character_profiles_data was an empty list from LLM, reflect that.
                update_book_state = {"character_profiles_developed": character_profiles_data if isinstance(character_profiles_data, list) else []}


            self.logger.info(message)
            return json.dumps({
                "message": message,
                "update_book_state": update_book_state
            })

        except Exception as e:
            self.logger.exception(f"An unexpected error occurred in CharacterDevelopmentAgent run method: {e}")
            return json.dumps({
                "message": f"CharacterDevelopmentAgent encountered an unexpected error: {str(e)}",
                "update_book_state": {"character_development_error": {"error": str(e), "details": "Unexpected exception in agent."}}
            })
