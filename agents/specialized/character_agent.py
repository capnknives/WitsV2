from typing import Any, Dict, Optional, List, AsyncGenerator
import json
import logging
from agents.base_agent import BaseAgent
from agents.book_writing_schemas import CharacterProfileSchema
from core.json_utils import safe_json_loads, balance_json_braces # For typing and validation
from core.schemas import StreamData

class CharacterDevelopmentAgent(BaseAgent):
    async def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[StreamData, None]:
        self.logger.info(f"'{self.agent_name}' received task for character development: {task_description}")
        if context is None:
            context = {}

        book_state_slice = context.get("book_writing_state_slice", {})
        project_name = book_state_slice.get("project_name", "Unknown Project")
        existing_profiles_data = book_state_slice.get("character_profiles", [])
        
        existing_chars_summary_for_prompt = []
        for prof in existing_profiles_data:
            if isinstance(prof, dict):
                existing_chars_summary_for_prompt.append(f"Name: {prof.get('name', 'N/A')}, Role: {prof.get('role_in_story', 'N/A')}")

        system_prompt = (
            f"You are an expert character creator for the book project '{project_name}'. "
            "Your task is to generate or update detailed and compelling character profiles based on user requests. "
            "Ensure each profile is rich with information that can be used to drive a narrative."
        )

        example_json_structure = CharacterProfileSchema.schema_json(indent=2)

        user_prompt = (
            f"Task: {task_description}\n\n"
            f"Project: '{project_name}'\n"
            f"Existing Characters Summary (Name, Role):\n{json.dumps(existing_chars_summary_for_prompt, indent=2) if existing_chars_summary_for_prompt else 'No existing characters provided in this slice.'}\n\n"
            f"Please generate or update character profiles based on the task. "
            f"If updating an existing character, ensure you use their correct name and provide the complete updated profile. "
            f"Return your response as a single, valid JSON list, where each element is an object representing a character. "
            f"Each character object MUST conform to the following Pydantic schema:\n"
            f"{example_json_structure}\n\n"
            f"Ensure your entire output is ONLY the JSON list. Do not include any explanatory text before or after the JSON list."
        )

        try:
            self.logger.debug(f"CharacterDevelopmentAgent LLM System Prompt:\n{system_prompt}")
            self.logger.debug(f"CharacterDevelopmentAgent LLM User Prompt:\n{user_prompt}")

            yield StreamData(type="info", content="Generating character profiles...")

            llm_response_str = await self.llm.chat_completion_async(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            self.logger.debug(f"CharacterDevelopmentAgent LLM Raw Response:\n{llm_response_str}")

            generated_character_profiles = []
            raw_llm_data = safe_json_loads(llm_response_str, session_id=context.get("session_id", "unknown"))
            if isinstance(raw_llm_data, list):
                for profile_data in raw_llm_data:
                    if isinstance(profile_data, dict):
                        try:
                            validated_profile = CharacterProfileSchema(**profile_data).dict()
                            generated_character_profiles.append(validated_profile)
                            yield StreamData(type="info", content=f"Generated profile for character: {profile_data.get('name', 'unnamed')}")
                        except Exception as e:
                            self.logger.warning(f"Skipping a character profile due to validation error: {e}. Data: {profile_data}")
                            yield StreamData(type="warning", content=f"Skipping invalid character profile: {e}")
                    else:
                        self.logger.warning(f"Skipping an item in LLM response list as it is not a dictionary: {profile_data}")
                        yield StreamData(type="warning", content="Skipped invalid profile data format")

            else:
                self.logger.error(f"LLM response was not a list as expected. Raw response: {llm_response_str}")
                yield StreamData(type="error", content="LLM response was not in the expected format")
                return

            if not generated_character_profiles:
                self.logger.info(f"CharacterAgent made no changes or failed to generate valid character profiles for task: {task_description}")
                yield StreamData(type="info", content="No valid character profiles were generated or updated")
                return

            output_for_orchestrator = {"character_profiles": generated_character_profiles}
            self.logger.info(f"'{self.agent_name}' completed. Generated/updated {len(generated_character_profiles)} profiles for project '{project_name}'.")
            yield StreamData(type="tool_response", content=json.dumps(output_for_orchestrator, default=str))

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}. Raw response: {llm_response_str}")
            yield StreamData(type="error", content=f"Failed to parse LLM response: {e}")
        except Exception as e:
            self.logger.error(f"'{self.agent_name}' Error during character generation: {e}. Response: {llm_response_str}", exc_info=True)
            yield StreamData(type="error", content=f"Error during character generation: {e}")
