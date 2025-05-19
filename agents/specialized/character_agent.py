from typing import Any, Dict, Optional, List
import json
import logging
from agents.base_agent import BaseAgent
from agents.book_writing_schemas import CharacterProfileSchema # For typing and validation

class CharacterDevelopmentAgent(BaseAgent):
    # Renamed from CharacterAgent to CharacterDevelopmentAgent for clarity if this was the original name
    # If it was CharacterAgent, then this class name should be CharacterAgent

    async def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> str:
        self.logger.info(f"'{self.agent_name}' received task for character development: {task_description}")
        if context is None:
            context = {}

        book_state_slice = context.get("book_writing_state_slice", {})
        project_name = book_state_slice.get("project_name", "Unknown Project")
        existing_profiles_data = book_state_slice.get("character_profiles", []) # List of dicts or CharacterProfileSchema objects
        
        # For the prompt, create a summary of existing characters
        existing_chars_summary_for_prompt = []
        for prof in existing_profiles_data:
            if isinstance(prof, dict):
                existing_chars_summary_for_prompt.append(f"Name: {prof.get('name', 'N/A')}, Role: {prof.get('role_in_story', 'N/A')}")
            # Add handling if prof is CharacterProfileSchema object, accessing attributes
            # elif isinstance(prof, CharacterProfileSchema):
            #     existing_chars_summary_for_prompt.append(f"Name: {prof.name}, Role: {prof.role_in_story}")

        system_prompt = (
            f"You are an expert character creator for the book project '{project_name}'. "
            "Your task is to generate or update detailed and compelling character profiles based on user requests. "
            "Ensure each profile is rich with information that can be used to drive a narrative."
        )

        # Using CharacterProfileSchema.schema_json() for a robust example structure
        example_json_structure = CharacterProfileSchema.schema_json(indent=2)
        # The LLM should return a list of these objects
        # So the top-level structure is List[CharacterProfileSchema]

        user_prompt = (
            f"Task: {task_description}\\n\\n"
            f"Project: '{project_name}'\\n"
            f"Existing Characters Summary (Name, Role):\\n{json.dumps(existing_chars_summary_for_prompt, indent=2) if existing_chars_summary_for_prompt else 'No existing characters provided in this slice.'}\\n\\n"
            f"Please generate or update character profiles based on the task. "
            f"If updating an existing character, ensure you use their correct name and provide the complete updated profile. "
            f"Return your response as a single, valid JSON list, where each element is an object representing a character. "
            f"Each character object MUST conform to the following Pydantic schema:\\n"
            f"{example_json_structure}\\n\\n"
            f"Ensure your entire output is ONLY the JSON list. Do not include any explanatory text before or after the JSON list."
        )

        try:
            self.logger.debug(f"CharacterDevelopmentAgent LLM System Prompt:\\n{system_prompt}")
            self.logger.debug(f"CharacterDevelopmentAgent LLM User Prompt:\\n{user_prompt}")

            llm_response_str = await self.llm.chat_completion_async(
                # model_name=self.agent_config.get("model_name", self.config_full.models.default),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # temperature=self.agent_config.get("temperature", 0.75) # Assuming these are in llm.chat_completion_async
            )
            self.logger.debug(f"CharacterDevelopmentAgent LLM Raw Response:\\n{llm_response_str}")

            generated_character_profiles = []
            raw_llm_data = None
            try:
                raw_llm_data = json.loads(llm_response_str)
                if isinstance(raw_llm_data, list):
                    for profile_data in raw_llm_data:
                        if isinstance(profile_data, dict):
                            try:
                                # Validate and convert to dict using Pydantic model
                                validated_profile = CharacterProfileSchema(**profile_data).dict()
                                generated_character_profiles.append(validated_profile)
                            except Exception as e: # Pydantic ValidationError
                                self.logger.warning(f"Skipping a character profile due to validation error: {e}. Data: {profile_data}")
                        else:
                            self.logger.warning(f"Skipping an item in LLM response list as it is not a dictionary: {profile_data}")
                else:
                    self.logger.error(f"LLM response was not a list as expected. Raw response: {llm_response_str}")

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse LLM response as JSON: {e}. Raw response: {llm_response_str}")
                # Return error structure for orchestrator
                return json.dumps({"error": "Failed to parse LLM response as JSON", "raw_response": llm_response_str})

            output_for_orchestrator = {}
            if generated_character_profiles:
                output_for_orchestrator["character_profiles"] = generated_character_profiles
            
            if not output_for_orchestrator:
                self.logger.info(f"CharacterAgent made no changes or failed to generate valid character profiles for task: {task_description}")
                return json.dumps({"message": "No valid character profiles were generated or updated based on the task."}) 

            self.logger.info(f"'{self.agent_name}' completed. Generated/updated {len(generated_character_profiles)} profiles for project '{project_name}'.")
            return json.dumps(output_for_orchestrator, default=str)

        except Exception as e:
            self.logger.exception(f"An unexpected error occurred in CharacterDevelopmentAgent for project '{project_name}': {e}")
            return json.dumps({"error": str(e), "message": "CharacterDevelopmentAgent encountered an internal error."})
