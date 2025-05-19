import json
from typing import Any, Dict, Optional, List, AsyncGenerator

from agents.base_agent import BaseAgent
from agents.book_writing_schemas import ChapterProseSchema, ChapterOutlineSchema, CharacterProfileSchema, WorldAnvilSchema
from core.schemas import StreamData

class ProseGenerationAgent(BaseAgent):
    # __init__ is inherited. If a specific model for prose generation is desired:
    # def __init__(self, agent_name: str, config: Dict[str, Any], llm_interface: LLMInterface, memory_manager: MemoryManager, tool_registry: Optional[Any] = None):
    #     super().__init__(agent_name, config, llm_interface, memory_manager, tool_registry)
    #     self.llm_model_name = self.config_full.models.get('prose_generator', self.config_full.models.default)
    #     self.logger.info(f"'{self.agent_name}' initialized with LLM model: {self.llm_model_name}.")

    async def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[StreamData, None]:
        effective_context = context if context is not None else {}
        self.logger.info(f"'{self.agent_name}' received task: {task_description}")

        book_state_slice = effective_context.get("book_writing_state_slice", {})
        project_name = book_state_slice.get("project_name", "Unknown Project")

        # Get chapter outlines (list of ChapterOutlineSchema or dicts)
        chapter_outlines_data = book_state_slice.get("detailed_chapter_outlines", [])
        chapter_outlines_for_prompt = []
        for co_data in chapter_outlines_data:
            if isinstance(co_data, ChapterOutlineSchema):
                chapter_outlines_for_prompt.append(co_data.dict(exclude_none=True))
            elif isinstance(co_data, dict):
                chapter_outlines_for_prompt.append(co_data)  # Assume it's already a dict

        # Get character profiles (list of CharacterProfileSchema or dicts)
        character_profiles_data = book_state_slice.get("character_profiles", [])
        character_profiles_for_prompt = []
        for cp_data in character_profiles_data:
            if isinstance(cp_data, CharacterProfileSchema):
                character_profiles_for_prompt.append(cp_data.dict(exclude_none=True))
            elif isinstance(cp_data, dict):
                character_profiles_for_prompt.append(cp_data)

        # Get world building notes (WorldAnvilSchema object or dict)
        world_anvil_data = book_state_slice.get("world_building_notes", {})
        world_anvil_for_prompt = {}
        if isinstance(world_anvil_data, WorldAnvilSchema):
            world_anvil_for_prompt = world_anvil_data.dict(exclude_none=True)
        elif isinstance(world_anvil_data, dict):
            world_anvil_for_prompt = world_anvil_data

        # Example of how ChapterProseSchema looks for the LLM
        example_prose_schema_json = ChapterProseSchema.schema_json(indent=2)

        prompt = f"""
You are the Prose Generation Agent for the book project \'{project_name}\'. 
Your task is to write compelling narrative prose (story text, dialogue, descriptions) based on the user\'s request and provided context.

User Task: {task_description}
(This task might specify a chapter number, scene, or a general prose generation request based on the outlines.)

Available Context:
1. Detailed Chapter Outlines:
{json.dumps(chapter_outlines_for_prompt, indent=2) if chapter_outlines_for_prompt else "No detailed chapter outlines provided."}

2. Character Profiles:
{json.dumps(character_profiles_for_prompt, indent=2) if character_profiles_for_prompt else "No character profiles provided."}

3. World Anvil / Building Notes:
{json.dumps(world_anvil_for_prompt, indent=2) if world_anvil_for_prompt else "No world building notes provided."}

Based on the user task and all the provided context, please write the prose.
Focus on vivid descriptions, engaging dialogue, and advancing the plot as per the outlines.
Ensure the tone is consistent with the genre and existing material.

Your response MUST be a single, valid JSON object containing a single key "generated_prose".
The value of "generated_prose" must be a LIST of objects, where each object conforms to the following Pydantic schema (ChapterProseSchema):
{example_prose_schema_json}

Each object in the list should represent a distinct block of prose (e.g., a scene, a part of a chapter).
- "chapter_number": (Required, integer) The chapter this prose belongs to.
- "scene_number": (Optional, integer) If applicable, the scene number within the chapter.
- "prose_text": (Required, string) The actual generated prose.
- "version": (Required, integer, default 1) The version number of this prose block.
- "status": (Optional, string, e.g., "draft", "revised") Current status of this prose.
- "notes": (Optional, string) Any notes related to this prose.

Example of expected output format:
{
  "generated_prose": [
    {
      "chapter_number": 1,
      "scene_number": 1,
      "prose_text": "The wind howled around the crumbling tower...",
      "version": 1,
      "status": "draft",
      "notes": "Opening scene for Chapter 1."
    },
    {
      "chapter_number": 1,
      "scene_number": 2,
      "prose_text": "Later, Elara met Kaelen in the tavern...",
      "version": 1,
      "status": "draft"
    }
  ]
}
Ensure your entire output is ONLY the JSON object. Do not include any explanatory text before or after it.
"""
        self.logger.debug(f"ProseGenerationAgent LLM Prompt for \'{project_name}\':\n{prompt}")

        yield StreamData(type="info", content="Generating prose...")

        try:
            llm_response_str = await self.llm.chat_completion_async(
                messages=[{"role": "user", "content": prompt}],
                format="json"  # Request JSON output format
            )
            self.logger.debug(f"ProseGenerationAgent LLM Raw Response for \'{project_name}\': {llm_response_str}")

            parsed_llm_json = json.loads(llm_response_str)
            if not isinstance(parsed_llm_json, dict) or "generated_prose" not in parsed_llm_json:
                raise ValueError("LLM response was not a JSON object with \'generated_prose\' key.")

            raw_prose_list = parsed_llm_json["generated_prose"]
            if not isinstance(raw_prose_list, list):
                raise ValueError("\'generated_prose\' key did not contain a list.")

            validated_prose_objects = []
            for prose_data in raw_prose_list:
                if isinstance(prose_data, dict):
                    try:
                        validated_prose = ChapterProseSchema(**prose_data).dict()
                        validated_prose_objects.append(validated_prose)
                        yield StreamData(type="info", content=f"Generated prose for Chapter {prose_data.get('chapter_number', 'unknown')}, Scene {prose_data.get('scene_number', 'N/A')}")
                    except Exception as e:  # Pydantic ValidationError
                        yield StreamData(type="warning", content=f"Skipping a prose object due to validation error: {e}")
                        self.logger.warning(f"Skipping a prose object due to validation error: {e}. Data: {prose_data}")
                else:
                    yield StreamData(type="warning", content=f"Skipping non-dictionary item in generated_prose list.")
                    self.logger.warning(f"Skipping an item in \'generated_prose\' list as it is not a dictionary: {prose_data}")

            if not validated_prose_objects:
                self.logger.info(f"No valid prose objects were generated or validated for task: {task_description}")
                yield StreamData(type="error", content="No valid prose was generated based on the task.")
                return

            output_for_orchestrator = {"generated_prose": validated_prose_objects}
            self.logger.info(f"'{self.agent_name}' completed task for '{project_name}'. Generated {len(validated_prose_objects)} prose object(s).")
            yield StreamData(type="tool_response", content=json.dumps(output_for_orchestrator), tool_name="book_prose_generator")

        except json.JSONDecodeError as e:
            self.logger.error(f"'{self.agent_name}' JSONDecodeError for '{project_name}': {e}. Response: {llm_response_str}")
            yield StreamData(type="error", content=f"Failed to parse JSON from LLM response: {e}")
        except Exception as e:  # Includes Pydantic ValidationError and ValueErrors from checks
            self.logger.error(f"'{self.agent_name}' Error processing LLM response for '{project_name}': {e}. Response: {llm_response_str}", exc_info=True)
            yield StreamData(type="error", content=f"Failed to generate prose: {e}")
