import json
from typing import Any, Dict, Optional, AsyncGenerator
import json

from agents.base_agent import BaseAgent
from agents.book_writing_schemas import WorldAnvilSchema # For typing and validation
from core.schemas import StreamData

class WorldbuilderAgent(BaseAgent):
    async def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[StreamData, None]:
        effective_context = context if context is not None else {}
        self.logger.info(f"'{self.agent_name}' received task: {task_description}")

        book_state_slice = effective_context.get("book_writing_state_slice", {})
        project_name = book_state_slice.get("project_name", "Unknown Project")
        existing_world_anvil_data = book_state_slice.get("world_building_notes", {})
        if isinstance(existing_world_anvil_data, WorldAnvilSchema):
            existing_world_anvil_dict = existing_world_anvil_data.dict(exclude_none=True)
        elif isinstance(existing_world_anvil_data, dict):
            existing_world_anvil_dict = existing_world_anvil_data
        else:
            existing_world_anvil_dict = {}
        
        example_json_structure = WorldAnvilSchema.schema_json(indent=2)

        prompt = f"""
You are the Worldbuilder Agent for the book project '{project_name}'. 
Your task is to generate or update detailed aspects of the fictional world based on the provided instructions.
Task: {task_description}

Existing World Anvil / World Building Notes (if any, to maintain consistency and build upon):
{json.dumps(existing_world_anvil_dict, indent=2) if existing_world_anvil_dict else "No existing world building notes provided. You might be creating initial details or adding new aspects."}

Please generate the requested world-building information.
Your response MUST be a single, valid JSON object that conforms to the following Pydantic schema for the ENTIRE world anvil (not just the changed part). 
If you are updating a part of the world anvil, include all other existing parts as well in your response, modified only where specified by the task.
Schema:
{example_json_structure}

Ensure your entire output is ONLY the JSON object. Do not include any explanatory text before or after it.
New/Updated World Anvil (JSON Object):
"""
        self.logger.debug(f"WorldbuilderAgent LLM Prompt for '{project_name}':\n{prompt}")
        yield StreamData(type="info", content="Generating world-building content...")

        llm_response_str = ""
        try:
            llm_response_str = await self.llm.chat_completion_async(
                messages=[{"role": "user", "content": prompt}],
                json_mode=True 
            )
            self.logger.debug(f"WorldbuilderAgent LLM Raw Response for '{project_name}': {llm_response_str}")

            parsed_llm_json = json.loads(llm_response_str)
            validated_world_anvil = WorldAnvilSchema(**parsed_llm_json)
            generated_world_anvil_dict = validated_world_anvil.dict()

        except json.JSONDecodeError as e:
            self.logger.error(f"'{self.agent_name}' JSONDecodeError for '{project_name}': {e}. Response: {llm_response_str}")
            yield StreamData(type="error", content=f"Failed to parse LLM JSON response: {e}")
            return

        except Exception as e: # Includes Pydantic ValidationError
            self.logger.error(f"'{self.agent_name}' Error processing LLM response for '{project_name}': {e}. Response: {llm_response_str}", exc_info=True)
            yield StreamData(type="error", content=f"Failed to validate or process LLM response: {e}")
            return

        output_for_orchestrator = {"world_building_notes": generated_world_anvil_dict}
        
        self.logger.info(f"'{self.agent_name}' completed task for '{project_name}'. Outputting updated World Anvil.")
        yield StreamData(type="tool_response", content=json.dumps(output_for_orchestrator, default=str))

    def get_description(self) -> str:
        return "Generates and updates detailed descriptions of fictional worlds, including locations, cultures, history, magic systems, and factions, based on provided tasks and existing world context. Expects to receive and return a complete WorldAnvilSchema object."
