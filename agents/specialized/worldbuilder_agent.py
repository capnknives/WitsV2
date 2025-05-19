import json
from typing import Any, Dict, Optional, AsyncGenerator

from agents.base_agent import BaseAgent
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager
from core.schemas import MemorySegment, MemorySegmentContent, StreamData

class WorldbuilderAgent(BaseAgent):
    def __init__(self, agent_name: str, config: Dict[str, Any], llm_interface: LLMInterface, memory_manager: MemoryManager, tool_registry: Optional[Any] = None):
        super().__init__(agent_name, config, llm_interface, memory_manager, tool_registry)
        self.llm_model_name = self.config_full.models.get('worldbuilder', self.config_full.models.default)
        self.logger.info(f"'{self.agent_name}' initialized with LLM model: {self.llm_model_name}.")

    async def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[StreamData, None]:
        effective_context = context if context is not None else {}
        session_id = effective_context.get("session_id", "unknown_session")
        user_goal_summary = effective_context.get("user_goal_summary", "unknown_goal") # For memory key consistency
        self.logger.info(f"'{self.agent_name}' (session: {session_id}) received task: {task_description}")
        yield StreamData(type="info", content=f"'{self.agent_name}' starting task: {task_description[:100]}...")

        book_state = effective_context.get("book_writing_state", {})
        existing_world_details = book_state.get("world_details", {})
        
        prompt = f"""
You are the Worldbuilder Agent. Your task is to generate detailed aspects of a fictional world based on the provided instructions.
Task: {task_description}

Existing World Details (if any, to maintain consistency):
{json.dumps(existing_world_details, indent=2) if existing_world_details else "No existing world details provided. You are creating the initial details or adding to them."}

Please generate the requested world-building information.
Structure your response as a JSON object containing the new or updated world details.
For example, if asked to describe a city, you might return:
{{
  "city_elysia": {{
    "description": "A shimmering city built on floating islands...",
    "districts": [
      {{"name": "Skyport", "details": "Handles all aerial traffic."}},
      {{"name": "Crystal Gardens", "details": "Known for its bioluminescent flora."}}
    ],
    "governance": "Ruled by a council of mages."
  }}
}}
Ensure the keys in your JSON response are descriptive and avoid overwriting existing distinct elements unless the task explicitly asks for an update to a specific element.
If adding to existing details, try to integrate them smoothly.
New/Updated World Details (JSON):
"""
        yield StreamData(type="llm_prompt", content=prompt)
        self.logger.debug(f"WorldbuilderAgent LLM Prompt (session {session_id}):\n{prompt}")
        llm_response_str = ""
        try:
            llm_response_str = await self.llm.chat_completion_async(
                model_name=self.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config_full.agent_temperature, # Use agent_temperature
                max_tokens=self.config_full.agent_max_tokens, # Use agent_max_tokens
                json_mode=True # Request JSON output if supported by LLM
            )
            yield StreamData(type="llm_response", content=llm_response_str)
            self.logger.debug(f"WorldbuilderAgent LLM Raw Response (session {session_id}): {llm_response_str}")

            generated_details = json.loads(llm_response_str)
            if not isinstance(generated_details, dict):
                raise ValueError("LLM response was not a JSON object (dict).")

        except json.JSONDecodeError as e:
            self.logger.error(f"'{self.agent_name}' (session: {session_id}) JSONDecodeError: {e}. Response: {llm_response_str}")
            yield StreamData(type="error", content=f"Error decoding LLM response for world building: {e}", error_details=str(e))
            yield StreamData(type="tool_response", content={"message": "Failed to generate world details due to LLM response format error.", "update_book_state": {}})
            return
        except Exception as e:
            self.logger.error(f"'{self.agent_name}' (session: {session_id}) Error during LLM call or processing: {e}")
            yield StreamData(type="error", content=f"Error generating world details: {e}", error_details=str(e))
            yield StreamData(type="tool_response", content={"message": f"Failed to generate world details: {e}", "update_book_state": {}})
            return

        # Save the generated world details to memory
        memory_segment_content = MemorySegmentContent(
            text=json.dumps(generated_details),
            tool_name=self.agent_name,
            tool_args={"task_description": task_description, "existing_world_details_summary": f"{len(json.dumps(existing_world_details))} chars"},
            tool_output="Generated world details."
        )
        
        world_details_segment = MemorySegment(
            type="WORLD_DETAILS_CHUNK", 
            source=self.agent_name,
            content=memory_segment_content,
            importance=0.7,
            metadata={
                "session_id": session_id,
                "task": task_description[:100],
                "agent_name": self.agent_name,
                "user_goal_summary": user_goal_summary # For linking back if needed
            }
        )
        await self.memory.add_memory_segment(world_details_segment)
        self.logger.info(f"'{self.agent_name}' (session: {session_id}) saved generated world details to memory. Segment ID: {world_details_segment.id}")
        yield StreamData(type="memory_add", content=f"Saved world details (segment {world_details_segment.id}) to memory.")

        # Prepare response for Orchestrator
        # The orchestrator expects update_book_state to merge with its existing book_writing_state.
        # So, we should return the new details under the 'world_details' key,
        # and the orchestrator's logic should merge this into its main book_writing_state.world_details.
        
        # Create a new dictionary for the update to avoid modifying the original book_state directly here.
        # The orchestrator will handle the merging.
        update_for_orchestrator = {"world_details": generated_details}

        final_response_payload = {
            "message": f"'{self.agent_name}' successfully generated and saved world details based on task: {task_description[:100]}...",
            "update_book_state": update_for_orchestrator 
        }
        
        self.logger.info(f"'{self.agent_name}' (session: {session_id}) completed task. Output for orchestrator: {json.dumps(final_response_payload)}")
        yield StreamData(type="tool_response", content=final_response_payload)

    def get_description(self) -> str:
        return "Generates detailed descriptions of fictional worlds, including locations, cultures, history, magic systems, and factions, based on provided tasks and existing world context."

    def get_config_key(self) -> str:
        return "book_worldbuilder" # Matches the key in config.yaml
