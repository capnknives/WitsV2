import json
from typing import Any, Dict, Optional, AsyncGenerator

from agents.base_agent import BaseAgent
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager
from core.schemas import MemorySegment, MemorySegmentContent, StreamData

class ProseGenerationAgent(BaseAgent):
    def __init__(self, agent_name: str, config: Dict[str, Any], llm_interface: LLMInterface, memory_manager: MemoryManager, tool_registry: Optional[Any] = None):
        super().__init__(agent_name, config, llm_interface, memory_manager, tool_registry)
        self.llm_model_name = self.config_full.models.get('prose_generator', self.config_full.models.default)
        self.logger.info(f"'{self.agent_name}' initialized with LLM model: {self.llm_model_name}.")

    async def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[StreamData, None]:
        effective_context = context if context is not None else {}
        session_id = effective_context.get("session_id", "unknown_session")
        user_goal_summary = effective_context.get("user_goal_summary", "unknown_goal")
        # Extract chapter/scene identifier from task_description if possible, e.g., "Write Chapter 1" -> "chapter_1"
        # This is a simple extraction, might need more robust parsing for complex tasks.
        task_parts = task_description.lower().split()
        prose_identifier = "unknown_prose_chunk"
        if "chapter" in task_parts:
            try:
                idx = task_parts.index("chapter")
                if idx + 1 < len(task_parts):
                    prose_identifier = f"chapter_{task_parts[idx+1].replace('.','').replace(',','')}"
            except ValueError:
                pass # "chapter" not found
        elif "scene" in task_parts:
            try:
                idx = task_parts.index("scene")
                if idx + 1 < len(task_parts):
                    prose_identifier = f"scene_{task_parts[idx+1].replace('.','').replace(',','')}"
            except ValueError:
                pass # "scene" not found

        self.logger.info(f"'{self.agent_name}' (session: {session_id}) received task: {task_description} (Prose ID: {prose_identifier})")
        yield StreamData(type="info", content=f"'{self.agent_name}' starting task: {task_description[:100]}... for {prose_identifier}")

        book_state = effective_context.get("book_writing_state", {})
        plot_outline = book_state.get("plot_outline", "Not available.")
        chapter_outlines = book_state.get("chapter_outlines", {})
        scene_descriptions = book_state.get("scene_descriptions", {})
        character_profiles = book_state.get("character_profiles", {})
        world_details = book_state.get("world_details", {})

        # Try to get specific context for the chapter/scene if available
        relevant_chapter_outline = chapter_outlines.get(prose_identifier, "Outline for this specific chapter/scene not found.")
        relevant_scene_description = scene_descriptions.get(prose_identifier, "Scene description for this specific part not found.")

        prompt = f"""
You are the Prose Generation Agent. Your task is to write compelling narrative prose (story text, dialogue, descriptions) for a specific part of a book.

Task: {task_description}

Overall Plot Outline:
{json.dumps(plot_outline, indent=2) if isinstance(plot_outline, dict) else plot_outline}

Relevant Chapter/Scene Outline (for '{prose_identifier}'):
{json.dumps(relevant_chapter_outline, indent=2) if isinstance(relevant_chapter_outline, dict) else relevant_chapter_outline}

Relevant Scene Description (for '{prose_identifier}'):
{json.dumps(relevant_scene_description, indent=2) if isinstance(relevant_scene_description, dict) else relevant_scene_description}

Character Profiles:
{json.dumps(character_profiles, indent=2) if character_profiles else "No character profiles provided."}

World Details:
{json.dumps(world_details, indent=2) if world_details else "No world details provided."}

Based on all the provided context, please write the prose for the specified task. 
Focus on vivid descriptions, engaging dialogue, and advancing the plot as per the outlines.
Ensure the tone is consistent with the genre and existing material (if any).

Generated Prose:
"""
        yield StreamData(type="llm_prompt", content=prompt)
        self.logger.debug(f"ProseGenerationAgent LLM Prompt (session {session_id}, id: {prose_identifier}):\n{prompt}")
        llm_response_str = ""
        try:
            generated_prose_text = await self.llm.chat_completion_async(
                model_name=self.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config_full.agent_temperature,
                max_tokens=self.config_full.agent_max_tokens, # Adjust as needed for prose length
                # No json_mode=True here, as we expect plain text prose
            )
            yield StreamData(type="llm_response", content=generated_prose_text)
            self.logger.debug(f"ProseGenerationAgent LLM Raw Response (session {session_id}, id: {prose_identifier}): {generated_prose_text[:300]}...")

            if not generated_prose_text or not generated_prose_text.strip():
                raise ValueError("LLM returned empty or whitespace-only prose.")

        except Exception as e:
            self.logger.error(f"'{self.agent_name}' (session: {session_id}, id: {prose_identifier}) Error during LLM call or processing: {e}")
            yield StreamData(type="error", content=f"Error generating prose for {prose_identifier}: {e}", error_details=str(e))
            yield StreamData(type="tool_response", content={"message": f"Failed to generate prose for {prose_identifier}: {e}", "update_book_state": {}})
            return

        # Save the generated prose to memory
        memory_segment_content = MemorySegmentContent(
            text=generated_prose_text,
            tool_name=self.agent_name,
            tool_args={"task_description": task_description, "prose_identifier": prose_identifier},
            tool_output=f"Generated prose for {prose_identifier}. Length: {len(generated_prose_text)} chars."
        )
        
        prose_segment = MemorySegment(
            type=f"PROSE_CONTENT", # General type, specific ID in metadata
            source=self.agent_name,
            content=memory_segment_content,
            importance=0.8, # Prose is highly important
            metadata={
                "session_id": session_id,
                "user_goal_summary": user_goal_summary,
                "prose_identifier": prose_identifier, # e.g., "chapter_1", "scene_5b"
                "agent_name": self.agent_name,
                "task": task_description[:100]
            }
        )
        await self.memory.add_memory_segment(prose_segment)
        self.logger.info(f"'{self.agent_name}' (session: {session_id}) saved generated prose for '{prose_identifier}' to memory. Segment ID: {prose_segment.id}")
        yield StreamData(type="memory_add", content=f"Saved prose for '{prose_identifier}' (segment {prose_segment.id}) to memory.")

        # Prepare response for Orchestrator
        # Update the 'chapters' or a similar structure in book_state.
        # The key will be the prose_identifier.
        update_for_orchestrator = {
            "generated_prose": {
                prose_identifier: generated_prose_text
            }
        }

        final_response_payload = {
            "message": f"'{self.agent_name}' successfully generated and saved prose for '{prose_identifier}'.",
            "update_book_state": update_for_orchestrator 
        }
        
        self.logger.info(f"'{self.agent_name}' (session: {session_id}) completed task for '{prose_identifier}'. Output for orchestrator: {json.dumps(final_response_payload)[:200]}...")
        yield StreamData(type="tool_response", content=final_response_payload)

    def get_description(self) -> str:
        return "Generates narrative prose (story text, dialogue, descriptions) for specific chapters or scenes based on outlines, character profiles, and world details."

    def get_config_key(self) -> str:
        return "book_prose_generator" # Matches the key in config.yaml
