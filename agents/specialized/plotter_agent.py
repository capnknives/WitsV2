from typing import Any, Dict, Optional, List
import json
import logging # Ensure logging is imported if not already via BaseAgent changes
from agents.base_agent import BaseAgent
from core.schemas import MemorySegment, MemorySegmentContent # Assuming these are your schemas

class PlotterAgent(BaseAgent):
    # __init__ is inherited from BaseAgent, no need to redefine if super() is called correctly
    # and BaseAgent now handles logger and tool_registry

    async def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> str:
        self.logger.info(f"'{self.agent_name}' received task: {task_description}")
        if context is None:
            context = {}

        session_id = context.get("session_id", "unknown_session")
        # book_writing_state = context.get("book_writing_state", {}) # Access if needed
        
        genre = context.get("genre", "fantasy") 
        themes = context.get("themes", []) 
        characters = context.get("characters", []) 
        initial_ideas = task_description 

        self.logger.info(f"PlotterAgent - Genre: {genre}, Themes: {themes}, Initial Ideas: {initial_ideas[:100]}...")

        try:
            # 1. Generate High-Level Plot Outline
            high_level_prompt = f"""
You are a master storyteller and plot architect.
Based on the following criteria, generate a compelling high-level plot outline for a {genre} novel.
Themes: {', '.join(themes) if themes else 'Not specified, be creative.'}
Initial Ideas/User Request: {initial_ideas}
Characters to consider (if any): {', '.join(characters) if characters else 'Feel free to introduce key archetypes.'}

The plot outline should cover the main acts (e.g., Act I, Act II, Act III) and key turning points.
Return the high-level plot outline as a JSON object with a key "high_level_outline" containing a list of strings or a structured object.
Example: {{"high_level_outline": ["Inciting incident...", "Rising action point 1...", "Climax...", "Resolution..."]}}
"""
            self.logger.debug(f"PlotterAgent LLM Prompt (High-Level):\n{high_level_prompt}")
            high_level_response_str = await self.llm.chat_completion_async(
                model_name=self.agent_config.get("model_name", self.config_full.models.default),
                messages=[{"role": "user", "content": high_level_prompt}],
                temperature=self.agent_config.get("temperature", 0.7)
            )
            self.logger.debug(f"PlotterAgent LLM Response (High-Level):\n{high_level_response_str}")
            
            try:
                high_level_plot_data = json.loads(high_level_response_str)
                high_level_outline = high_level_plot_data.get("high_level_outline", {"error": "No high-level outline in LLM JSON response", "raw_response": high_level_response_str})
            except json.JSONDecodeError:
                self.logger.error("Failed to parse high-level plot JSON from LLM.")
                high_level_outline = {"error": "Failed to parse high-level plot JSON", "raw_response": high_level_response_str}

            await self.memory.add_memory_segment(MemorySegment(
                type="PLOT_OUTLINE",
                source=self.agent_name,
                # Storing structured data as a JSON string in the 'text' field.
                # Adjust if MemorySegmentContent has a dedicated field like 'data' or 'tool_output' for structured content.
                content=MemorySegmentContent(text=json.dumps(high_level_outline), tool_output=f"High-level plot outline for task: {task_description[:50]}..."),
                metadata={"session_id": session_id, "task_description": task_description}
            ))
            self.logger.info("Saved high-level plot outline to memory.")

            # 2. Generate Chapter Outlines
            if isinstance(high_level_outline, dict) and "error" in high_level_outline:
                 chapter_outlines = {"error": "Skipping chapter outlines due to error in high-level plot generation."}
            else:
                chapter_prompt = f"""
Given the following high-level plot outline:
{json.dumps(high_level_outline, indent=2)}

Break this down into detailed chapter outlines. Each chapter outline should summarize the key events, character developments, and plot advancements for that chapter.
Return the chapter outlines as a JSON object with a key "chapter_outlines" which is a list of objects, each object having "chapter_number" and "summary".
Example: {{"chapter_outlines": [{{"chapter_number": 1, "summary": "Introduction of protagonist and the initial problem."}}, ...]}}
"""
                self.logger.debug(f"PlotterAgent LLM Prompt (Chapters):\n{chapter_prompt}")
                chapter_response_str = await self.llm.chat_completion_async(
                    model_name=self.agent_config.get("model_name", self.config_full.models.default),
                    messages=[{"role": "user", "content": chapter_prompt}],
                    temperature=self.agent_config.get("temperature", 0.7)
                )
                self.logger.debug(f"PlotterAgent LLM Response (Chapters):\n{chapter_response_str}")
                try:
                    chapter_outlines_data = json.loads(chapter_response_str)
                    chapter_outlines = chapter_outlines_data.get("chapter_outlines", {"error": "No chapter outlines in LLM JSON response", "raw_response": chapter_response_str})
                except json.JSONDecodeError:
                    self.logger.error("Failed to parse chapter outlines JSON from LLM.")
                    chapter_outlines = {"error": "Failed to parse chapter outlines JSON", "raw_response": chapter_response_str}

                await self.memory.add_memory_segment(MemorySegment(
                    type="CHAPTER_OUTLINES",
                    source=self.agent_name,
                    content=MemorySegmentContent(text=json.dumps(chapter_outlines), tool_output=f"Chapter outlines for task: {task_description[:50]}..."),
                    metadata={"session_id": session_id, "task_description": task_description}
                ))
                self.logger.info("Saved chapter outlines to memory.")

            # 3. Generate Scene Descriptions (Simplified for first chapter)
            scene_descriptions = {} # Initialize
            if isinstance(chapter_outlines, list) and chapter_outlines and isinstance(chapter_outlines[0], dict):
                first_chapter_summary = chapter_outlines[0].get("summary", "No summary for first chapter.")
                scene_prompt = f"""
Based on the summary for the first chapter: "{first_chapter_summary}"
Generate a few key scene descriptions for this chapter. Each scene should include a brief description of the setting, characters involved, and the main action or dialogue.
Return as a JSON object with a key "scene_descriptions" which is a list of objects, each with "scene_number" and "description".
Example: {{"scene_descriptions": [{{"scene_number": 1, "description": "Character A meets Character B in a dark alley..."}}, ...]}}
"""
                self.logger.debug(f"PlotterAgent LLM Prompt (Scenes):\n{scene_prompt}")
                scene_response_str = await self.llm.chat_completion_async(
                    model_name=self.agent_config.get("model_name", self.config_full.models.default),
                    messages=[{"role": "user", "content": scene_prompt}],
                    temperature=self.agent_config.get("temperature", 0.7)
                )
                self.logger.debug(f"PlotterAgent LLM Response (Scenes):\n{scene_response_str}")
                try:
                    scene_data = json.loads(scene_response_str)
                    scene_descriptions = scene_data.get("scene_descriptions", {"error": "No scene descriptions in LLM JSON response", "raw_response": scene_response_str})
                except json.JSONDecodeError:
                    self.logger.error("Failed to parse scene descriptions JSON from LLM.")
                    scene_descriptions = {"error": "Failed to parse scene descriptions JSON", "raw_response": scene_response_str}
                
                await self.memory.add_memory_segment(MemorySegment(
                    type="SCENE_DESCRIPTIONS",
                    source=self.agent_name,
                    content=MemorySegmentContent(text=json.dumps(scene_descriptions), tool_output=f"Scene descriptions for first chapter, task: {task_description[:50]}..."),
                    metadata={"session_id": session_id, "task_description": task_description, "chapter_number": 1}
                ))
                self.logger.info("Saved scene descriptions for the first chapter to memory.")
            elif isinstance(chapter_outlines, dict) and "error" in chapter_outlines:
                scene_descriptions = {"error": "Skipping scene descriptions due to error in chapter outlines generation."}
            else:
                scene_descriptions = {"info": "No chapter outlines available or in expected format to generate scenes."}

            processed_output = {
                "message": "PlotterAgent completed. Generated high-level outline, chapter outlines, and initial scene descriptions.",
                "update_book_state": { 
                    "plot_outline": high_level_outline,
                    "chapter_outlines": chapter_outlines,
                    "scene_descriptions_chapter_1": scene_descriptions
                }
            }
            self.logger.info(f"'{self.agent_name}' completed task. Output: {json.dumps(processed_output)[:200]}...")
            return json.dumps(processed_output)

        except Exception as e:
            self.logger.exception(f"Error in PlotterAgent run method: {e}")
            error_output = {"error": str(e), "message": "PlotterAgent encountered an error."}
            return json.dumps(error_output)
