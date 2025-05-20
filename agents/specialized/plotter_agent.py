from typing import Any, Dict, Optional, List, AsyncGenerator
import json
import logging
import re
from agents.base_agent import BaseAgent
from core.schemas import StreamData
from agents.book_writing_schemas import ChapterOutlineSchema # For typing and validation
from core.json_utils import safe_json_loads, balance_json_braces
import uuid

class PlotterAgent(BaseAgent):
    async def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[StreamData, None]:
        """Process book plotting tasks and generate/update plot summaries and chapter outlines."""
        if context is None:
            context = {}
        
        session_id = context.get("session_id", f"plotter_{str(uuid.uuid4())[:8]}")
        self.logger.info(f"'{self.agent_name}' received task: {task_description}")

        try:
            # Extract book_writing_state_slice and other relevant context
            book_state_slice = context.get("book_writing_state_slice", {})
            project_name = book_state_slice.get("project_name", "Unknown Project")
            existing_plot_summary = book_state_slice.get("overall_plot_summary", "")
            existing_chapters = book_state_slice.get("detailed_chapter_outlines", [])
            genre = context.get("genre", "fantasy") 
            themes = context.get("themes", []) 

            self.logger.info(f"PlotterAgent for project '{project_name}' - Task: {task_description[:100]}...")
            self.logger.debug(f"Received book_state_slice: {json.dumps(book_state_slice, default=str)[:200]}...")

            new_overall_plot_summary = existing_plot_summary
            if "overall plot" in task_description.lower() or not existing_plot_summary:
                plot_summary_prompt = f"""
As a master storyteller for the project '{project_name}', refine or generate an overall plot summary.
Genre: {genre}
Themes: {', '.join(themes) if themes else 'Not specified'}
User Request: {task_description}
Existing Plot Summary (if any, refine this): {existing_plot_summary if existing_plot_summary else 'None'}

Return a concise and compelling overall plot summary for the novel.
Respond with a JSON object containing a single key "overall_plot_summary" with the text of the summary.
Example: {{"overall_plot_summary": "A young hero discovers a hidden power and must save the kingdom..."}}
"""
                self.logger.debug(f"PlotterAgent LLM Prompt (Overall Plot Summary):\n{plot_summary_prompt}")
                yield StreamData(type="info", content="Generating overall plot summary...")
                
                chat_response = await self.llm.chat_completion_async(
                    messages=[{"role": "user", "content": plot_summary_prompt}],
                )
                response_content = chat_response.message.content if hasattr(chat_response, 'message') else str(chat_response)
                
                try:
                    summary_data = json.loads(response_content)
                    new_overall_plot_summary = summary_data.get("overall_plot_summary", existing_plot_summary)
                    if new_overall_plot_summary != existing_plot_summary:
                        yield StreamData(type="tool_response", content=f"Generated new plot summary: {new_overall_plot_summary[:100]}...")
                except json.JSONDecodeError:
                    self.logger.error("Failed to parse overall plot summary JSON from LLM. Using existing or empty.")
                    yield StreamData(type="error", content="Failed to parse plot summary response from LLM")

            existing_chapters_summary = [
                f"Ch{ch.get('chapter_number', 'N/A')}: {ch.get('title', 'Untitled')} - {ch.get('status', 'Planned')} - Summary: {ch.get('summary', 'N/A')[:50]}..." 
                for ch in existing_chapters
            ]

            chapter_prompt = f"""
As a master plot developer for the project '{project_name}', your task is: {task_description}
Current Overall Plot Summary: {new_overall_plot_summary if new_overall_plot_summary else 'Not yet defined.'}
Existing Chapter Outlines (titles, status, brief summary):
{json.dumps(existing_chapters_summary, indent=2) if existing_chapters_summary else 'No detailed chapter outlines yet.'}

Based on the task, provide detailed chapter outlines. 
Each chapter outline must be a JSON object with the following fields:
- "chapter_number": (integer) The number of the chapter.
- "title": (string) A working title for the chapter.
- "summary": (string) A detailed summary of events, character arcs, and plot points in this chapter.
- "status": (string, e.g., "To Outline", "Outlined", "Drafted", "Revised") The current status.
- "key_scenes": (optional, list of strings) Brief descriptions of key scenes within this chapter.
- "notes": (optional, string) Any additional notes for this chapter.

Return your response as a JSON object with a single key "detailed_chapter_outlines", 
which is a list of these chapter outline objects.
Example: {{"detailed_chapter_outlines": [{{"chapter_number": 1, "title": "The Beginning", "summary": "...", "status": "Outlined", "key_scenes": ["Scene A..."]}}]}}
"""
            self.logger.debug(f"PlotterAgent LLM Prompt (Detailed Chapters):\n{chapter_prompt}")
            yield StreamData(type="info", content="Generating chapter outlines...")
            
            chat_response = await self.llm.chat_completion_async(
                messages=[{"role": "user", "content": chapter_prompt}],
            )
            response_content = chat_response.message.content if hasattr(chat_response, 'message') else str(chat_response)
            
            generated_chapter_outlines = []
            try:
                chapter_outlines_data = json.loads(response_content)
                raw_outlines = chapter_outlines_data.get("detailed_chapter_outlines", [])
                for outline_data in raw_outlines:
                    try:
                        validated_outline = ChapterOutlineSchema(**outline_data).dict()
                        generated_chapter_outlines.append(validated_outline)
                        yield StreamData(type="tool_response", 
                                      content=f"Generated outline for chapter {outline_data.get('chapter_number')}: {outline_data.get('title')}")
                    except Exception as e: 
                        self.logger.warning(f"Skipping a chapter outline due to validation error: {e}. Data: {outline_data}")
                        yield StreamData(type="error", content=f"Error validating chapter outline: {str(e)}")

            except json.JSONDecodeError:
                self.logger.error("Failed to parse detailed chapter outlines JSON from LLM.")
                yield StreamData(type="error", content="Failed to parse chapter outlines response from LLM")

            output_for_orchestrator = {}
            if new_overall_plot_summary != existing_plot_summary:
                output_for_orchestrator["overall_plot_summary"] = new_overall_plot_summary
            
            if generated_chapter_outlines:
                output_for_orchestrator["detailed_chapter_outlines"] = generated_chapter_outlines
            
            if not output_for_orchestrator:
                self.logger.info(f"PlotterAgent made no changes to plot summary or chapter outlines for task: {task_description}")
                yield StreamData(type="tool_response", content="No changes to plot elements based on the task.")
            else:
                self.logger.info(f"'{self.agent_name}' completed task. Output: {json.dumps(output_for_orchestrator, default=str)[:200]}...")
                yield StreamData(type="tool_response", content=json.dumps(output_for_orchestrator))

        except Exception as e:
            self.logger.exception(f"Error in PlotterAgent run method for project '{project_name}': {e}")
            yield StreamData(type="error", content=f"PlotterAgent encountered an error: {str(e)}")
