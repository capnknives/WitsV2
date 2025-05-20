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
    # __init__ is inherited

    def _extract_json_from_complex_response(self, response: Any, session_id: str) -> Dict[str, Any]:
        """
        Extract JSON from potentially complex response formats.
        
        This handles:
        - Dictionary responses with text in the 'response', 'content', or 'message.content' fields
        - Plain strings that might contain JSON
        - JSON strings that might have markdown code blocks or other formatting
        
        Args:
            response: The response from the LLM, could be dict, string, or complex object
            session_id: Session identifier for logging
            
        Returns:
            Extracted dictionary, or empty dict if extraction fails
        """
        extracted_text = ""
        
        # Handle various response formats
        if isinstance(response, dict):
            # Try common keys where content might be
            for key in ['response', 'content', 'message', 'text', 'result']:
                if key in response and response[key]:
                    if key == 'message' and isinstance(response[key], dict) and 'content' in response[key]:
                        extracted_text = response[key]['content']
                        break
                    elif isinstance(response[key], str):
                        extracted_text = response[key]
                        break
        elif hasattr(response, 'message') and hasattr(response.message, 'content'):
            # OpenAI-like response object
            extracted_text = response.message.content
        elif isinstance(response, str):
            # Direct string response
            extracted_text = response
            
        if not extracted_text:
            self.logger.warning(f"Could not extract text content from LLM response: {str(response)[:100]}...")
            return {}
            
        # Look for JSON in the extracted text
        json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', extracted_text)
        if json_match:
            # JSON within code blocks
            json_str = json_match.group(1)
            self.logger.debug(f"Extracted JSON from code block: {json_str[:100]}...")
        else:
            # Try to find JSON between curly braces if not in code blocks
            json_match = re.search(r'({[\s\S]*})', extracted_text)
            if json_match:
                json_str = json_match.group(1)
                self.logger.debug(f"Extracted JSON between curly braces: {json_str[:100]}...")
            else:
                # Use the whole text as a fallback
                json_str = extracted_text
                self.logger.debug(f"Using full response as JSON input: {json_str[:100]}...")
                
        try:
            return safe_json_loads(json_str, session_id)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse extracted JSON from LLM response in session {session_id}")
            return {}

    async def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[StreamData, None]:
        self.logger.info(f"'{self.agent_name}' received task: {task_description}")
        if context is None:
            context = {}
        
        session_id = context.get("session_id", f"plotter_{str(uuid.uuid4())[:8]}")


        book_state_slice = context.get("book_writing_state_slice", {})
        project_name = book_state_slice.get("project_name", "Unknown Project")
        current_overall_plot_summary = book_state_slice.get("overall_plot_summary", "")
        # Ensure current_detailed_chapter_outlines is a list of dicts, not Pydantic models, if coming from state
        raw_current_chapters = book_state_slice.get("detailed_chapter_outlines", [])
        current_detailed_chapter_outlines = [dict(ch) if not isinstance(ch, dict) else ch for ch in raw_current_chapters]

        genre = context.get("genre", book_state_slice.get("genre", "fantasy")) 
        themes = context.get("themes", book_state_slice.get("themes", []))

        self.logger.info(f"PlotterAgent for project '{project_name}' - Task: {task_description[:100]}...")
        self.logger.debug(f"Received book_state_slice (summary): overall_plot_summary exists: {bool(current_overall_plot_summary)}, num_chapters: {len(current_detailed_chapter_outlines)}")

        new_overall_plot_summary = current_overall_plot_summary
        generated_chapter_outlines: List[Dict[str, Any]] = [] 

        try:
            if "overall plot" in task_description.lower() or not current_overall_plot_summary:
                plot_summary_prompt = f"""
As a master storyteller for the project '{project_name}', refine or generate an overall plot summary.
Genre: {genre}
Themes: {', '.join(themes) if themes else 'Not specified'}
User Request: {task_description}
Existing Plot Summary (if any, refine this): {current_overall_plot_summary if current_overall_plot_summary else 'None'}

Return a concise and compelling overall plot summary for the novel.
Respond with a JSON object containing a single key "overall_plot_summary" with the text of the summary.
Example: {{"overall_plot_summary": "A young hero discovers a hidden power and must save the kingdom..."}}
"""
                self.logger.debug(f"PlotterAgent LLM Prompt (Overall Plot Summary):\n{plot_summary_prompt}")
                yield StreamData(type="info", content="Generating overall plot summary...")
                
                response_content_summary = "" 
                try:
                    chat_response_summary = await self.llm.chat_completion_async(
                        messages=[{"role": "user", "content": plot_summary_prompt}],
                    )
                    
                    # Extract the text content from the response
                    if hasattr(chat_response_summary, 'message') and hasattr(chat_response_summary.message, 'content'):
                        response_content_summary = chat_response_summary.message.content
                    else:
                        response_content_summary = str(chat_response_summary)
                    
                    self.logger.debug(f"PlotterAgent LLM Response (Overall Plot Summary):\n{response_content_summary[:300]}")
                    
                    # Use enhanced JSON extraction
                    summary_data = self._extract_json_from_complex_response(chat_response_summary, session_id)
                    
                    if summary_data and "overall_plot_summary" in summary_data:
                        new_overall_plot_summary = summary_data["overall_plot_summary"]
                        if new_overall_plot_summary != current_overall_plot_summary:
                            yield StreamData(type="tool_response", content=f"Generated new plot summary: {new_overall_plot_summary[:100]}...")
                    else:
                        # Try to extract from the raw text if JSON parsing failed
                        # Look for anything that resembles a plot summary
                        plot_match = re.search(r'"overall_plot_summary"\s*:\s*"([^"]+)"', response_content_summary)
                        if plot_match:
                            new_overall_plot_summary = plot_match.group(1)
                            yield StreamData(type="tool_response", content=f"Extracted plot summary: {new_overall_plot_summary[:100]}...")
                        else:
                            self.logger.warning(f"Failed to extract plot summary from LLM response. Response: {response_content_summary[:200]}")
                            yield StreamData(type="warning", content="Could not extract plot summary from LLM response.")
                
                except json.JSONDecodeError as e: 
                    self.logger.error(f"Failed to parse overall plot summary JSON from LLM: {e}. Response was: {response_content_summary[:300]}")
                    yield StreamData(type="error", content="Failed to parse plot summary response from LLM (JSONDecodeError).")
                except Exception as e:
                    self.logger.error(f"Error during overall plot summary generation: {e}", exc_info=True)
                    yield StreamData(type="error", content=f"Error generating plot summary: {str(e)}")

            generate_chapters_task = "chapter" in task_description.lower() or \
                                   (new_overall_plot_summary != current_overall_plot_summary and bool(new_overall_plot_summary))

            if generate_chapters_task:
                existing_chapters_summary_for_prompt = [
                    f"Ch{ch.get('chapter_number', 'N/A')}: {ch.get('title', 'Untitled')} - Status: {ch.get('status', 'Planned')} - Summary: {ch.get('summary', 'N/A')[:50]}..." 
                    for ch in current_detailed_chapter_outlines # Use the dict list
                ]

                chapter_prompt = f"""
As a master plot developer for the project '{project_name}', your task is: {task_description}
Current Overall Plot Summary: {new_overall_plot_summary if new_overall_plot_summary else 'Not yet defined.'}
Existing Chapter Outlines (titles, status, brief summary):
{json.dumps(existing_chapters_summary_for_prompt, indent=2) if existing_chapters_summary_for_prompt else 'No detailed chapter outlines yet.'}

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
If the task is to update a specific chapter, provide the full updated outline for that chapter.
If the task is to create new chapters, provide outlines for those new chapters.
If the task implies modifying existing chapters (e.g. "add more detail to chapter 2"), return the complete modified outline for chapter 2.
If the task is to generate a full list of chapters from scratch, provide all of them.
Example: {{"detailed_chapter_outlines": [{{"chapter_number": 1, "title": "The Beginning", "summary": "...", "status": "Outlined", "key_scenes": ["Scene A..."]}}]}}
"""
                self.logger.debug(f"PlotterAgent LLM Prompt (Chapter Outlines):\n{chapter_prompt}")
                yield StreamData(type="info", content="Generating chapter outlines...")
                
                response_content_chapters = "" 
                try:
                    chat_response_chapters = await self.llm.chat_completion_async(
                        messages=[{"role": "user", "content": chapter_prompt}],
                    )
                    
                    # Extract the text content from the response
                    if hasattr(chat_response_chapters, 'message') and hasattr(chat_response_chapters.message, 'content'):
                        response_content_chapters = chat_response_chapters.message.content
                    else:
                        response_content_chapters = str(chat_response_chapters)
                        
                    self.logger.debug(f"PlotterAgent LLM Response (Chapter Outlines):\n{response_content_chapters[:300]}")

                    # Use enhanced JSON extraction
                    chapter_outlines_data = self._extract_json_from_complex_response(chat_response_chapters, session_id)
                    if chapter_outlines_data and "detailed_chapter_outlines" in chapter_outlines_data:
                        parsed_outlines = chapter_outlines_data["detailed_chapter_outlines"]
                        if isinstance(parsed_outlines, list):
                            validated_outlines = []
                            for i, outline_dict in enumerate(parsed_outlines):
                                try:
                                    # Ensure outline_dict is a dict before validation
                                    if not isinstance(outline_dict, dict):
                                        self.logger.warning(f"Chapter outline item {i+1} is not a dict: {type(outline_dict)}. Skipping.")
                                        yield StreamData(type="warning", content=f"Invalid chapter outline data format for an item.")
                                        continue
                                    ChapterOutlineSchema.model_validate(outline_dict) 
                                    validated_outlines.append(outline_dict)
                                except Exception as val_err: 
                                    self.logger.warning(f"Validation failed for a chapter outline ({i+1}): {val_err}. Outline: {outline_dict}")
                                    yield StreamData(type="warning", content=f"Invalid chapter outline data for chapter {outline_dict.get('chapter_number', 'unknown')}: {val_err}")
                            generated_chapter_outlines = validated_outlines 
                            yield StreamData(type="tool_response", content=f"Generated/updated {len(generated_chapter_outlines)} chapter outlines.")
                        else:
                            self.logger.error(f"'detailed_chapter_outlines' is not a list in LLM response. Got: {type(parsed_outlines)}")
                            yield StreamData(type="error", content="LLM returned chapter outlines in an unexpected format.")
                    else:
                        self.logger.warning(f"Failed to get 'detailed_chapter_outlines' key from LLM JSON response for chapters. Response: {response_content_chapters[:200]}")
                        yield StreamData(type="warning", content="Could not extract chapter outlines from LLM response.")

                except json.JSONDecodeError as e: 
                    self.logger.error(f"Failed to parse chapter outlines JSON from LLM: {e}. Response was: {response_content_chapters}")
                    yield StreamData(type="error", content="Failed to parse chapter outlines response from LLM (JSONDecodeError).")
                except Exception as e: 
                    self.logger.error(f"Error processing chapter outlines: {str(e)}", exc_info=True)
                    yield StreamData(type="error", content=f"Error processing chapter outlines: {str(e)}")
            else:
                self.logger.info("Skipping chapter outline generation as task does not seem to require it or plot summary hasn't changed significantly.")
                # If not generating, the final list will be based on current_detailed_chapter_outlines
                # No, generated_chapter_outlines should remain empty if not generated. Merge logic will handle it.


            # Merge generated outlines with existing ones
            final_chapter_outlines = list(current_detailed_chapter_outlines) # Start with a copy of existing (as dicts)
            if generated_chapter_outlines: # Only merge if new ones were successfully generated and validated
                temp_final_outlines_map = {ch.get('chapter_number'): ch for ch in final_chapter_outlines if ch.get('chapter_number') is not None}
                
                for new_ch_data in generated_chapter_outlines:
                    ch_num = new_ch_data.get('chapter_number')
                    if ch_num is not None:
                        temp_final_outlines_map[ch_num] = new_ch_data # Add or overwrite
                    else:
                        final_chapter_outlines.append(new_ch_data)


                processed_ch_numbers = set()
                merged_outlines = []
                # Add updated/new from map
                # Ensure keys are sortable (e.g. filter out None if chapter_number could be None and used as key)
                sortable_keys = [k for k in temp_final_outlines_map.keys() if k is not None]
                for ch_num in sorted(list(sortable_keys)): # Explicitly convert keys to list and sort
                    merged_outlines.append(temp_final_outlines_map[ch_num])
                    processed_ch_numbers.add(ch_num)
                
                for ch in final_chapter_outlines: 
                    ch_num = ch.get('chapter_number')
                    if ch_num is None or ch_num not in processed_ch_numbers:
                        is_already_added = any(new_ch is ch for new_ch in generated_chapter_outlines if new_ch.get('chapter_number') is None)
                        if not is_already_added or ch not in generated_chapter_outlines :
                            if not any(item is ch for item in merged_outlines):
                                merged_outlines.append(ch)
                final_chapter_outlines = merged_outlines


            updated_book_state_slice = {
                "project_name": project_name,
                "overall_plot_summary": new_overall_plot_summary,
                "detailed_chapter_outlines": final_chapter_outlines, 
                "genre": genre, 
                "themes": themes, 
            }
            for key, value in book_state_slice.items():
                if key not in updated_book_state_slice: # Persist other unmanaged fields
                    updated_book_state_slice[key] = value

            yield StreamData(type="update_state", content={"book_writing_state_slice": updated_book_state_slice})
            self.logger.info(f"PlotterAgent finished. Overall plot summary length: {len(new_overall_plot_summary)}. Chapter outlines: {len(final_chapter_outlines)}")

        except Exception as e:
            self.logger.error(f"Critical error in PlotterAgent run for project '{project_name}': {str(e)}", exc_info=True)
            yield StreamData(type="error", content=f"Critical error in PlotterAgent: {str(e)}")
