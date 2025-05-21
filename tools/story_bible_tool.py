import json
import logging
from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel, Field

from core.memory_manager import MemoryManager
from core.schemas import MemorySegment, MemorySegmentContent
from tools.base_tool import BaseTool, ToolResponse

# --- Pydantic Models for Args and Response ---

class StoryBibleToolArgsInput(BaseModel):
    action: Literal["add", "get", "get_all"] = Field(..., description="The action to perform on the story bible: 'add' an entry, 'get' an entry, or 'get_all' entries.")
    path: Optional[str] = Field(None, description="Dot-separated path for 'add' or 'get' actions (e.g., 'characters.hero.name', 'world.locations.capital_city'). Required for 'add' and 'get'.")
    content: Optional[Dict[str, Any]] = Field(None, description="The JSON content (as a dictionary) to add or update. Required for 'add' action.")
    # session_id and user_goal_summary will be passed in context by the agent calling the tool

class StoryBibleToolResponseOutput(BaseModel):
    status: str = Field(..., description="Status of the operation (e.g., 'success', 'error').")
    message: str = Field(..., description="A message describing the outcome of the operation.")
    data: Optional[Dict[str, Any]] = Field(None, description="The retrieved data for 'get' or 'get_all' actions, or the added/updated content for 'add'.")
    path: Optional[str] = Field(None, description="The path operated on, if applicable.")


class StoryBibleTool(BaseTool):
    name: str = "StoryBibleTool"
    description: str = (
        "Manages a persistent, structured story bible. "
        "Allows adding, updating, and retrieving entries like plot points, character profiles, "
        "world-building details, etc. Entries are organized hierarchically using dot-separated paths."
    )
    args_schema = StoryBibleToolArgsInput

    def __init__(self, memory_manager: MemoryManager, llm_interface: Optional[Any] = None, tool_registry: Optional[Any] = None):
        super().__init__()
        self.memory_manager = memory_manager  # Explicitly set memory_manager as instance attribute
        self.llm_interface = llm_interface  # Store llm_interface as instance attribute if needed
        self.tool_registry = tool_registry  # Store tool_registry as an instance attribute
        self.logger = logging.getLogger(__name__)

    def _get_story_bible_memory_key(self, session_id: str, user_goal_summary: str) -> str:
        return f"story_bible_data_{session_id}_{user_goal_summary[:50].replace(' ', '_')}"

    async def _load_story_bible(self, memory_key: str) -> Dict[str, Any]:
        """Loads the story bible from the memory manager."""
        try:
            # Fallback to search_memory as retrieve_memory_segment_by_key is not standard
            self.logger.warning(f"StoryBibleTool: Attempting search_memory for key '{memory_key}' as direct key retrieval is not standard on MemoryManager.")
            
            # Construct a query that is likely to match the metadata key
            query_for_key = f"memory_segment_key_is_{memory_key}" # Make query more specific
            
            # Assuming search_memory might exist and take these general parameters
            if hasattr(self.memory_manager, "search_memory") and callable(getattr(self.memory_manager, "search_memory")):
                results = await self.memory_manager.search_memory(
                    query=query_for_key, 
                    limit=1
                )
                segment = results[0] if results else None
            else:
                self.logger.error("StoryBibleTool: MemoryManager does not have a callable search_memory method.")
                return {}

            if segment and segment.content and segment.content.text:
                return json.loads(segment.content.text)
            self.logger.info(f"StoryBibleTool: No existing story bible found for key '{memory_key}' or content is not text.")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"StoryBibleTool: Failed to decode JSON from story bible memory (key: {memory_key}): {e}")
            return {}
        except Exception as e:
            self.logger.error(f"StoryBibleTool: Error loading story bible (key: {memory_key}): {e}")
            return {} # Return empty dict on error to allow fresh creation

    async def save_story_bible(self, bible_data: Dict[str, Any], session_id: str, memory_key: str):
        try:
            content_text = json.dumps(bible_data, indent=2)
            
            # Directly use add_segment from MemoryManager
            if hasattr(self.memory_manager, "add_segment") and callable(getattr(self.memory_manager, "add_segment")):
                await self.memory_manager.add_segment(
                    segment_type="STORY_BIBLE_DATA",
                    content_text=content_text,
                    source=self.name,
                    tool_name=self.name, # Passing tool_name if add_segment uses it
                    meta={ # Passing other metadata under 'meta' as per add_segment signature
                        "session_id": session_id,
                        "key": memory_key,
                    }
                )
                self.logger.info(f"StoryBibleTool: Successfully saved story bible to memory using add_segment (key: {memory_key}).")
            else:
                self.logger.error("StoryBibleTool: MemoryManager does not have a callable add_segment method.")
                raise NotImplementedError("MemoryManager does not support the required add_segment memory operation.")
            
        except Exception as e:
            self.logger.error(f"StoryBibleTool: Error saving story bible (key: {memory_key}): {e}")
            raise  # Re-raise the exception to handle it at a higher level

    def _get_value_at_path(self, data: Dict[str, Any], path: str) -> Optional[Any]:
        keys = path.split('.')
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def _set_value_at_path(self, data: Dict[str, Any], path: str, value: Any, overwrite: bool = True):
        keys = path.split('.')
        current = data
        for i, key in enumerate(keys[:-1]):
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        final_key = keys[-1]
        if not overwrite and final_key in current and isinstance(current[final_key], dict) and isinstance(value, dict):
            current[final_key].update(value) # Merge if both are dicts and overwrite is false
        else:
            current[final_key] = value # Overwrite or set new value
            
    async def execute(self, args: StoryBibleToolArgsInput, context: Optional[Dict[str, Any]] = None) -> ToolResponse[StoryBibleToolResponseOutput]:
        """
        Execute the story bible tool with the given arguments and context.
        
        Args:
            args: StoryBibleToolArgsInput containing the action, path, and content.
            context: Optional context dictionary containing session_id and user_goal_summary.
            
        Returns:
            ToolResponse with appropriate StoryBibleToolResponseOutput based on the action and result.
        """
        effective_context = context if context is not None else {}
        session_id = effective_context.get("session_id")
        user_goal_summary = effective_context.get("user_goal_summary")

        if not session_id or not user_goal_summary:
            self.logger.error("StoryBibleTool: session_id or user_goal_summary missing from context.")
            return ToolResponse(
                status_code=400,
                error_message="session_id and user_goal_summary are required in context to use StoryBibleTool.",
                output=StoryBibleToolResponseOutput(
                    status="error", 
                    message="session_id and user_goal_summary are required in context to use StoryBibleTool.",
                    data=None,
                    path=None
                )
            )

        memory_key = self._get_story_bible_memory_key(session_id, user_goal_summary)
        story_bible = await self._load_story_bible(memory_key)

        try:
            if args.action == "add":
                if not args.path or not args.content:
                    return ToolResponse(
                        status_code=400, 
                        error_message="'path' and 'content' are required for 'add' action.",
                        output=StoryBibleToolResponseOutput(
                            status="error", 
                            message="'path' and 'content' are required for 'add' action.",
                            data=None,
                            path=args.path
                        )
                    )
                
                # For 'add', we assume content is a dict
                self._set_value_at_path(story_bible, args.path, args.content, overwrite=True)
                await self.save_story_bible(story_bible, session_id, memory_key)
                response_data = args.content
                return ToolResponse(
                    status_code=200,
                    error_message=None,
                    output=StoryBibleToolResponseOutput(
                        status="success", 
                        message=f"Entry added/updated at '{args.path}'.", 
                        data=response_data, 
                        path=args.path
                    )
                )
            
            elif args.action == "get":
                if not args.path:
                    return ToolResponse(
                        status_code=400, 
                        error_message="'path' is required for 'get' action.",
                        output=StoryBibleToolResponseOutput(
                            status="error", 
                            message="'path' is required for 'get' action.",
                            data=None,
                            path=args.path
                        )
                    )
                
                retrieved_value = self._get_value_at_path(story_bible, args.path)
                if retrieved_value is not None:
                    # Ensure data is a dict for the response model
                    response_data = retrieved_value if isinstance(retrieved_value, dict) else {"value": retrieved_value}
                    return ToolResponse(
                        output=StoryBibleToolResponseOutput(
                            status="success", 
                            message=f"Entry retrieved from '{args.path}'.", 
                            data=response_data, 
                            path=args.path
                        )
                    )
                else:
                    return ToolResponse(
                        output=StoryBibleToolResponseOutput(
                            status="not_found", 
                            message=f"No entry found at '{args.path}'.", 
                            data=None, 
                            path=args.path
                        ),
                        status_code=404,
                        error_message=f"No entry found at '{args.path}'."
                    )
                
            elif args.action == "get_all":
                response_data = story_bible
                return ToolResponse(
                    output=StoryBibleToolResponseOutput(
                        status="success", 
                        message="All story bible entries retrieved.", 
                        data=response_data, 
                        path=None
                    )
                )
            else:
                # Should not happen due to Literal type in args
                return ToolResponse(
                    status_code=400, 
                    error_message=f"Unknown action: {args.action}",
                    output=StoryBibleToolResponseOutput(
                        status="error", 
                        message=f"Unknown action: {args.action}",
                        data=None,
                        path=args.path
                    )
                )
                
        except Exception as e:
            self.logger.exception(f"StoryBibleTool: Error during action '{args.action}' (path: {args.path}): {e}")
            return ToolResponse(
                status_code=500, 
                error_message=f"An unexpected error occurred: {str(e)}",
                output=StoryBibleToolResponseOutput(
                    status="error", 
                    message=f"An unexpected error occurred: {str(e)}", 
                    data=None, 
                    path=args.path
                )
            )
