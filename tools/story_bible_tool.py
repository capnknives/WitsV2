import json
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
    # LLMInterface is not directly used in this version but kept for consistency with BaseTool
    # MemoryManager is crucial and will be accessed via self.memory_manager

    def __init__(self, memory_manager: MemoryManager, llm_interface: Optional[Any] = None, tool_registry: Optional[Any] = None):
        super().__init__(llm_interface=llm_interface, memory_manager=memory_manager, tool_registry=tool_registry)
        # self.memory_manager is inherited from BaseTool and initialized there

    def _get_story_bible_memory_key(self, session_id: str, user_goal_summary: str) -> str:
        return f"story_bible_data_{session_id}_{user_goal_summary[:50].replace(' ', '_')}"

    async def _load_story_bible(self, memory_key: str) -> Dict[str, Any]:
        """Loads the story bible from the memory manager."""
        try:
            if hasattr(self.memory_manager, "retrieve_memory_segment_by_key"):
                segment = await self.memory_manager.retrieve_memory_segment_by_key(memory_key)
            else: # Fallback if the specific method isn't available
                self.logger.warning(f"StoryBibleTool: retrieve_memory_segment_by_key not found on MemoryManager. Attempting search_memory for key '{memory_key}'.")
                # This fallback is a placeholder and might need specific query logic for your MemoryManager's search_memory
                results = await self.memory_manager.search_memory(query_text=f"story_bible_data for key {memory_key}", limit=1, filter_metadata={"key": memory_key, "type": "STORY_BIBLE_DATA"})
                segment = results[0] if results else None

            if segment and segment.content and segment.content.text:
                return json.loads(segment.content.text)
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"StoryBibleTool: Failed to decode JSON from story bible memory (key: {memory_key}): {e}")
            return {}
        except Exception as e:
            self.logger.error(f"StoryBibleTool: Error loading story bible (key: {memory_key}): {e}")
            return {} # Return empty dict on error to allow fresh creation

    async def _save_story_bible(self, memory_key: str, bible_data: Dict[str, Any], session_id: str, user_goal_summary: str):
        """Saves the story bible to the memory manager."""
        try:
            content_obj = MemorySegmentContent(
                text=json.dumps(bible_data, indent=2),
                tool_output="Story bible data updated."
            )
            segment = MemorySegment(
                type="STORY_BIBLE_DATA",
                source=self.name,
                content=content_obj,
                metadata={
                    "session_id": session_id,
                    "user_goal_summary": user_goal_summary[:50],
                    "key": memory_key,
                    "tool_name": self.name
                }
            )
            await self.memory_manager.add_memory_segment(segment)
            self.logger.info(f"StoryBibleTool: Successfully saved story bible to memory (key: {memory_key}).")
        except Exception as e:
            self.logger.error(f"StoryBibleTool: Error saving story bible (key: {memory_key}): {e}")
            # Depending on desired behavior, might raise or return an error status

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
        effective_context = context if context is not None else {}
        session_id = effective_context.get("session_id")
        user_goal_summary = effective_context.get("user_goal_summary")

        if not session_id or not user_goal_summary:
            self.logger.error("StoryBibleTool: session_id or user_goal_summary missing from context.")
            return ToolResponse(
                status_code=400,
                output=StoryBibleToolResponseOutput(
                    status="error", 
                    message="session_id and user_goal_summary are required in context to use StoryBibleTool."
                )
            )

        memory_key = self._get_story_bible_memory_key(session_id, user_goal_summary)
        story_bible = await self._load_story_bible(memory_key)
        response_data: Optional[Dict[str, Any]] = None

        try:
            if args.action == "add":
                if not args.path or args.content is None:
                    return ToolResponse(status_code=400, output=StoryBibleToolResponseOutput(status="error", message="'path' and 'content' are required for 'add' action."))
                
                # For 'add', we assume content is a dict. If it's a string, it should be wrapped in a dict by the caller if meant for JSON structure.
                self._set_value_at_path(story_bible, args.path, args.content, overwrite=True) # Defaulting to overwrite=True for simplicity, can be parameterized
                await self._save_story_bible(memory_key, story_bible, session_id, user_goal_summary)
                response_data = args.content
                return ToolResponse(output=StoryBibleToolResponseOutput(status="success", message=f"Entry added/updated at '{args.path}'.", data=response_data, path=args.path))

            elif args.action == "get":
                if not args.path:
                    return ToolResponse(status_code=400, output=StoryBibleToolResponseOutput(status="error", message="'path' is required for 'get' action."))
                
                retrieved_value = self._get_value_at_path(story_bible, args.path)
                if retrieved_value is not None:
                    # Ensure data is a dict for the response model, even if a single value is retrieved.
                    # If retrieved_value is not a dict, wrap it or adjust response model.
                    # For now, assuming retrieved_value can be any JSON-serializable type, but response.data expects Dict.
                    # This might need refinement based on how it's used.
                    response_data = retrieved_value if isinstance(retrieved_value, dict) else {"value": retrieved_value}
                    return ToolResponse(output=StoryBibleToolResponseOutput(status="success", message=f"Entry retrieved from '{args.path}'.", data=response_data, path=args.path))
                else:
                    return ToolResponse(output=StoryBibleToolResponseOutput(status="not_found", message=f"No entry found at '{args.path}'.", path=args.path), status_code=404)

            elif args.action == "get_all":
                response_data = story_bible
                return ToolResponse(output=StoryBibleToolResponseOutput(status="success", message="Full story bible retrieved.", data=response_data))
            
            else:
                # Should not happen due to Literal type in args
                return ToolResponse(status_code=400, output=StoryBibleToolResponseOutput(status="error", message=f"Unknown action: {args.action}"))

        except Exception as e:
            self.logger.exception(f"StoryBibleTool: Error during action '{args.action}' (path: {args.path}): {e}")
            return ToolResponse(
                status_code=500, 
                output=StoryBibleToolResponseOutput(status="error", message=f"An unexpected error occurred: {str(e)}", path=args.path)
            )
