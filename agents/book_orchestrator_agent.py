# agents/book_orchestrator_agent.py
from typing import Any, Dict, List, Optional, AsyncGenerator, Union
import json
import logging
import asyncio
import re
from datetime import datetime

from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager
from core.schemas import StreamData, MemorySegment, MemorySegmentContent
from agents.base_agent import BaseAgent
from agents.base_orchestrator_agent import BaseOrchestratorAgent
from agents.book_writing_schemas import (
    BookWritingState, ChapterOutlineSchema, CharacterProfileSchema, 
    WorldAnvilSchema
)

class BookOrchestratorAgent(BaseOrchestratorAgent):
    """Specialized orchestrator for managing book writing projects."""
    
    def __init__(self, 
                 agent_name: str, 
                 config: Dict[str, Any], 
                 llm_interface: LLMInterface, 
                 memory_manager: MemoryManager, 
                 tool_registry: Optional[Any] = None, 
                 delegation_targets: Optional[Union[Dict[str, BaseAgent], List[str]]] = None, 
                 max_iterations: int = 5):
        super().__init__(agent_name, config, llm_interface, memory_manager, tool_registry, delegation_targets, max_iterations)
        
        # Initialize book writing specific state
        self.book_writing_mode = False
        self.book_writing_state = BookWritingState(project_name="Uninitialized")
        self.current_project_name = None

    def _extract_project_name(self, user_goal: str, context: Dict[str, Any]) -> Optional[str]:
        """Extract project name from user goal or context."""
        # First check context
        if context and "project_name" in context:
            return context["project_name"]
            
        # Then try to extract from goal text
        patterns = [
            r"(?:project named|book named|novel named|story named|project|book|novel|story)\s+(?:\*\*)?['\"]([^'\"\*]+)['\"](?:\*\*)?",
            r"create a new book titled\s+(?:\*\*)?['\"]([^'\"\*]+)['\"](?:\*\*)?",
            r"start book\s+(?:\*\*)?['\"]([^'\"\*]+)['\"](?:\*\*)?",
            r"create a new book project titled\s+(?:\*\*)?['\"]([^'\"\*]+)['\"](?:\*\*)?",
            r"create a new book project named\s+(?:\*\*)?['\"]([^'\"\*]+)['\"](?:\*\*)?"
        ]
        for pattern in patterns:
            match = re.search(pattern, user_goal, re.IGNORECASE)
            if match:
                project_name = match.group(1).strip()
                if project_name.endswith('**'):
                    project_name = project_name[:-2].strip()
                if project_name:
                    return project_name
        return None

    def initialize_book_writing_state(self, project_name: str) -> None:
        """Initialize the book writing state for a new project."""
        self.book_writing_mode = True
        self.book_writing_state = BookWritingState(
            project_name=project_name,
            overall_plot_summary="",
            character_profiles=[],
            detailed_chapter_outlines=[],
            world_building_notes=WorldAnvilSchema(),
            writing_style_guide=None,
            generated_prose={},
            revision_notes=None
        )
        self.logger.info(f"Book writing mode active for project: {project_name} \\o/")

    async def _save_current_book_state_to_memory(self) -> None:
        """Save the current book writing state to memory."""
        try:            # Serialize state to dictionary, excluding None values
            state_dict = self.book_writing_state.model_dump(exclude_none=True)
            
            # Create memory segment
            memory_segment = MemorySegment(
                id=f"book_state_{self.current_project_name}",
                type="book_writing_state",
                source="BookOrchestratorAgent",
                content=MemorySegmentContent(
                    text=f"Book writing state for project '{self.current_project_name}'",
                    tool_output=json.dumps(state_dict)
                ),
                metadata={
                    "project_name": self.current_project_name,
                    "timestamp": datetime.now().isoformat()
                },
                importance=0.8
            )

            # Store in memory
            await self.memory.add_memory_segment(memory_segment)
            self.logger.info(f"Successfully saved book writing state for project '{self.current_project_name}'")
        except Exception as e:
            self.logger.error(f"Failed to save book writing state: {e}")

    async def _handle_agent_delegation(self, agent_name: str, goal: str, context: Dict[str, Any]) -> AsyncGenerator[StreamData, None]:
        """Override to add book-specific state to delegation context."""
        if not self.agents_registry or agent_name not in self.agents_registry:
            self.logger.error(f"No agents_registry or {agent_name} not found in registry.")
            yield StreamData(type="error", content=f"Agent {agent_name} not found in registry")
            return

        delegate_agent = self.agents_registry[agent_name]
        self.logger.info(f"Found agent {delegate_agent}")

        try:
            # Prepare delegation context 
            delegation_context = context.copy() if context else {}
            if self.book_writing_mode and self.book_writing_state:
                agent_specific_state_slice = {}
                
                # Add common fields
                agent_specific_state_slice["project_name"] = self.book_writing_state.project_name
                    
                # Add agent-specific fields
                if agent_name == "book_plotter":
                    agent_specific_state_slice.update({
                        "overall_plot_summary": self.book_writing_state.overall_plot_summary,
                        "detailed_chapter_outlines": [o.model_dump() for o in self.book_writing_state.detailed_chapter_outlines]
                    })
                elif agent_name == "book_character_dev":
                    agent_specific_state_slice.update({
                        "overall_plot_summary": self.book_writing_state.overall_plot_summary,
                        "character_profiles": [p.model_dump() for p in self.book_writing_state.character_profiles]
                    })
                
                delegation_context["book_writing_state_slice"] = agent_specific_state_slice
                self.logger.info(f"Passing book_writing_state_slice to {agent_name} with keys: {list(agent_specific_state_slice.keys())}")

            # Execute delegated task and handle response
            async for data in delegate_agent.run(task_description=goal, context=delegation_context):
                if isinstance(data, StreamData):
                    # Handle state updates from delegates
                    if data.type == "update_state" and self.book_writing_mode:
                        content = data.content
                        if isinstance(content, dict):
                            # Update plot-related data
                            if "overall_plot_summary" in content and content["overall_plot_summary"]:
                                self.book_writing_state.overall_plot_summary = content["overall_plot_summary"]
                                
                            # Update chapter outlines
                            if "detailed_chapter_outlines" in content:
                                try:
                                    chapter_outlines = []
                                    for outline in content["detailed_chapter_outlines"]:
                                        if isinstance(outline, dict):
                                            chapter_outline = ChapterOutlineSchema(**outline)
                                            chapter_outlines.append(chapter_outline)
                                    if chapter_outlines:
                                        self.book_writing_state.detailed_chapter_outlines = chapter_outlines
                                except Exception as e:
                                    self.logger.warning(f"Failed to update chapter outlines: {e}")
                                    
                            # Update character profiles
                            if "character_profiles" in content:
                                try:
                                    character_profiles = []
                                    for profile in content["character_profiles"]:
                                        if isinstance(profile, dict):
                                            character_profile = CharacterProfileSchema(**profile)
                                            character_profiles.append(character_profile)
                                    if character_profiles:
                                        self.book_writing_state.character_profiles = character_profiles
                                except Exception as e:
                                    self.logger.warning(f"Failed to update character profiles: {e}")
                                    
                            # Update world building notes
                            if "world_building_notes" in content and isinstance(content["world_building_notes"], dict):
                                try:
                                    world_anvil = WorldAnvilSchema(**content["world_building_notes"])
                                    self.book_writing_state.world_building_notes = world_anvil
                                except Exception as e:
                                    self.logger.warning(f"Failed to update world building notes: {e}")
                                
                            # Save state to memory after updates
                            asyncio.create_task(self._save_current_book_state_to_memory())
                    
                    yield data
                else:
                    yield StreamData(type="tool_response", content=str(data))

        except Exception as e:
            error_msg = f"Error during delegation to agent '{agent_name}' for session '{context.get('session_id', 'unknown')}': {str(e)}"
            self.logger.error(error_msg)
            yield StreamData(type="error", content=error_msg)

    async def run(self, user_goal: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[StreamData, None]:
        """Run a goal through the book orchestrator."""
        context = context or {}        # Initialize book writing state if needed
        if (project_name := self._extract_project_name(user_goal, context)) is not None:
            self.current_project_name = project_name
            self.book_writing_mode = True
            if not self.book_writing_state or self.book_writing_state.project_name != project_name:
                self.initialize_book_writing_state(project_name)
                # Save initial state
                await self._save_current_book_state_to_memory()
            self.logger.info(f"Book writing mode active for project: {project_name} \\o/")
        elif "book" in user_goal.lower():
            self.logger.warning(f"Book mode detected but no project name found? O.o Goal: {user_goal}")

        # Run the orchestrator
        async for data in super().run(user_goal, context):
            yield data
