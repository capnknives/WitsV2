from typing import List, Optional, Dict
from pydantic import BaseModel, ConfigDict

class ChapterOutlineSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    chapter_number: int
    title: Optional[str] = None
    summary: Optional[str] = None
    key_scenes: List[str] = []

class CharacterProfileSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    name: str
    description: Optional[str] = None
    role: Optional[str] = None
    motivations: List[str] = []
    background: Optional[str] = None

class WorldAnvilSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    locations: Dict[str, str] = {} # name: description
    lore: Dict[str, str] = {}      # topic: details
    rules: Dict[str, str] = {}     # rule_name: description

class ChapterProseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    chapter_number: int
    scenes: Dict[str, str] = {} # scene_title_or_number: scene_text

class ScribeOutputSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    formatted_text: str
    summary: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None

class BookWritingState(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    project_name: str
    overall_plot_summary: Optional[str] = None
    detailed_chapter_outlines: List[ChapterOutlineSchema] = []
    character_profiles: List[CharacterProfileSchema] = []
    world_building_notes: Optional[WorldAnvilSchema] = None
    generated_prose: Dict[str, ChapterProseSchema] = {} # chapter_number as string key
    revision_notes: Optional[str] = None
    writing_style_guide: Optional[str] = None # Added for Phase 3
    tone_guide: Optional[str] = None          # Added for Phase 3
