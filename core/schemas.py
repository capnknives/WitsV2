# Welcome to Schema Land! Where data structures come alive! \\o/
# core/schemas.py
from typing import Dict, Any, Optional, List  # Type hints make everything better! ^_^
from datetime import datetime  # Time keeper extraordinaire! =D
import uuid  # UUID: Because random numbers aren't random enough! >.>
from pydantic import BaseModel, Field  # Our data validation superhero! \\o/

class MemorySegmentContent(BaseModel):
    """
    The juicy bits of our memories! Like a sandwich filling, but for data! ^_^
    """
    text: Optional[str] = None  # The actual words (if any) =D
    tool_name: Optional[str] = None  # Which tool did the thing? \\o/
    tool_args: Optional[Dict[str, Any]] = None  # Tool settings and stuff =P
    tool_output: Optional[str] = None  # What did the tool say? O.o

class MemorySegment(BaseModel):
    """Our memory segment model! Each piece of info we wanna remember ^.^"""
    id: str = Field(default_factory=lambda: f"mem_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uuid.uuid4().hex[:6]}")  # Super unique ID! No duplicates allowed! ^_^
    timestamp: datetime = Field(default_factory=datetime.now)  # When did it happen? Time is weird! >.>
    type: str  # What kind of memory is this? (USER_INPUT, LLM_THOUGHT, etc) =D
    source: str  # Who/what created this memory? Everyone's a contributor! \\o/
    content: MemorySegmentContent  # The actual stuff we want to remember! ^_^
    importance: float = 0.5  # How important is it? (0 = meh, 1 = SUPER important!) =P
    embedding: Optional[List[float]] = None  # Vector magic for finding similar memories! O.o
    relevance_score: Optional[float] = None  # How close to what we're looking for? \\o/
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Extra info for the curious! x.x

    class Config:
        arbitrary_types_allowed = True  # Because sometimes types need to be free! \\o/

class LLMToolCall(BaseModel):
    """
    When our AI friends want to use tools! Like a function call, but fancier! ^_^
    """
    tool_name: str  # Which tool do they want? =D
    args: Dict[str, Any]  # What settings should we use? \\o/
    explanation: Optional[str] = None  # Why are we doing this? (Always good to know! >.>)

class OrchestratorThought(BaseModel):
    """
    What's going on in our AI's mind? Let's peek! \\o/
    Deep thoughts from our digital friend! So philosophical! x.x
    """
    reasoning: str  # The "why" behind the action ^_^
    plan: Optional[List[str]] = None  # The master plan! (hopefully it's a good one =P)
    concerns: Optional[List[str]] = None  # What could go wrong? Everything! O.o
    additional_info: Optional[Dict[str, Any]] = None  # Random but important stuff! >.>

class OrchestratorAction(BaseModel):
    """
    Time to take action! What's our AI going to do? =D
    Could be using a tool, talking to us, or plotting world domination! 
    (Just kidding about that last one... I hope! x.x)
    """
    action_type: str  # What kind of action? (USE_TOOL, RESPOND, etc) \\o/
    tool_calls: Optional[List[LLMToolCall]] = None  # Tools to use! Let's get crafty! ^_^
    response: Optional[str] = None  # What to tell the humans =P
    delegate_to: Optional[str] = None  # Pass it to another agent? Tag, you're it! O.o

class OrchestratorLLMResponse(BaseModel):
    """
    The complete thought process and decision from our AI friend! \\o/
    This is where all the magic comes together! *waves wand* ✨
    """
    thought_process: OrchestratorThought  # What were they thinking? O.o
    chosen_action: OrchestratorAction  # What did they decide to do? =D
    confidence: Optional[float] = None  # How sure are they? (0-1, like test scores! x.x)
    processing_time_ms: Optional[int] = None  # How long did they think about it? >.>

class StreamData(BaseModel):
    """
    Our real-time update system! Because waiting is boring! \\o/
    Keeping everyone in the loop with fresh updates! =D
    """
    type: str  # What kind of update is this? ^_^
    content: Any  # The actual stuff we want to share! O.o
    tool_name: Optional[str] = None  # Did a tool help with this? =P
    tool_args: Optional[Dict[str, Any]] = None  # Tool settings (if any) >.>
    iteration: Optional[int] = None  # Which step are we on? x.x
    max_iterations: Optional[int] = None  # How many steps total? \\o/
    # For deep thoughts
    reasoning: Optional[str] = None  # The "why" behind it all ^_^
    plan: Optional[List[str]] = None  # Our grand scheme! =D
    # For oopsies
    error_details: Optional[str] = None  # What went wrong? (It happens! x.x)
    # Extra bits for our control center
    goal_statement: Optional[str] = None  # What are we trying to do? O.o
    clarification_question: Optional[str] = None  # Need help understanding? >.>

    class Config:
        arbitrary_types_allowed = True  # Freedom for all types! \\o/

class BookProjectBase(BaseModel):
    """
    The foundation of every great book project! Let's write something epic! ^_^
    """
    project_name: str  # What shall we call this masterpiece? =D
    description: Optional[str] = None  # What's it all about? \\o/
    genre: Optional[str] = None  # Fantasy? Sci-fi? Romance? The world is our oyster! O.o

class BookProjectCreate(BookProjectBase):
    """Time to birth a new book project into existence! *dramatic music* \\o/"""
    pass

class BookProjectUpdate(BookProjectBase):
    """
    Something needs changing? No problem! We're flexible! ^_^
    Edit all the things! (Responsibly, of course! x.x)
    """
    project_name: Optional[str] = None  # Even names can change! =P

class BookProject(BookProjectBase):
    """
    The fully-formed book project in all its glory! \\o/
    Complete with timestamps and everything! So professional! =D
    """
    id: str  # The project's unique identity! O.o
    created_at: datetime = Field(default_factory=datetime.now)  # Birthday! ^_^
    last_modified_at: datetime = Field(default_factory=datetime.now)  # Last time we poked it >.>

    class Config:
        orm_mode = True  # ORM friendly! Because databases need love too =P
        arbitrary_types_allowed = True  # Type freedom! \\o/

class MemoryConfig(BaseModel):
    """Configuration for our memory system! Time to get organized! =P"""
    vector_model: str = Field(default="all-MiniLM-L6-v2", description="The model to use for vectorization ^.^")
    memory_file: Optional[str] = Field(default=None, description="Where to save our memories! \\o/")
    debug_enabled: bool = Field(default=False, description="Enable debug mode? O.o")
    debug_components: Dict[str, bool] = Field(
        default_factory=lambda: {
            "log_embeddings": False,
            "log_searches": False,
            "log_initialization": False,
            "log_additions": False
        },
        description="What to debug? =D"
    )
