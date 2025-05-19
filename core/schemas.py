# core/schemas.py
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
from pydantic import BaseModel, Field

class MemorySegmentContent(BaseModel):
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_output: Optional[str] = None

class MemorySegment(BaseModel):
    id: str = Field(default_factory=lambda: f"mem_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uuid.uuid4().hex[:6]}")
    timestamp: datetime = Field(default_factory=datetime.now)
    type: str  # E.g., "USER_INPUT", "LLM_THOUGHT", "TOOL_CALL", "TOOL_RESULT", "ORCHESTRATOR_RESPONSE"
    source: str  # E.g., "USER", "ORCHESTRATOR_LLM", "WebSearchTool", "ORCHESTRATOR_AGENT"
    content: MemorySegmentContent
    importance: float = 0.5
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)  # For search distances, debug info, etc.

    class Config:
        arbitrary_types_allowed = True

class LLMToolCall(BaseModel):
    """
    Represents the LLM's intention to call a tool.
    The orchestrator expects the LLM to produce this structure in JSON.
    """
    tool_name: str = Field(..., description="The exact name of the tool to be called.")
    arguments: Dict[str, Any] = Field(default_factory=dict, 
                                     description="A dictionary of arguments for the tool, "
                                                 "conforming to the tool's specific Pydantic args_schema.")

class OrchestratorThought(BaseModel):
    """Represents the thought process of the orchestrator before deciding an action."""
    thought: str
    reasoning: Optional[str] = None
    plan: Optional[List[str]] = None # Optional short-term plan or next steps

class OrchestratorAction(BaseModel):
    """Represents the action chosen by the orchestrator's LLM."""
    action_type: str = Field(..., description="Type of action: 'tool_call', 'delegate_to_agent', 'final_answer', 'clarification_request'")
    tool_call: Optional[LLMToolCall] = None
    delegate_to_agent_key: Optional[str] = None
    delegated_task_description: Optional[str] = None
    final_answer: Optional[str] = None
    clarification_question: Optional[str] = None
    
    # TODO: Add validation to ensure only relevant fields are populated based on action_type

class OrchestratorLLMResponse(BaseModel):
    """Structured response expected from the Orchestrator's LLM after its reasoning step."""
    thought_process: OrchestratorThought
    chosen_action: OrchestratorAction

class StreamData(BaseModel):
    type: str # e.g., "info", "error", "llm_thought", "tool_call", "tool_result", "final_answer", "clarification_request_to_user"
    content: Any
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    iteration: Optional[int] = None
    max_iterations: Optional[int] = None
    # For thoughts
    reasoning: Optional[str] = None
    plan: Optional[List[str]] = None
    # For errors
    error_details: Optional[str] = None
    # For WCCA specific stream types
    goal_statement: Optional[str] = None # When goal is defined
    clarification_question: Optional[str] = None # When clarification is needed by WCCA

    class Config:
        arbitrary_types_allowed = True
