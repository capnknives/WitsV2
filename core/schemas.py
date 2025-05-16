# core/schemas.py
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
from pydantic import BaseModel, Field

class MemorySegment(BaseModel):
    id: str = Field(default_factory=lambda: f"mem_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uuid.uuid4().hex[:6]}")
    timestamp: datetime = Field(default_factory=datetime.now)
    type: str # E.g., "USER_INPUT", "LLM_THOUGHT", "TOOL_CALL", "TOOL_RESULT", "ORCHESTRATOR_RESPONSE" 
    source: str # E.g., "USER", "ORCHESTRATOR_LLM", "WebSearchTool", "ORCHESTRATOR_AGENT"
    metadata: Dict[str, Any] = Field(default_factory=dict)  # For search distances, debug info, etc.

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

# --- Memory Schemas (can be expanded from your v1) ---
class MemorySegmentContent(BaseModel):
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_output: Optional[str] = None
    # ... other content types

class MemorySegment(BaseModel):
    id: str = Field(default_factory=lambda: f"mem_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uuid.uuid4().hex[:6]}")
    timestamp: datetime = Field(default_factory=datetime.now)
    type: str # E.g., "USER_INPUT", "LLM_THOUGHT", "TOOL_CALL", "TOOL_RESULT", "ORCHESTRATOR_RESPONSE"
    source: str # E.g., "USER", "ORCHESTRATOR_LLM", "WebSearchTool", "ORCHESTRATOR_AGENT"
    content: MemorySegmentContent
    importance: float = 0.5
    embedding: Optional[List[float]] = None

    class Config:
        arbitrary_types_allowed = True
