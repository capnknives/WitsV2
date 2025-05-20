\
# test_orchestrator_run.py
import asyncio
import logging
import os
import sys
import json
import pytest
from datetime import datetime
from typing import Dict, Any, Optional, AsyncGenerator, List, Union, Type
from pydantic import BaseModel

# Adjust path to import from sibling directories
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core modules and configs
from agents.base_agent import BaseAgent
from agents.book_orchestrator_agent import BookOrchestratorAgent
from agents.base_orchestrator_agent import BaseOrchestratorAgent
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager, MemoryConfig
from core.tool_registry import ToolRegistry
from core.schemas import StreamData, MemorySegment, MemorySegmentContent
from core import config as core_config
from core.config import (
    AppConfig, AgentProfileConfig, ModelsConfig, WebInterfaceConfig,
    MemoryManagerConfig, DebugConfig, GitIntegrationConfig, RouterConfig,
    DebugComponentsConfig, DebugComponentConfig
)
from agents.book_writing_schemas import BookWritingState, ChapterOutlineSchema, CharacterProfileSchema, WorldAnvilSchema

# Models for responses
class OrchestratorThought(BaseModel):
    thought: str
    reasoning: Optional[str] = None
    
    def model_dump(self) -> Dict[str, Any]:
        return {
            "thought": self.thought,
            "reasoning": self.reasoning
        }

class LLMToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    
    def model_dump(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments
        }

class OrchestratorAction(BaseModel):
    action_type: str
    tool_call: Optional[LLMToolCall] = None
    final_answer: Optional[str] = None
    
    def model_dump(self) -> Dict[str, Any]:
        result = {
            "action_type": self.action_type,
            "final_answer": self.final_answer
        }
        if self.tool_call:
            result["tool_call"] = self.tool_call.model_dump()
        return result

# Mock classes
class MockLLMInterface:
    def __init__(self, response_override=None, model_name="gpt-4"):
        self.response_override = response_override
        self.calls = []
        self.model_name = model_name

    async def agenerate(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        if self.response_override:
            return self.response_override
        return {
            "thought_process": {"thought": "Test thought"},
            "chosen_action": {"action_type": "final_answer", "final_answer": "Test answer"}
        }

    async def chat_completion_async(self, model_name: str, messages: List[Dict[str, str]]) -> Dict[str, str]:
        self.calls.append({"model_name": model_name, "messages": messages})
        if self.response_override:
            return {"response": json.dumps(self.response_override)}
        return {
            "response": json.dumps({
                "thought_process": {"thought": "Test thought"},
                "chosen_action": {"action_type": "final_answer", "final_answer": "Test answer"}
            })
        }

class MockMemoryManager:
    def __init__(self):
        self.segments = []
        
    async def add_memory_segment(self, segment: MemorySegment):
        self.segments.append(segment)
        
    async def get_relevant_segments(self, *args, **kwargs):
        return self.segments

class MockToolRegistry:
    def __init__(self):
        self.tools = {}
        
    def register_tool(self, name, tool):
        self.tools[name] = tool
        
    def get_tool(self, name):
        return self.tools.get(name)

class MockDelegateAgent(BaseAgent):
    def __init__(self, agent_name: str, response_override=None):
        self.agent_name = agent_name
        self.response_override = response_override
        self.calls = []
        
    async def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[StreamData, None]:
        self.calls.append({"task": task_description, "context": context})
        if self.response_override:
            yield StreamData(type="tool_response", content=json.dumps(self.response_override))
        else:
            yield StreamData(type="tool_response", content=json.dumps({"status": "success"}))

# Test fixtures
@pytest.fixture
def mock_llm():
    return MockLLMInterface()

@pytest.fixture
def mock_memory():
    return MockMemoryManager()

@pytest.fixture
def mock_tool_registry():
    return MockToolRegistry()

@pytest.fixture
def delegate_agents():
    return {
        "book_plotter": MockDelegateAgent("book_plotter"),
        "book_character_dev": MockDelegateAgent("book_character_dev"),
        "book_worldbuilder": MockDelegateAgent("book_worldbuilder"),
        "book_prose_generator": MockDelegateAgent("book_prose_generator"),
        "book_editor": MockDelegateAgent("book_editor")
    }

@pytest.fixture
def orchestrator_config():
    return {
        "agent_name": "test_orchestrator",
        "agent_type": "orchestrator",
        "llm_model_name": "gpt-4",
        "model": "gpt-4",
        "temperature": 0.7,
        "max_iterations": 5,
        "delegation_target_profile_names": [
            "book_plotter",
            "book_character_dev",
            "book_worldbuilder",
            "book_prose_generator",
            "book_editor"
        ],
        "agent_specific_params": {
            "max_iterations": 5
        }
    }

@pytest.fixture
def orchestrator(mock_llm, mock_memory, mock_tool_registry, delegate_agents, orchestrator_config):
    agent = BookOrchestratorAgent(
        agent_name="test_orchestrator",
        config=orchestrator_config,
        llm_interface=mock_llm,
        memory_manager=mock_memory,
        tool_registry=mock_tool_registry,
        delegation_targets=delegate_agents,
        max_iterations=5
    )
    return agent

# Test cases
@pytest.mark.asyncio
async def test_orchestrator_initialization(orchestrator):
    assert orchestrator.agent_name == "test_orchestrator"
    assert orchestrator.book_writing_mode is False
    assert orchestrator.book_writing_state.project_name == "Uninitialized"
    assert orchestrator.current_project_name is None
    assert len(orchestrator.delegation_targets) == 5

@pytest.mark.asyncio
async def test_project_initialization(orchestrator):
    user_goal = 'Create a book project named "Test Novel"'
    context = {"session_id": "test_session"}
    
    response_stream = orchestrator.run(user_goal, context)
    
    # Collect all responses
    responses = []
    async for data in response_stream:
        responses.append(data)
        
    # Verify project initialization
    assert orchestrator.book_writing_mode is True
    assert orchestrator.current_project_name == "Test Novel"
    assert orchestrator.book_writing_state.project_name == "Test Novel"
    
    # Verify memory segment was created
    memory_segments = orchestrator.memory.segments
    assert any(seg.type == "book_writing_state" for seg in memory_segments)

@pytest.mark.asyncio
async def test_delegation_flow(orchestrator):
    # Set up a specific test scenario
    orchestrator.book_writing_mode = True
    orchestrator.current_project_name = "Test Novel"
    orchestrator.book_writing_state = BookWritingState(
        project_name="Test Novel",
        overall_plot_summary="A test plot summary",
        character_profiles=[],
        detailed_chapter_outlines=[],
        world_building_notes=WorldAnvilSchema(),
        writing_style_guide=None,
        generated_prose={},
        revision_notes=None
    )
    
    # Mock LLM to trigger delegation
    orchestrator.llm.response_override = {
        "thought_process": {"thought": "Need to develop characters"},
        "chosen_action": {
            "action_type": "tool_call",
            "tool_call": {
                "tool_name": "book_character_dev",  # Remove delegate_to_ prefix
                "arguments": {"goal": "Create main character profiles"}
            }
        }
    }
    
    # Run orchestrator
    user_goal = "Create character profiles for the story"
    context = {"session_id": "test_session"}
    
    responses = []
    async for data in orchestrator.run(user_goal, context):
        responses.append(data)
        
    # Verify delegation occurred and check responses
    character_dev = orchestrator.agents_registry["book_character_dev"]
    assert len(character_dev.calls) > 0, "Character dev agent was not called"
    
    # Verify context passing
    delegation_call = character_dev.calls[0]
    assert "book_writing_state_slice" in delegation_call["context"], "Book writing state slice not passed to delegate"
    state_slice = delegation_call["context"]["book_writing_state_slice"]
    assert state_slice["project_name"] == "Test Novel", "Project name not correctly passed"
    assert "character_profiles" in state_slice, "Character profiles not included in state slice"
    
    # Verify responses contain expected data
    tool_responses = [r for r in responses if isinstance(r, StreamData) and r.type == "tool_response"]
    assert len(tool_responses) > 0, "No tool responses received"
    
    # Verify the task description was passed correctly
    assert delegation_call["task"] == "Create main character profiles", "Incorrect task description passed to delegate"

@pytest.mark.asyncio
async def test_error_handling(orchestrator):
    # Set up a failing delegate agent with an error response
    failing_agent = MockDelegateAgent(
        agent_name="failing_agent", 
        response_override={"status": "error", "message": "Test error"}
    )
    
    # Add failing agent to registry
    orchestrator.agents_registry["failing_agent"] = failing_agent
    
    # Mock LLM to trigger delegation to failing agent
    orchestrator.llm.response_override = {
        "thought_process": {"thought": "Need to use failing agent"},
        "chosen_action": {
            "action_type": "delegate",
            "tool_call": {
                "tool_name": "delegate_to_failing_agent",
                "arguments": {"goal": "This will fail"}
            }
        }
    }
    
    # Run orchestrator
    responses = []
    async for data in orchestrator.run("Test error handling", {"session_id": "test_session"}):
        responses.append(data)
    
    # Verify error response was received
    error_responses = [r for r in responses if r.type == "error"]
    assert len(error_responses) > 0

@pytest.mark.asyncio
async def test_delegate_agent_error_handling(orchestrator):
    # Set up the delegate agent to return an error
    orchestrator.book_writing_mode = True
    orchestrator.current_project_name = "Test Novel"
    
    # Create a delegate agent that will raise an error
    error_agent = MockDelegateAgent("error_agent", response_override={"status": "error", "message": "Test error message"})
    orchestrator.agents_registry["error_agent"] = error_agent
    orchestrator.delegation_targets.append("error_agent")
    
    # Mock LLM to trigger delegation to the error agent
    orchestrator.llm.response_override = {
        "thought_process": {"thought": "Testing error handling"},
        "chosen_action": {
            "action_type": "tool_call",
            "tool_call": {
                "tool_name": "error_agent",  # Remove delegate_to_ prefix
                "arguments": {"goal": "This should trigger an error"}
            }
        }
    }
    
    # Run orchestrator and collect responses
    responses = []
    async for data in orchestrator.run("Test error handling", {"session_id": "test_session"}):
        responses.append(data)
    
    # Verify the error agent was called
    assert len(error_agent.calls) > 0, "Error agent was not called"
    
    # Verify error response was captured
    error_responses = [r for r in responses if isinstance(r, StreamData) and r.type == "tool_response"]
    assert len(error_responses) > 0, "No error response received"
    
    # Verify error message content
    error_data = json.loads(error_responses[0].content)
    assert error_data["status"] == "error", "Error status not captured"
    assert error_data["message"] == "Test error message", "Error message not captured correctly"

if __name__ == "__main__":
    pytest.main([__file__])
