\
# test_orchestrator_run.py
import asyncio
import logging
import os
import sys
import json # Added for dummy manifest
from datetime import datetime
from typing import Dict, Any, Optional, AsyncGenerator, List, Union # Added Union

# Adjust path to import from sibling directories
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.orchestrator_agent import OrchestratorAgent
from agents.base_agent import BaseAgent
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager
from core.tool_registry import ToolRegistry
# Import necessary Pydantic models from core.config
import core.config # Import the module itself to allow mocking load_app_config
from core.config import (
    AppConfig, AgentProfileConfig, ModelsConfig, WebInterfaceConfig,
    MemoryManagerConfig, DebugConfig, GitIntegrationConfig, RouterConfig,
    DebugComponentsConfig, DebugComponentConfig # Ensure all nested types are imported
)
from core.schemas import StreamData

# --- Early definitions for Mocks and Fallbacks ---

class MockBookAgent(BaseAgent):
    def __init__(self, agent_name: str, profile_config: AgentProfileConfig, llm_interface: LLMInterface, memory_manager: MemoryManager, tool_registry: ToolRegistry, app_config_param: AppConfig):
        super().__init__(agent_name=agent_name, config=profile_config, llm_interface=llm_interface, memory_manager=memory_manager, tool_registry=tool_registry)
        self.app_config = app_config_param # Store it if needed by MockBookAgent
        self.logger = logging.getLogger(f"WITS.MockBookAgent.{agent_name}")
        self.logger.info(f"MockBookAgent \'{agent_name}\' initialized.")

    async def run(self, user_input_or_task: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[StreamData, None]:
        """
        Run the MockBookAgent with the given input/task.
        
        Args:
            user_input_or_task: The user input or delegated task (matches the parameter name expected by OrchestratorAgent)
            context: Optional context dictionary with additional information
            
        Yields:
            StreamData chunks for the agent's output
        """
        self.logger.info(f"MockBookAgent \'{self.agent_name}\' received task: {user_input_or_task}")
        
        # Extract project name from the task if available
        project_name = "Chronicles of the Starlight Drifter"  # Hardcoded for this test
        
        # Simulate some work with appropriate delays for realistic behavior
        yield StreamData(type="thought", content=f"MockBookAgent thinking about book project: {project_name}")
        await asyncio.sleep(0.1)
        
        yield StreamData(type="tool_call", content=json.dumps({
            "tool_name": "mock_book_tool", 
            "arguments": {"project_name": project_name}
        }))
        await asyncio.sleep(0.1)
        
        yield StreamData(type="tool_result", content=json.dumps({
            "tool_name": "mock_book_tool", 
            "result": "Mock book project created successfully."
        }))
        await asyncio.sleep(0.1)
        
        yield StreamData(type="final_response", content=f"MockBookAgent finished processing book project: {project_name}. Ready for further instructions.")

    async def process_stream_response(
        self, 
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        json_mode: bool = False
    ) -> AsyncGenerator[StreamData, None]:
        # Simply delegate to run() for this mock implementation
        async for chunk in self.run(prompt, context):
            yield chunk

class MinimalToolRegistry(ToolRegistry):
    def __init__(self, config: AppConfig):
        # super().__init__(config) # Avoid calling parent's __init__ if it does complex things like file loading
        self.config = config
        self.tools: Dict[str, Any] = {}
        self.logger = logging.getLogger("WITS.MinimalToolRegistry")
        self.logger.info("Initialized a minimal, empty ToolRegistry.")
        self._tool_schemas: Dict[str, Dict[str, Any]] = {} # Added to match parent class attribute

    def register_tool(self, tool_instance: Any, tool_name: Optional[str] = None) -> None: # Match signature
        self.logger.info(f"MinimalToolRegistry: Ignoring register_tool call for {tool_name or tool_instance}")
        pass

    def get_tool(self, tool_name: str) -> Optional[Any]:
        self.logger.info(f"MinimalToolRegistry: get_tool called for {tool_name}, returning None.")
        return None

    def get_all_tools(self) -> Dict[str, Any]:
        return self.tools

    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]: # Match signature
        self.logger.info(f"MinimalToolRegistry: get_tool_schema called for {tool_name}, returning None.")
        return None

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())
    
    def load_tools_from_manifest(self, manifest_path: Optional[str] = None) -> None: # Match signature
        self.logger.info("MinimalToolRegistry: load_tools_from_manifest called, doing nothing.")
        pass


# --- Configuration & Mocks ---

# 1. Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestScript")

# 2. AppConfig
app_config_data = {
    "app_name": "Test WITS Orchestrator",
    "output_directory": "output/test_orchestrator_run",
    "memory_manager": {
        "vector_model": "mock-embedding-model",
        "memory_file_path": "data/memory/test_orchestrator_run_memory.json"
    },
    "debug": {
        "enabled": True, "log_level": "INFO", "console_logging_enabled": True,
        "console_log_level": "INFO", "file_logging_enabled": True,
        "log_directory": "logs/test_orchestrator_run",
        "performance_monitoring": False,
        "components": {
            "llm_interface": {"log_prompts": True, "log_responses": True, "log_tokens": False, "log_args": False, "log_results": False, "log_embeddings": False, "log_searches": False, "log_thoughts": False, "log_actions": False, "log_delegations": False},
            "memory_manager": {"log_prompts": False, "log_responses": False, "log_tokens": False, "log_args": False, "log_results": False, "log_embeddings": True, "log_searches": True, "log_thoughts": False, "log_actions": False, "log_delegations": False},
            "tools": {"log_prompts": False, "log_responses": False, "log_tokens": False, "log_args": True, "log_results": True, "log_embeddings": False, "log_searches": False, "log_thoughts": False, "log_actions": False, "log_delegations": False},
            "agents": {"log_prompts": True, "log_responses": True, "log_tokens": False, "log_args": False, "log_results": False, "log_embeddings": False, "log_searches": False, "log_thoughts": True, "log_actions": True, "log_delegations": True}
        }
    },
    "models": {
        "default": "mock-model-default",
        "orchestrator": "mock-model-orchestrator",
        "scribe": "mock-model-scribe",
        "analyst": "mock-model-analyst",
        "engineer": "mock-model-engineer",
        "researcher": "mock-model-researcher",
        "planner": "mock-model-planner"
    },
    "agent_profiles": {
        "test_orchestrator_profile": {
            "agent_class": "agents.orchestrator_agent.OrchestratorAgent",
            "agent_type": "orchestrator",
            "llm_model_name": "orchestrator", 
            "temperature": 0.6,
            "max_iterations": 3,
            "orchestrator_max_iterations": 3,
            "tool_names": [],
            "system_prompt_override": "You are a helpful test orchestrator.",
            "agent_specific_params": {"description": "Test Orchestrator for project name extraction."}
        },
        "mock_book_agent_profile": {
            "agent_class": "test_orchestrator_run.MockBookAgent",
            "agent_type": "specialized",
            "llm_model_name": "default", 
            "temperature": 0.6,
            "max_iterations": 1,
            "orchestrator_max_iterations": 1,
            "tool_names": [],
            "system_prompt_override": "You are a mock book agent.",
            "agent_specific_params": {"description": "A mock book agent."}
        }
    },
    "web_interface": {
        "enabled": False, "port": 5002, "host": "127.0.0.1", "debug": False,
        "enable_file_uploads": False, "max_file_size_mb": 5
    },
    "git_integration": {"enabled": False, "repo_path": ".", "auto_commit": False, "git_executable": "git"},
    "internet_access": False,
    "allow_code_execution": False,
    "ethics_enabled": False,
    "default_agent_profile_name": "test_orchestrator_profile",
    "ollama_url": "http://localhost:11433",
    "ollama_request_timeout": 60,
    "default_temperature": 0.6,
    "voice_input": False, # Using alias directly
    "voice_input_duration": 5, # Using alias directly
    "whisper_model": "tiny", # Using alias directly
    "whisper_fp16": False, # Using alias directly
    "router": {"fallback_agent": "test_orchestrator_profile"}
}

# Create AppConfig instance using Pydantic model directly
app_config = AppConfig(
    app_name=app_config_data["app_name"],
    output_directory=app_config_data["output_directory"],
    memory_manager=MemoryManagerConfig(**app_config_data["memory_manager"]),
    debug=DebugConfig(
        enabled=app_config_data["debug"]["enabled"],
        log_level=app_config_data["debug"]["log_level"],
        console_logging_enabled=app_config_data["debug"]["console_logging_enabled"],
        console_log_level=app_config_data["debug"]["console_log_level"],
        file_logging_enabled=app_config_data["debug"]["file_logging_enabled"],
        log_directory=app_config_data["debug"]["log_directory"],
        performance_monitoring=app_config_data["debug"]["performance_monitoring"],
        components=DebugComponentsConfig(
            llm_interface=DebugComponentConfig(**app_config_data["debug"]["components"]["llm_interface"]),
            memory_manager=DebugComponentConfig(**app_config_data["debug"]["components"]["memory_manager"]),
            tools=DebugComponentConfig(**app_config_data["debug"]["components"]["tools"]),
            agents=DebugComponentConfig(**app_config_data["debug"]["components"]["agents"]),
        )
    ),
    models=ModelsConfig(**app_config_data["models"]),
    agent_profiles={name: AgentProfileConfig(**profile) for name, profile in app_config_data["agent_profiles"].items()},
    web_interface=WebInterfaceConfig(**app_config_data["web_interface"]),
    git_integration=GitIntegrationConfig(**app_config_data["git_integration"]),
    internet_access=app_config_data["internet_access"],
    allow_code_execution=app_config_data["allow_code_execution"],
    ethics_enabled=app_config_data["ethics_enabled"],
    default_agent_profile_name=app_config_data["default_agent_profile_name"],
    ollama_url=app_config_data["ollama_url"],
    ollama_request_timeout=app_config_data["ollama_request_timeout"],
    default_temperature=app_config_data["default_temperature"],
    voice_input=app_config_data["voice_input"], # Changed from voice_input_enabled
    voice_input_duration=app_config_data["voice_input_duration"], # Changed from voice_input_duration_seconds
    whisper_model=app_config_data["whisper_model"], # Changed from whisper_model_name
    whisper_fp16=app_config_data["whisper_fp16"],
    router=RouterConfig(**app_config_data["router"])
)

# Mock load_app_config
_original_load_app_config = core.config.load_app_config
def _mock_load_app_config(config_path: Optional[str] = None) -> AppConfig:
    logger.info(f"_mock_load_app_config called, returning pre-defined app_config for testing.")
    return app_config
core.config.load_app_config = _mock_load_app_config


# Ensure directories exist (AppConfig validator for output_directory should handle its own)
if not os.path.exists(app_config.debug.log_directory): # type: ignore
    os.makedirs(app_config.debug.log_directory, exist_ok=True) # type: ignore
memory_dir = os.path.dirname(app_config.memory_manager.memory_file_path)
if not os.path.exists(memory_dir):
    os.makedirs(memory_dir, exist_ok=True)
    logger.info(f"Created memory directory: {memory_dir}")

# 3. Orchestrator's AgentProfileConfig (fetched from AppConfig)
orchestrator_profile_name = app_config.default_agent_profile_name
orchestrator_profile_config = app_config.agent_profiles[orchestrator_profile_name]

# 4. Mock LLMInterface
class MockLLMInterface(LLMInterface):
    def __init__(self, global_config: AppConfig, agent_llm_model_key: str): # agent_llm_model_key is "orchestrator", "default" etc.
        self.app_config = global_config
        self.actual_model_name = getattr(self.app_config.models, agent_llm_model_key, self.app_config.models.default)
        self._model_name = self.actual_model_name
        # Ensure default_temperature is float, provide a fallback if None
        self._temperature: float = global_config.default_temperature if global_config.default_temperature is not None else 0.7
        self.logger = logging.getLogger(f"WITS.MockLLMInterface.{self.actual_model_name}")
        self.logger.info(f"MockLLMInterface initialized for model key '{agent_llm_model_key}\' (actual model: '{self.actual_model_name}', temp: {self._temperature})")
        self.provider_name = "mock"

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._temperature = value

    async def get_llm_response_async(
        self,
        prompt: str,
        temperature: float = 0.7, # This temperature is for this specific call
        max_tokens: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        json_mode: bool = False
    ) -> str:
        self.logger.info(f"MockLLMInterface received prompt (first 100 chars): {prompt[:100]}...")
        self.logger.info(f"MockLLMInterface get_llm_response_async call-specific temperature: {temperature}")
        if json_mode:
            return """
            {
                "thought_process": {
                    "thought": "Mock thought: User wants to do something. Project name seems to be extracted.",
                    "reasoning": "This is a mock response. If project name is correct, I'd proceed.",
                    "plan": ["Delegate to a book agent if project name is valid."]
                },
                "chosen_action": {
                    "action_type": "tool_call",
                    "tool_call": {
                        "tool_name": "book_plotter",
                        "arguments": {"goal": "Create plot for the extracted project."}
                    }
                }
            }
            """
        return "Mock LLM response for non-JSON mode."
    
    # This is the method the OrchestratorAgent is trying to call
    async def chat_completion_async(
        self,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        format: Optional[str] = None
    ) -> str:  # OrchestratorAgent expects a JSON string
        prompt_content = " | ".join([m.get('content', '') for m in messages if m.get('role') == 'user'])
        self.logger.info(f"MockLLMInterface.chat_completion_async called for model: {model_name or self._model_name}")
        self.logger.info(f"MockLLMInterface.chat_completion_async effective prompt: {prompt_content[:100]}...")
        
        # Extract temperature from options if provided
        effective_temperature = self.temperature  # Default to instance temperature
        if options and "temperature" in options:
            effective_temperature = options["temperature"]
            self.logger.info(f"Using temperature from options: {effective_temperature}")
        
        # The OrchestratorAgent always expects a JSON response in a specific format
        json_response = {
            "thought_process": {
                "thought": "Mock thought: User wants to create a book project.",
                "reasoning": "The prompt mentions 'Create a new book project titled \"Chronicles of the Starlight Drifter\"'. This indicates book writing mode.",
                "plan": [
                    "Extract project name.",
                    "Confirm book writing mode.",
                    "Delegate to Book Agent if project name is valid."
                ],
                "critique": "The extraction of the project name needs to be robust.",
                "confidence_score": 0.9
            },
            "chosen_action": {
                "action_type": "tool_call", 
                "tool_call": {  
                    "tool_name": "MockBookAgent",
                    "arguments": {
                        "goal": f"Manage book project: Chronicles of the Starlight Drifter based on user prompt: {prompt_content[:50]}..."
                    }
                }
            },
            "project_name_extracted": "Chronicles of the Starlight Drifter" # This will help OrchestratorAgent
        }
        return json.dumps(json_response)

# 5. MemoryManager

# Mock _initialize_vector_search to prevent actual model loading
_original_initialize_vector_search = MemoryManager._initialize_vector_search
def _mock_initialize_vector_search(self):
    self.logger.info("Mocked _initialize_vector_search: Skipping actual model loading.")
    self.embedding_model = None  # Or a mock object if methods are called on it
    self.index = None
    self.document_map = {}
    self.is_initialized = True # Mark as initialized
MemoryManager._initialize_vector_search = _mock_initialize_vector_search

memory_manager = MemoryManager(config=app_config)

# Restore original method if other tests in the same process might need it
# MemoryManager._initialize_vector_search = _original_initialize_vector_search


# 6. ToolRegistry
tool_manifest_path_for_test = "capabilities/tool_manifest.json" # Relative to workspace root
tool_registry: Optional[ToolRegistry] = None # Corrected type hint
try:
    tool_registry = ToolRegistry(config=app_config) # Constructor might load tools
    manifest_dir_for_tr = os.path.dirname(tool_manifest_path_for_test)
    if not os.path.exists(manifest_dir_for_tr):
        os.makedirs(manifest_dir_for_tr, exist_ok=True)
    # Ensure the manifest file exists, as the ToolRegistry might expect it
    if not os.path.exists(tool_manifest_path_for_test):
        with open(tool_manifest_path_for_test, "w") as f:
            json.dump({"tools": []}, f)
        logger.info(f"Created dummy tool_manifest.json at {tool_manifest_path_for_test}")
    # Removed problematic call: tool_registry.load_tools_from_manifest(tool_manifest_path_for_test)
    # If ToolRegistry loads from a default path or on init, having the file present should be enough.
except Exception as e:
    logger.error(f"Could not initialize main ToolRegistry: {e}", exc_info=True)
    # tool_registry remains None or is set to None explicitly

if tool_registry is None:
    logger.warning("Main ToolRegistry initialization failed or it was None. Using a minimal, empty ToolRegistry.")
    tool_registry = MinimalToolRegistry(config=app_config)


# 7. Delegation Targets (Mock Agents)
delegation_targets: Dict[str, BaseAgent] = {}
mock_book_agent_profile_name = "mock_book_agent_profile"

if mock_book_agent_profile_name in app_config.agent_profiles:
    profile_config = app_config.agent_profiles[mock_book_agent_profile_name]
    agent_llm_model_key = profile_config.llm_model_name or "default"
    mock_llm_for_book_agent = MockLLMInterface(global_config=app_config, agent_llm_model_key=agent_llm_model_key)
    
    agent_instance_name = "MockBookAgent" # Use the known class name directly

    # Ensure tool_registry is not None before passing
    if tool_registry is None: # This check should ideally be redundant due to fallback
        logger.error("ToolRegistry is None even after fallback. Cannot instantiate MockBookAgent.")
        # Handle error appropriately, maybe sys.exit(1) or raise
    else:
        delegation_targets[agent_instance_name] = MockBookAgent(
            agent_name=agent_instance_name,
            profile_config=profile_config,
            llm_interface=mock_llm_for_book_agent,
            memory_manager=memory_manager,
            tool_registry=tool_registry, # Now guaranteed to be at least MinimalToolRegistry
            app_config_param=app_config
        )
        logger.info(f"Instantiated {agent_instance_name} with profile: {mock_book_agent_profile_name}")
else:
    logger.warning(f"Profile \'{mock_book_agent_profile_name}\' not found in app_config.agent_profiles.")

# 8. Instantiate OrchestratorAgent
# llm_model_name in AgentProfileConfig is a key for ModelsConfig (e.g., "orchestrator", "default")
orchestrator_llm_model_key = orchestrator_profile_config.llm_model_name
if not orchestrator_llm_model_key or not hasattr(app_config.models, orchestrator_llm_model_key):
    logger.warning(f"Orchestrator LLM model key '{orchestrator_llm_model_key}' not found in ModelsConfig, using default.")
    orchestrator_llm_model_key = app_config.models.default # Fallback to the actual default model string if key is bad

orchestrator_llm = MockLLMInterface(app_config, orchestrator_llm_model_key)

orchestrator_agent = OrchestratorAgent(
    agent_name=orchestrator_profile_name,
    config=orchestrator_profile_config,
    llm_interface=orchestrator_llm,
    memory_manager=memory_manager,
    delegation_targets=delegation_targets,
    tool_registry=tool_registry, # Pass the (potentially minimal) tool_registry
    max_iterations=3
)

# --- Test Execution ---
async def main_test_runner():
    logger.info("Starting orchestrator test script...")

    user_goal = (
        "Create a new book project titled \'Chronicles of the Starlight Drifter\'. "
        "Set the genre to \'Space Opera\'. Then, navigate to its Story Dashboard. "
        "Add a main character named \'Captain Eva Rostova\' with the description "
        "\'A former Alliance pilot, now a freelance trader with a mysterious past "
        "and a customized freighter, \"The Comet\\'s Tail\".\' Next, add a key plot point: "
        "\'Eva intercepts a coded message hinting at a lost treasure that could shift "
        "the balance of power in the Outer Rim.\'"
    )
    
    initial_context: Dict[str, Any] = {
        "session_id": "test_session_orchestrator_run_001"
    }

    logger.info(f"User Goal: {user_goal}")
    logger.info(f"Initial Context: {initial_context}")

    stream_count = 0
    try:
        async for stream_item in orchestrator_agent.run(user_goal, context=initial_context):
            logger.info(f"Stream Item [{stream_count}]: Type=\'{stream_item.type}\', Content=\'{str(stream_item.content)[:200]}...\'")
            
            if stream_count == 0: # After the first yield (usually an info or goal statement)
                logger.info(f"AGENT STATE (after 1st stream): is_book_writing_mode: {orchestrator_agent.is_book_writing_mode}")
                logger.info(f"AGENT STATE (after 1st stream): current_project_name: {orchestrator_agent.current_project_name}")
                if orchestrator_agent.book_writing_state:
                     logger.info(f"AGENT STATE (after 1st stream): book_writing_state.project_name: {orchestrator_agent.book_writing_state.project_name}")
                else:
                    logger.info(f"AGENT STATE (after 1st stream): book_writing_state is None")
            stream_count += 1
            if stream_count >= 5: 
                logger.info("Reached stream item limit for test.")
                break
    except Exception as e:
        logger.error(f"Error during orchestrator run: {e}", exc_info=True)
    
    logger.info("--- FINAL AGENT STATE CHECK ---")
    logger.info(f"Agent is_book_writing_mode: {orchestrator_agent.is_book_writing_mode}")
    logger.info(f"Agent current_project_name: {orchestrator_agent.current_project_name}")
    if orchestrator_agent.book_writing_state:
         logger.info(f"Agent book_writing_state.project_name: {orchestrator_agent.book_writing_state.project_name}")
    else:
        logger.info(f"Agent book_writing_state is None")

if __name__ == "__main__":
    # Dummy manifest creation is now handled before ToolRegistry instantiation.
    asyncio.run(main_test_runner())
    # Restore original load_app_config
    core.config.load_app_config = _original_load_app_config
