# core/config.py
import yaml
import os
import logging
from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, List, Any

class ModelsConfig(BaseModel):
    default: str = "llama3"
    orchestrator: str = "openhermes:latest" # Ensure this is a powerful model
    scribe: Optional[str] = None
    analyst: Optional[str] = None
    engineer: Optional[str] = None
    researcher: Optional[str] = None
    planner: Optional[str] = None
    # Add other agent-specific models here

class RouterConfig(BaseModel):
    fallback_agent: str = "orchestrator_agent"

class WebInterfaceConfig(BaseModel):
    enabled: bool = False
    port: int = 5001
    host: str = "0.0.0.0"
    debug: bool = True
    enable_file_uploads: bool = True
    max_file_size_mb: int = 10

class MemoryManagerConfig(BaseModel):
    vector_model: str = "all-MiniLM-L6-v2"
    memory_file_path: str = "data/memory/wits_memory.json" # Central memory file, stored in dedicated memory directory

class DebugComponentConfig(BaseModel):
    log_prompts: bool = False
    log_responses: bool = False
    log_tokens: bool = False
    log_args: bool = False
    log_results: bool = False
    log_embeddings: bool = False
    log_searches: bool = False
    log_thoughts: bool = False
    log_actions: bool = False
    log_delegations: bool = False

class DebugComponentsConfig(BaseModel):
    llm_interface: DebugComponentConfig = Field(default_factory=DebugComponentConfig)
    memory_manager: DebugComponentConfig = Field(default_factory=DebugComponentConfig)
    tools: DebugComponentConfig = Field(default_factory=DebugComponentConfig)
    agents: DebugComponentConfig = Field(default_factory=DebugComponentConfig)

class DebugConfig(BaseModel):
    enabled: bool = True
    log_level: str = "DEBUG"
    console_logging_enabled: bool = True
    console_log_level: str = "INFO"
    file_logging_enabled: bool = True
    log_directory: str = "logs"
    performance_monitoring: bool = True
    components: DebugComponentsConfig = Field(default_factory=DebugComponentsConfig)

class GitIntegrationConfig(BaseModel):
    enabled: bool = False
    repo_path: str = "."
    auto_commit: bool = False
    git_executable: str = "git"

class AgentProfileConfig(BaseModel):
    """Configuration for an agent profile."""
    agent_class: Optional[str] = None # Added agent_class
    agent_type: str = "orchestrator"  # Default to orchestrator
    llm_model_name: Optional[str] = None
    temperature: float = 0.7
    max_iterations: int = 5
    orchestrator_max_iterations: int = 10  # Added this attribute to match AppConfig
    tool_names: List[str] = Field(default_factory=list)
    system_prompt_override: Optional[str] = None
    agent_specific_params: Dict[str, Any] = Field(default_factory=dict)
    delegation_target_profile_names: Optional[List[str]] = None # Added for orchestrators

    class Config:
        extra = 'ignore' # Ignore extra fields from YAML not defined in Pydantic model

class AppConfig(BaseModel):
    app_name: str = "WITS-NEXUS v2"
    internet_access: bool = True
    allow_code_execution: bool = False
    ethics_enabled: bool = True
    output_directory: str = Field(default="output")
    default_temperature: Optional[float] = 0.7 # Added default_temperature
    default_agent_profile_name: str = "general_orchestrator" # Added default agent profile name
    voice_input_enabled: bool = Field(default=False, alias="voice_input")
    voice_input_duration_seconds: int = Field(default=5, alias="voice_input_duration")
    whisper_model_name: str = Field(default="base", alias="whisper_model")
    whisper_use_fp16: bool = Field(default=False, alias="whisper_fp16")
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    router: RouterConfig = Field(default_factory=RouterConfig) # Though direct call to orchestrator in run.py
    web_interface: WebInterfaceConfig = Field(default_factory=WebInterfaceConfig)
    memory_manager: MemoryManagerConfig = Field(default_factory=MemoryManagerConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    git_integration: GitIntegrationConfig = Field(default_factory=GitIntegrationConfig)
    agent_profiles: Dict[str, AgentProfileConfig] = Field(default_factory=dict, description="Map of agent profile names to their configurations")
    ollama_url: str = "http://localhost:11434"
    ollama_request_timeout: int = 120

    @validator('output_directory', pre=True, always=True)
    def ensure_output_directory_exists(cls, v, values):
        # Resolve relative to project root if not absolute
        # Assuming this config.py is in core/, and project root is one level up from core/
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isabs(v):
            v = os.path.abspath(os.path.join(project_root, v))
        
        try:
            os.makedirs(v, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Could not create output directory '{v}': {e}")
        return v

    class Config:
        extra = 'ignore' # Ignore extra fields from YAML not defined in Pydantic model

CONFIG_INSTANCE: Optional[AppConfig] = None

CONFIG_FILE_PATH = os.getenv('WITS_CONFIG_PATH', 'config.yaml')

_app_config_cache: Optional[AppConfig] = None

def load_app_config(config_path: str = CONFIG_FILE_PATH) -> AppConfig:
    """Loads the application configuration from a YAML file."""
    global _app_config_cache
    if _app_config_cache is not None:
        return _app_config_cache

    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found at {config_path}. Using default values.")
        # Fallback to default Pydantic model if config file is missing
        # This ensures the application can start with defaults, though it might not be fully functional.
        _app_config_cache = AppConfig()
        return _app_config_cache

    try:
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        if raw_config is None:
            logging.warning(f"Configuration file {config_path} is empty. Using default values.")
            _app_config_cache = AppConfig()
            return _app_config_cache

        # Pydantic will validate and parse the raw_config dict into the AppConfig model
        _app_config_cache = AppConfig(**raw_config)
        logging.info(f"Application configuration loaded successfully from {config_path}.")
        return _app_config_cache
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration file {config_path}: {e}", exc_info=True)
        raise ValueError(f"Invalid YAML format in {config_path}") from e
    except Exception as e: # Catch Pydantic validation errors or other issues
        logging.error(f"Error loading or validating application configuration from {config_path}: {e}", exc_info=True)
        # Depending on severity, you might want to raise an error or fall back to defaults
        # For now, re-raise to make configuration issues explicit during startup
        raise ValueError(f"Failed to load or validate AppConfig from {config_path}: {e}") from e
