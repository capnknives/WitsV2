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
    llm_interface: DebugComponentConfig = Field(default_factory=lambda: DebugComponentConfig())
    memory_manager: DebugComponentConfig = Field(default_factory=lambda: DebugComponentConfig())
    tools: DebugComponentConfig = Field(default_factory=lambda: DebugComponentConfig())
    agents: DebugComponentConfig = Field(default_factory=lambda: DebugComponentConfig())

class DebugConfig(BaseModel):
    enabled: bool = True
    log_level: str = "DEBUG"
    console_logging_enabled: bool = True
    console_log_level: str = "INFO"
    file_logging_enabled: bool = True
    log_directory: str = "logs"
    performance_monitoring: bool = True
    components: DebugComponentsConfig = Field(default_factory=lambda: DebugComponentsConfig())

class GitIntegrationConfig(BaseModel):
    enabled: bool = False
    repo_path: str = "."
    auto_commit: bool = False
    git_executable: str = "git"

class AgentProfileConfig(BaseModel):
    """Configuration for an agent profile."""
    agent_type: str = "orchestrator"  # Default to orchestrator
    llm_model_name: Optional[str] = None
    temperature: float = 0.7
    max_iterations: int = 5
    tool_names: List[str] = Field(default_factory=list)
    system_prompt_override: Optional[str] = None
    agent_specific_params: Dict[str, Any] = Field(default_factory=dict)

class AppConfig(BaseModel):
    app_name: str = "WITS-NEXUS v2"
    internet_access: bool = True
    allow_code_execution: bool = False
    ethics_enabled: bool = True
    output_directory: str = Field(default="output")
    voice_input_enabled: bool = Field(False, alias="voice_input")
    voice_input_duration_seconds: int = Field(5, alias="voice_input_duration")
    whisper_model_name: str = Field("base", alias="whisper_model")
    whisper_use_fp16: bool = Field(False, alias="whisper_fp16")
    models: ModelsConfig = Field(default_factory=lambda: ModelsConfig())
    router: RouterConfig = Field(default_factory=lambda: RouterConfig()) # Though direct call to orchestrator in run.py
    web_interface: WebInterfaceConfig = Field(default_factory=lambda: WebInterfaceConfig())
    memory_manager: MemoryManagerConfig = Field(default_factory=lambda: MemoryManagerConfig())
    debug: DebugConfig = Field(default_factory=lambda: DebugConfig())
    git_integration: GitIntegrationConfig = Field(default_factory=lambda: GitIntegrationConfig())
    agent_profiles: Dict[str, AgentProfileConfig] = Field(default_factory=dict, description="Map of agent profile names to their configurations")
    
    orchestrator_max_iterations: int = 10
    file_tool_base_path: str = Field(default="data/user_files", description="Base path for file operations in FileTool")
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

def load_app_config(config_file_path: str = "config.yaml") -> AppConfig:
    global CONFIG_INSTANCE
    if CONFIG_INSTANCE is None:
        # Default configuration
        default_config = {
            'voice_input': False,
            'voice_input_duration': 5,
            'whisper_model': 'base',
            'whisper_fp16': False,
        }

        # Ensure config_file_path is absolute or relative to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isabs(config_file_path):
            config_file_path = os.path.join(project_root, config_file_path)

        if os.path.exists(config_file_path):
            try:
                with open(config_file_path, 'r', encoding='utf-8') as f:
                    raw_config = yaml.safe_load(f) or {}
                # Merge with defaults
                config_data = {**default_config, **raw_config}
                CONFIG_INSTANCE = AppConfig(**config_data)
                print(f"[Config] Loaded and validated configuration from '{config_file_path}'.")
            except Exception as e:
                print(f"[Config_ERROR] Error loading or validating '{config_file_path}': {e}. Using default config.")
                CONFIG_INSTANCE = AppConfig(**default_config)
        else:
            print(f"[Config_WARN] '{config_file_path}' not found. Using default config and attempting to save it.")
            CONFIG_INSTANCE = AppConfig(**default_config)
            try:
                os.makedirs(os.path.dirname(config_file_path), exist_ok=True)
                with open(config_file_path, 'w', encoding='utf-8') as f:
                    # Handle both Pydantic v1 and v2 compatibility
                    if hasattr(CONFIG_INSTANCE, "model_dump"):
                        config_dict = CONFIG_INSTANCE.model_dump(by_alias=True)
                    else:
                        config_dict = CONFIG_INSTANCE.dict(by_alias=True)
                    yaml.dump(config_dict, f, sort_keys=False)
                print(f"[Config] Saved default configuration to '{config_file_path}'. Please review and customize it.")
            except Exception as e_save:
                print(f"[Config_ERROR] Could not save default configuration: {e_save}")
    return CONFIG_INSTANCE
