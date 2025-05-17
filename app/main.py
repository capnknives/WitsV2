"""Main FastAPI application server for WITS NEXUS v2."""
from typing import Dict, Any, Optional
import logging
from fastapi import FastAPI
import pydantic

from core.config import AppConfig, load_app_config, AgentProfileConfig
from core.memory_manager import MemoryManager
from core.tool_registry import ToolRegistry

# Only import LLM interface if not skipping initialization
app_config: AppConfig = load_app_config()
if not getattr(app_config, 'skip_llm_initialization', False):
    from core.llm_interface import LLMInterface
    from agents.orchestrator_agent import OrchestratorAgent
    from agents.specialized.analyst_agent import AnalystAgent
    from agents.specialized.engineer_agent import EngineerAgent
    from agents.specialized.researcher_agent import ResearcherAgent
    from agents.specialized.scribe_agent import ScribeAgent

logger = logging.getLogger('WITS.API')
app = FastAPI(title="WITS NEXUS v2")

# Initialize based on config
if not getattr(app_config, 'skip_llm_initialization', False):
    # Map of agent types to their classes
    AGENT_TYPES = {
        "orchestrator": OrchestratorAgent,
        "analyst": AnalystAgent,
        "engineer": EngineerAgent, 
        "researcher": ResearcherAgent,
        "scribe": ScribeAgent
    }
    
    # Global session stores
    SESSION_AGENTS: Dict[str, Any] = {}  # Keyed by 'session_id_profile'
    SESSION_LLMS: Dict[str, LLMInterface] = {}  # Keyed by 'session_id_profile_llm'
else:
    AGENT_TYPES = {}
    SESSION_AGENTS = {}
    SESSION_LLMS = {}

memory_manager_global = MemoryManager(app_config)
tool_registry_global = ToolRegistry(app_config)

async def update_llm_for_session_agent(
    session_id: str,
    profile_name: str,
    app_config: AppConfig,
    new_model_name: Optional[str] = None,
    new_temperature: Optional[float] = None
) -> Optional[LLMInterface]:
    """
    Updates an existing LLM instance or creates a new one for a given session's agent profile,
    stores it in SESSION_LLMS, and updates the agent in SESSION_AGENTS to use it.
    """
    llm_key = f"{session_id}_{profile_name}_llm"
    agent_key = f"{session_id}_{profile_name}"

    current_llm = SESSION_LLMS.get(llm_key)
    agent_instance = SESSION_AGENTS.get(agent_key)

    if not agent_instance:
        logger.warning(f"Agent {agent_key} not found when trying to update its LLM. Attempting to create agent first.")
        try:
            agent_instance = await get_or_create_agent_for_session(
                session_id, profile_name, app_config,
                memory_manager_global,
                tool_registry_global
            )
            current_llm = SESSION_LLMS.get(llm_key)
        except ValueError as ve:
            logger.error(f"Could not create agent {agent_key} to update its LLM: {ve}")
            return None

    # Access agent profiles from app_config
    agent_profiles = app_config.agent_profiles if hasattr(app_config, 'agent_profiles') else {}
    profile_data = agent_profiles.get(profile_name, AgentProfileConfig())  # Fallback to default if profile not found

    target_model_name = new_model_name
    if not target_model_name:
        if current_llm and hasattr(current_llm, 'model_name'):
            target_model_name = current_llm.model_name
        else:
            target_model_name = profile_data.llm_model_name or getattr(app_config, 'llm_model_name', None)
    if not target_model_name:
        raise ValueError("No model name specified and no default found in config")

    target_temperature = new_temperature
    if target_temperature is None:
        if current_llm and hasattr(current_llm, 'temperature'):
            target_temperature = current_llm.temperature
        else:
            target_temperature = profile_data.temperature

    if not current_llm or (new_model_name and getattr(current_llm, 'model_name', None) != new_model_name):
        logger.info(f"Creating/Replacing LLM for {llm_key} with model: {target_model_name}, temp: {target_temperature}")
        
        llm_config = {
            'model_name': target_model_name,
            'temperature': target_temperature,
        }
        
        # Add optional config if present
        if hasattr(app_config, 'ollama_url'):
            llm_config['ollama_url'] = app_config.ollama_url
        if hasattr(app_config, 'ollama_request_timeout'):
            llm_config['request_timeout'] = app_config.ollama_request_timeout
            
        updated_llm = LLMInterface(**llm_config)
        
        SESSION_LLMS[llm_key] = updated_llm
        if agent_instance:
            agent_instance.llm = updated_llm
        logger.info(f"Agent {agent_key}'s LLM instance updated to new instance with model {target_model_name}.")
        return updated_llm
    
    elif new_temperature is not None and current_llm and hasattr(current_llm, 'temperature'):
        if new_temperature != current_llm.temperature:
            logger.info(f"Updating temperature for LLM {llm_key} (model: {getattr(current_llm, 'model_name', 'unknown')}) to {new_temperature}")
            setattr(current_llm, 'temperature', new_temperature)
        return current_llm
    
    logger.debug(f"LLM for {llm_key} (model: {getattr(current_llm, 'model_name', 'N/A') if current_llm else 'N/A'}) did not require significant changes or no update parameters were sufficient.")
    return current_llm

async def get_or_create_agent_for_session(
    session_id: str,
    profile_name: str,
    app_config: AppConfig,
    memory_manager: MemoryManager,
    tool_registry: ToolRegistry
) -> Any:
    """Create or retrieve an agent for a session and profile combination."""
    agent_key = f"{session_id}_{profile_name}"

    if agent_key in SESSION_AGENTS:
        return SESSION_AGENTS[agent_key]

    # Get agent profiles from app_config
    agent_profiles = app_config.agent_profiles if hasattr(app_config, 'agent_profiles') else {}

    # Get the profile's LLM model name (fallback to config default if needed)
    profile_data = agent_profiles.get(profile_name, AgentProfileConfig())
    profile_llm_model = profile_data.llm_model_name or getattr(app_config, 'llm_model_name', None)
    
    if not profile_llm_model:
        raise ValueError("No LLM model specified in profile or config defaults")

    # Create or reuse LLM for this profile
    llm_key = f"{agent_key}_llm"
    llm_instance = SESSION_LLMS.get(llm_key)

    if not llm_instance or (hasattr(llm_instance, 'model_name') and llm_instance.model_name != profile_llm_model):
        # Initialize new LLM with settings from profile/config
        llm_config = {
            'model_name': profile_llm_model,
            'temperature': profile_data.temperature
        }
        
        # Add optional config if present
        if hasattr(app_config, 'ollama_url'):
            llm_config['ollama_url'] = app_config.ollama_url
        if hasattr(app_config, 'ollama_request_timeout'):
            llm_config['request_timeout'] = app_config.ollama_request_timeout
            
        llm_instance = LLMInterface(**llm_config)
        SESSION_LLMS[llm_key] = llm_instance

    # Create agent instance based on profile type
    agent_type = profile_data.agent_type  # Has default of "orchestrator"
    agent_class = AGENT_TYPES.get(agent_type)
    if not agent_class:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent_instance = agent_class(
        llm=llm_instance,
        memory_manager=memory_manager,
        tool_registry=tool_registry,
        config={
            "max_iterations": profile_data.max_iterations
        }
    )

    SESSION_AGENTS[agent_key] = agent_instance
    logger.info(
        f"Created new agent for session {session_id} with profile {profile_name} "
        f"(type: {agent_type}, model: {profile_llm_model})"
    )
    
    return agent_instance

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    try:
        await memory_manager_global.initialize_db()
        logger.info("Memory manager initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing memory manager: {e}")
        raise