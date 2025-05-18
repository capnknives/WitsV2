# Helper function (e.g., in app/main.py or app/utils.py)
import importlib
import logging # Add this line
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__) # Add this line

# Assuming your BaseAgent and other core components are correctly pathed
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager
from core.tool_registry import ToolRegistry as CoreToolRegistry # Rename to avoid conflict if any
from core.config import AppConfig as CoreConfig
from agents.base_agent import BaseAgent


# Store for session-specific agent instances
# In a real multi-user app, this would be tied to a proper session store (e.g., Redis, DB)
# For simplicity, a global dict keyed by session_id. Be mindful of memory in a real deployment.
SESSION_AGENTS: Dict[str, BaseAgent] = {}
SESSION_LLMS: Dict[str, LLMInterface] = {} # If agents need dedicated LLM instances

async def get_or_create_agent_for_session(
    session_id: str,
    profile_name: str,
    app_config: CoreConfig, # The main application config object
    global_memory_manager: MemoryManager,
    global_tool_registry: CoreToolRegistry
) -> BaseAgent:
    """
    Retrieves or creates an agent instance for a given session and profile.
    Manages agent-specific LLM instances if profile specifies a different model.
    """
    agent_profiles = app_config.agent_profiles # Use attribute access
    profile_data = agent_profiles.get(profile_name) if agent_profiles else None # Use .get() on the dict

    if not profile_data:
        logger.error(f"Agent profile '{profile_name}' not found. Falling back to default.")
        default_profile_name = app_config.default_agent_profile_name # Use attribute access
        profile_data = agent_profiles.get(default_profile_name) if agent_profiles else None # Use .get()
        if not profile_data: # Should not happen if default is configured
             raise ValueError(f"Default agent profile '{default_profile_name}' also not found.")
        profile_name = default_profile_name # Update profile_name to actual used

    # Construct a unique key for the agent instance based on session and profile
    # This allows a session to switch agents and get a fresh instance for that profile.
    agent_key = f"{session_id}_{profile_name}"

    if agent_key in SESSION_AGENTS:
        logger.debug(f"Reusing existing agent for session {session_id}, profile {profile_name}")
        return SESSION_AGENTS[agent_key]

    logger.info(f"Creating new agent for session {session_id}, profile {profile_name}")

    # profile_data is now the Pydantic model for the specific agent profile
    module_path, class_name = profile_data.agent_class.rsplit('.', 1) # Access agent_class directly
    try:
        agent_module = importlib.import_module(module_path)
        agent_class_constructor = getattr(agent_module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import agent class {profile_data.agent_class}: {e}", exc_info=True)
        raise ValueError(f"Invalid agent class specified in profile '{profile_name}': {profile_data.agent_class}")

    # LLM Interface for this agent
    # profile_data is an AgentProfileConfig instance.
    # app_config is an AppConfig instance.
    profile_llm_model = profile_data.llm_model_name or app_config.models.default # Correct: app_config.models.default
    
    llm_key = f"{session_id}_{profile_name}_llm" 
    if llm_key not in SESSION_LLMS or SESSION_LLMS[llm_key].model_name != profile_llm_model:
        logger.info(f"Creating new LLM interface for agent {agent_key} with model {profile_llm_model}")
        agent_llm_interface = LLMInterface(
            ollama_url=app_config.ollama_url, # Use attribute access
            model_name=profile_llm_model,
            request_timeout=app_config.ollama_request_timeout, # Use attribute access
            temperature=profile_data.temperature if profile_data.temperature is not None else app_config.default_temperature # Use attribute access
        )
        SESSION_LLMS[llm_key] = agent_llm_interface
    else:
        logger.debug(f"Reusing existing LLM interface for agent {agent_key}")
        agent_llm_interface = SESSION_LLMS[llm_key]

    # Tool Registry for this agent profile
    agent_tool_registry = CoreToolRegistry(config=app_config) # Pass app_config to ToolRegistry
    profile_tool_names = profile_data.tool_names or [] # Access tool_names
    for tool_name in profile_tool_names:
        tool_instance = global_tool_registry.get_tool(tool_name)
        if tool_instance:
            agent_tool_registry.register_tool(tool_instance)
        else:
            logger.warning(f"Tool '{tool_name}' for profile '{profile_name}' not found in global registry.")

    # Prepare agent_init_params using direct attribute access on profile_data (Pydantic model)
    # and app_config (Pydantic model)
    agent_init_params = {
        "agent_name": profile_name, # Pass the agent's profile name
        "config": profile_data, # Pass the specific agent profile config (Pydantic model)
        "llm_interface": agent_llm_interface,
        "memory_manager": global_memory_manager,
        "tool_registry": agent_tool_registry,
        # max_iterations is often part of the agent's own config or a global default
        # BaseAgent constructor expects 'config' which is the agent_profile_config
    }
    # BaseAgent's __init__ takes agent_name, config (which is the profile_data here), llm_interface, memory_manager.
    # OrchestratorAgent adds tool_registry and specialized_agents.
    # We need to ensure the 'config' passed to BaseAgent is the specific agent's profile config.
    # The OrchestratorAgent specific 'max_iterations' comes from its config.

    # If the agent class is OrchestratorAgent, it might expect 'specialized_agents'
    # This part needs to be more dynamic if other agent types have different required params.
    # For now, assuming OrchestratorAgent or similar BaseAgent derivatives.
    # The 'config' parameter in BaseAgent.__init__ is used to get agent_config.
    # So, profile_data (which is an AgentProfile) should be passed as 'config'.

    # The OrchestratorAgent's __init__ also sets self.max_iterations from self.config_full.orchestrator_max_iterations
    # self.config_full is derived from the 'config' (AgentProfile) passed to it.
    # So, if AgentProfile has an orchestrator_max_iterations field, it will be used.
    # Otherwise, we might need to pass it explicitly if it's a global config.
    # Let's assume AgentProfile model in core.config.py can have orchestrator_max_iterations.

    try:
        # agent_class_constructor is OrchestratorAgent or similar
        agent_instance = agent_class_constructor(**agent_init_params)
        SESSION_AGENTS[agent_key] = agent_instance
        return agent_instance
    except Exception as e:
        logger.error(f"Failed to instantiate agent {profile_data.agent_class} for profile {profile_name}: {e}", exc_info=True)
        # Include the original error message for better diagnostics
        raise ValueError(f"Could not create agent from profile '{profile_name}'. Original error: {type(e).__name__}: {str(e)}. Check agent class and parameters.")

# In app/utils.py (add this new function)

# (SESSION_AGENTS and SESSION_LLMS are already defined in your utils.py)
# (LLMInterface, CoreConfig should already be imported in your utils.py)

async def update_llm_for_session_agent(
    session_id: str,
    profile_name: str, 
    app_config: CoreConfig,
    new_model_name: Optional[str] = None,
    new_temperature: Optional[float] = None
) -> Optional[LLMInterface]:
    llm_key = f"{session_id}_{profile_name}_llm"
    agent_key = f"{session_id}_{profile_name}"

    current_llm = SESSION_LLMS.get(llm_key)
    agent_instance = SESSION_AGENTS.get(agent_key)

    if not agent_instance:
        logger.warning(f"Agent {agent_key} not found while trying to update its LLM. Cannot update LLM.")
        return None

    # Determine the model to use
    target_model_name = new_model_name if new_model_name else (current_llm.model_name if current_llm else app_config.models.default)
    
    # Determine temperature
    # Fallback to agent's current config temp, then global default
    agent_profile_config = agent_instance.config_full # This should be the AgentProfile Pydantic model
    default_temp_from_profile = agent_profile_config.temperature if agent_profile_config and agent_profile_config.temperature is not None else app_config.default_temperature
    target_temperature = new_temperature if new_temperature is not None else (current_llm.temperature if current_llm and hasattr(current_llm, 'temperature') and current_llm.temperature is not None else default_temp_from_profile)


    if not current_llm or \
       (new_model_name and new_model_name != current_llm.model_name) or \
       (new_temperature is not None and new_temperature != current_llm.temperature):
        
        logger.info(f"Creating/Replacing LLM for {llm_key} with model: {target_model_name}, temp: {target_temperature}")
        updated_llm = LLMInterface(
            ollama_url=app_config.ollama_url, # Attribute access
            model_name=target_model_name,
            request_timeout=app_config.ollama_request_timeout, # Attribute access
            temperature=target_temperature
        )
        SESSION_LLMS[llm_key] = updated_llm
        agent_instance.llm = updated_llm 
        # if hasattr(updated_llm, 'temperature_ui_override'): 
        #      updated_llm.temperature_ui_override = target_temperature

        logger.info(f"Agent {agent_key}'s LLM instance updated to new instance with model {target_model_name}, temp {target_temperature}.")
        return updated_llm
    
    logger.debug(f"LLM for {llm_key} did not require changes or no update parameters provided.")
    return current_llm