# Helper function (e.g., in app/main.py or app/utils.py)
import importlib
from typing import Dict, Any

# Assuming your BaseAgent and other core components are correctly pathed
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager
from core.tool_registry import ToolRegistry as CoreToolRegistry # Rename to avoid conflict if any
from core.config import Config as CoreConfig
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
    agent_profiles = app_config.get("agent_profiles", {})
    profile = agent_profiles.get(profile_name)

    if not profile:
        logger.error(f"Agent profile '{profile_name}' not found. Falling back to default.")
        default_profile_name = app_config.get("default_agent_profile_name", "general_orchestrator")
        profile = agent_profiles.get(default_profile_name)
        if not profile: # Should not happen if default is configured
             raise ValueError(f"Default agent profile '{default_profile_name}' also not found.")
        profile_name = default_profile_name # Update profile_name to actual used

    # Construct a unique key for the agent instance based on session and profile
    # This allows a session to switch agents and get a fresh instance for that profile.
    agent_key = f"{session_id}_{profile_name}"

    if agent_key in SESSION_AGENTS:
        logger.debug(f"Reusing existing agent for session {session_id}, profile {profile_name}")
        return SESSION_AGENTS[agent_key]

    logger.info(f"Creating new agent for session {session_id}, profile {profile_name}")

    module_path, class_name = profile["agent_class"].rsplit('.', 1)
    try:
        agent_module = importlib.import_module(module_path)
        agent_class_constructor = getattr(agent_module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import agent class {profile['agent_class']}: {e}", exc_info=True)
        raise ValueError(f"Invalid agent class specified in profile '{profile_name}': {profile['agent_class']}")

    # LLM Interface for this agent
    # Check if profile specifies a unique model different from the global/default one
    profile_llm_model = profile.get("llm_model_name", app_config.get('llm_model_name'))
    
    # Use a dedicated LLM instance for this agent_key if model is different or to ensure isolation
    llm_key = f"{session_id}_{profile_name}_llm" # Unique key for this agent's LLM
    if llm_key not in SESSION_LLMS or SESSION_LLMS[llm_key].model_name != profile_llm_model:
        logger.info(f"Creating new LLM interface for agent {agent_key} with model {profile_llm_model}")
        agent_llm_interface = LLMInterface(
            ollama_url=app_config.get('ollama_url'),
            model_name=profile_llm_model, # Use profile-specific model
            request_timeout=app_config.get('ollama_request_timeout', 120)
            # Add temperature from profile if needed, or let UI control it for this instance
        )
        # await agent_llm_interface.check_connection_async() # Good practice
        SESSION_LLMS[llm_key] = agent_llm_interface
    else:
        logger.debug(f"Reusing existing LLM interface for agent {agent_key}")
        agent_llm_interface = SESSION_LLMS[llm_key]


    # Tool Registry for this agent profile
    agent_tool_registry = CoreToolRegistry()
    profile_tool_names = profile.get("tool_names", [])
    for tool_name in profile_tool_names:
        tool_instance = global_tool_registry.get_tool(tool_name)
        if tool_instance:
            agent_tool_registry.register_tool(tool_instance)
        else:
            logger.warning(f"Tool '{tool_name}' for profile '{profile_name}' not found in global registry.")

    agent_init_params = {
        "llm_interface": agent_llm_interface,
        "memory_manager": global_memory_manager, # Shared memory, or could be session-specific
        "tool_registry": agent_tool_registry,
        "max_iterations": profile.get("max_iterations", app_config.get('max_iterations', 5)),
    }
    if "system_prompt_override" in profile:
        agent_init_params["system_prompt"] = profile["system_prompt_override"]

    # Add any other agent_specific_params from config
    if "agent_specific_params" in profile:
        agent_init_params.update(profile["agent_specific_params"])
    
    try:
        agent_instance = agent_class_constructor(**agent_init_params)
        SESSION_AGENTS[agent_key] = agent_instance
        return agent_instance
    except Exception as e:
        logger.error(f"Failed to instantiate agent {profile['agent_class']} for profile {profile_name}: {e}", exc_info=True)
        raise ValueError(f"Could not create agent from profile '{profile_name}'. Check agent class and parameters.")
    
    # In app/utils.py (add this new function)

from typing import Optional # Ensure typing.Optional is imported

# (SESSION_AGENTS and SESSION_LLMS are already defined in your utils.py)
# (LLMInterface, CoreConfig should already be imported in your utils.py)

async def update_llm_for_session_agent(
    session_id: str,
    profile_name: str, # The currently active profile name for the session
    app_config: CoreConfig,
    new_model_name: Optional[str] = None,
    new_temperature: Optional[float] = None
) -> Optional[LLMInterface]:
    """
    Updates an existing LLM instance or creates a new one for a given session's agent profile,
    stores it in SESSION_LLMS, and updates the agent in SESSION_AGENTS to use it.
    Returns the updated or new LLMInterface instance.
    """
    llm_key = f"{session_id}_{profile_name}_llm"
    agent_key = f"{session_id}_{profile_name}"

    current_llm = SESSION_LLMS.get(llm_key)
    agent_instance = SESSION_AGENTS.get(agent_key)

    if not agent_instance:
        logger.warning(f"Agent {agent_key} not found while trying to update its LLM. Cannot update LLM.")
        return None # Or raise an error

    # Determine the model to use
    target_model_name = new_model_name if new_model_name else (current_llm.model_name if current_llm else profile_name) # Fallback logic
    
    # Determine temperature
    target_temperature = new_temperature if new_temperature is not None else (current_llm.temperature if current_llm else 0.7)


    # If model changes, or if no current_llm, a new LLM instance is needed
    if not current_llm or (new_model_name and new_model_name != current_llm.model_name):
        logger.info(f"Creating/Replacing LLM for {llm_key} with model: {target_model_name}, temp: {target_temperature}")
        updated_llm = LLMInterface(
            ollama_url=app_config.get('ollama_url'),
            model_name=target_model_name,
            request_timeout=app_config.get('ollama_request_timeout', 120),
            temperature=target_temperature
        )
        # await updated_llm.check_connection_async() # Optional: check connection
        SESSION_LLMS[llm_key] = updated_llm
        agent_instance.llm = updated_llm # Update the agent's llm reference
        if hasattr(updated_llm, 'temperature_ui_override'): # If you use this for UI state
             updated_llm.temperature_ui_override = target_temperature

        logger.info(f"Agent {agent_key}'s LLM instance updated to new instance with model {target_model_name}.")
        return updated_llm
    # If only temperature changes for the existing LLM
    elif new_temperature is not None and current_llm and new_temperature != current_llm.temperature:
        logger.info(f"Updating temperature for LLM {llm_key} (model: {current_llm.model_name}) to {new_temperature}")
        current_llm.temperature = new_temperature
        if hasattr(current_llm, 'temperature_ui_override'):
            current_llm.temperature_ui_override = new_temperature
        # The agent_instance.llm already points to this current_llm, so it's updated.
        return current_llm
    
    logger.debug(f"LLM for {llm_key} did not require changes or no update parameters provided.")
    return current_llm # Return existing if no changes needed for it