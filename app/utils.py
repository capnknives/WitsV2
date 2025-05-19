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

    logger.info(f"Creating new agent for session {session_id}, profile {profile_name}")    # profile_data is now the Pydantic model for the specific agent profile
    if not profile_data.agent_class:
        raise ValueError(f"No agent_class specified in profile '{profile_name}'")
    
    # Split the full path into module components and class name
    module_path_parts = profile_data.agent_class.split('.')
    if len(module_path_parts) < 2:
        raise ValueError(f"Invalid agent_class path in profile '{profile_name}': {profile_data.agent_class}")

    class_name = module_path_parts[-1]
    module_path = '.'.join(module_path_parts[:-1])

    try:
        # Import the parent module first
        parent_path = '.'.join(module_path_parts[:-2]) if len(module_path_parts) > 2 else module_path_parts[0]
        if parent_path:
            logger.debug(f"Attempting to import parent module '{parent_path}'")
            importlib.import_module(parent_path)

        # Then try importing the direct module
        logger.debug(f"Attempting to import module '{module_path}' for agent class '{class_name}'")
        agent_module = importlib.import_module(module_path)
    except ImportError as e:
        logger.error(f"Failed to import agent module {module_path}: {e}", exc_info=True)
        raise ValueError(f"Could not import agent module for profile '{profile_name}': {e}")

    try:
        logger.debug(f"Getting class '{class_name}' from module '{module_path}'")
        agent_class_constructor = getattr(agent_module, class_name)
    except AttributeError as e:
        logger.error(f"Failed to get class {class_name} from module {module_path}: {e}", exc_info=True)
        raise ValueError(f"Could not find agent class '{class_name}' in module '{module_path}' for profile '{profile_name}'")
        
    # Verify that the class is a proper agent class
    if not issubclass(agent_class_constructor, BaseAgent):
        raise ValueError(f"Class '{class_name}' in profile '{profile_name}' is not a subclass of BaseAgent")

    # LLM Interface for this agent
    # profile_data is an AgentProfileConfig instance.
    # app_config is an AppConfig instance.
    profile_llm_model = profile_data.llm_model_name or app_config.models.default # Correct: app_config.models.default
    
    llm_key = f"{session_id}_{profile_name}_llm" 
    if llm_key not in SESSION_LLMS or SESSION_LLMS[llm_key].model_name != profile_llm_model:
        logger.info(f"Creating new LLM interface for agent {agent_key} with model {profile_llm_model}")
        
        # Handle temperature with guaranteed float value
        temperature = 0.7  # Default temperature
        if profile_data.temperature is not None:
            temperature = float(profile_data.temperature)
        elif hasattr(app_config, 'default_temperature') and app_config.default_temperature is not None:
            temperature = float(app_config.default_temperature)
        
        agent_llm_interface = LLMInterface(
            ollama_url=app_config.ollama_url, # Use attribute access
            model_name=profile_llm_model,
            request_timeout=app_config.ollama_request_timeout, # Use attribute access
            temperature=temperature # Now this is guaranteed to be a float
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
    }

    # If the agent class is OrchestratorAgent, prepare its delegation_targets
    if class_name == "OrchestratorAgent":
        delegation_targets_map: Dict[str, BaseAgent] = {}
        # profile_data is an AgentProfileConfig instance
        target_profile_names = profile_data.delegation_target_profile_names or []
        
        logger.debug(f"Orchestrator profile '{profile_name}' has delegation targets: {target_profile_names}")

        for target_name in target_profile_names:
            if target_name == profile_name: # Avoid self-delegation loop at this stage
                logger.warning(f"Skipping self-delegation for {target_name} in orchestrator profile {profile_name}")
                continue
            try:
                logger.debug(f"Creating delegate agent '{target_name}' for orchestrator '{profile_name}'")
                # Recursively get the delegate agent instance
                delegate_agent = await get_or_create_agent_for_session(
                    session_id,
                    target_name,
                    app_config,
                    global_memory_manager,
                    global_tool_registry
                )
                delegation_targets_map[target_name] = delegate_agent
                logger.debug(f"Successfully created delegate agent '{target_name}' for orchestrator '{profile_name}'")
            except Exception as e:
                logger.error(f"Failed to create delegate agent '{target_name}' for orchestrator '{profile_name}': {e}", exc_info=True)
                # Skip this delegate agent

        agent_init_params["delegation_targets"] = delegation_targets_map
        if "max_iterations" in profile_data.agent_specific_params:
            agent_init_params["max_iterations"] = profile_data.agent_specific_params["max_iterations"]

    # Create and store the agent instance
    agent_instance = agent_class_constructor(**agent_init_params)
    SESSION_AGENTS[agent_key] = agent_instance
    logger.info(f"Successfully created agent instance for profile '{profile_name}'")
    
    return agent_instance

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
    
    # Ensure target_temperature is a float, falling back to a default if necessary
    current_llm_temp = current_llm.temperature if current_llm and hasattr(current_llm, 'temperature') and current_llm.temperature is not None else default_temp_from_profile
    
    if new_temperature is not None:
        target_temperature = new_temperature
    elif current_llm_temp is not None:
        target_temperature = current_llm_temp
    elif default_temp_from_profile is not None: # Ensure default_temp_from_profile is not None
        target_temperature = default_temp_from_profile
    else: # Final fallback if all else is None
        target_temperature = 0.7 # Default fallback temperature

    if not isinstance(target_temperature, float): # Explicitly cast if it's somehow not float
        try:
            target_temperature = float(target_temperature)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert target_temperature \'{target_temperature}\' to float. Using default 0.7.")
            target_temperature = 0.7


    if not current_llm or \
       (new_model_name and new_model_name != current_llm.model_name) or \
       (new_temperature is not None and new_temperature != current_llm.temperature):
        
        logger.info(f"Creating/Replacing LLM for {llm_key} with model: {target_model_name}, temp: {target_temperature}")
        updated_llm = LLMInterface(
            ollama_url=app_config.ollama_url, # Attribute access
            model_name=target_model_name,
            request_timeout=app_config.ollama_request_timeout, # Attribute access
            temperature=target_temperature # This should now be a float
        )
        SESSION_LLMS[llm_key] = updated_llm
        agent_instance.llm = updated_llm 
        # if hasattr(updated_llm, 'temperature_ui_override'): 
        #      updated_llm.temperature_ui_override = target_temperature

        logger.info(f"Agent {agent_key}'s LLM instance updated to new instance with model {target_model_name}, temp {target_temperature}.")
        return updated_llm
    
    logger.debug(f"LLM for {llm_key} did not require changes or no update parameters provided.")
    return current_llm