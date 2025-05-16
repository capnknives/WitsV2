# --- wits_nexus_v2/app/main.py ---
import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Optional, Any # Added Any for profile_data typing

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import importlib # Needed for dynamic agent loading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import AppConfig, load_app_config
from core.llm_interface import LLMInterface # Ensure this is imported
from core.memory_manager import MemoryManager
from core.tool_registry import ToolRegistry # Keep as ToolRegistry
from agents.base_agent import BaseAgent

from tools.calculator_tool import CalculatorTool
from tools.datetime_tool import DateTimeTool
from tools.file_tools import FileTool

# --- Logger ---
if not hasattr(logging.getLogger(), 'handlers') or not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models (ensure these are defined as before) ---
class ChatRequest(BaseModel):
    goal: str
    session_id: str = f"web_session_{os.urandom(8).hex()}"
    agent_profile_name: Optional[str] = None

class ParameterUpdateRequest(BaseModel):
    temperature: Optional[float] = None
    model: Optional[str] = None
    session_id: str

class AgentSelectRequest(BaseModel):
    session_id: str
    agent_profile_name: str

# --- Global App State & Session Management Dictionaries (Moved back from utils.py) ---
app_config_global: AppConfig
memory_manager_global: MemoryManager
tool_registry_global: ToolRegistry

SESSION_AGENTS: Dict[str, BaseAgent] = {}
SESSION_LLMS: Dict[str, LLMInterface] = {}
SESSION_CURRENT_AGENT_PROFILE: Dict[str, str] = {}


# --- Helper Functions (Moved back from utils.py and integrated) ---

async def get_or_create_agent_for_session(
    session_id: str,
    profile_name: str,
    app_config: AppConfig,
    global_memory_manager: MemoryManager,
    global_tool_registry: ToolRegistry
) -> BaseAgent:
    """
    Retrieves or creates an agent instance for a given session and profile.
    Manages agent-specific LLM instances if profile specifies a different model.
    (This function is now self-contained in main.py and uses its SESSION_AGENTS, SESSION_LLMS)
    """
    agent_profiles = app_config.get("agent_profiles", {})
    profile_data = agent_profiles.get(profile_name) # Use profile_data for clarity

    if not profile_data:
        logger.warning(f"Agent profile '{profile_name}' not found. Falling back to default.")
        default_profile_name_cfg = app_config.get("default_agent_profile_name", "general_orchestrator")
        profile_data = agent_profiles.get(default_profile_name_cfg)
        if not profile_data:
             raise ValueError(f"Default agent profile '{default_profile_name_cfg}' also not found.")
        profile_name = default_profile_name_cfg # Update profile_name to actual used

    agent_key = f"{session_id}_{profile_name}"

    if agent_key in SESSION_AGENTS:
        logger.debug(f"Reusing existing agent for session {session_id}, profile {profile_name}")
        return SESSION_AGENTS[agent_key]

    logger.info(f"Creating new agent for session {session_id}, profile {profile_name}")

    module_path, class_name = profile_data["agent_class"].rsplit('.', 1)
    try:
        agent_module = importlib.import_module(module_path)
        agent_class_constructor = getattr(agent_module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import agent class {profile_data['agent_class']}: {e}", exc_info=True)
        raise ValueError(f"Invalid agent class specified in profile '{profile_name}': {profile_data['agent_class']}")

    profile_llm_model = profile_data.get("llm_model_name", app_config.get('llm_model_name'))
    llm_key = f"{session_id}_{profile_name}_llm"
    
    agent_llm_interface: LLMInterface
    if llm_key not in SESSION_LLMS or SESSION_LLMS[llm_key].model_name != profile_llm_model:
        logger.info(f"Creating new LLM interface for agent {agent_key} with model {profile_llm_model}")
        agent_llm_interface = LLMInterface(
            ollama_url=app_config.get('ollama_url'),
            model_name=profile_llm_model,
            request_timeout=app_config.get('ollama_request_timeout', 120),
            temperature=profile_data.get("temperature", 0.7) # Default temp from profile or global
        )
        SESSION_LLMS[llm_key] = agent_llm_interface
    else:
        logger.debug(f"Reusing existing LLM interface for agent {agent_key}")
        agent_llm_interface = SESSION_LLMS[llm_key]

    agent_tool_registry = ToolRegistry() # Create a fresh ToolRegistry for this agent
    profile_tool_names = profile_data.get("tool_names", [])
    for tool_name in profile_tool_names:
        tool_instance = global_tool_registry.get_tool(tool_name) # Get from the global_tool_registry
        if tool_instance:
            agent_tool_registry.register_tool(tool_instance)
        else:
            logger.warning(f"Tool '{tool_name}' for profile '{profile_name}' not found in global registry.")

    agent_init_params = {
        "llm_interface": agent_llm_interface,
        "memory_manager": global_memory_manager,
        "tool_registry": agent_tool_registry,
        "max_iterations": profile_data.get("max_iterations", app_config.get('max_iterations', 5)),
    }
    if "system_prompt_override" in profile_data:
        agent_init_params["system_prompt"] = profile_data["system_prompt_override"]
    if "agent_specific_params" in profile_data: # For specialized agent __init__ args
        agent_init_params.update(profile_data["agent_specific_params"])
    
    try:
        agent_instance = agent_class_constructor(**agent_init_params)
        SESSION_AGENTS[agent_key] = agent_instance
        return agent_instance
    except Exception as e:
        logger.error(f"Failed to instantiate agent {profile_data['agent_class']} for profile {profile_name}: {e}", exc_info=True)
        raise ValueError(f"Could not create agent from profile '{profile_name}'. Check agent class and parameters.")

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
    (This function is now self-contained in main.py)
    """
    llm_key = f"{session_id}_{profile_name}_llm"
    agent_key = f"{session_id}_{profile_name}"

    current_llm = SESSION_LLMS.get(llm_key)
    agent_instance = SESSION_AGENTS.get(agent_key)

    if not agent_instance:
        logger.warning(f"Agent {agent_key} not found when trying to update its LLM. Attempting to create agent first.")
        try:
            # Ensure agent (and its initial LLM) exists before trying to update LLM
            agent_instance = await get_or_create_agent_for_session(
                session_id, profile_name, app_config, 
                memory_manager_global, # Assumes global_memory_manager is accessible
                tool_registry_global   # Assumes tool_registry_global is accessible
            )
            current_llm = SESSION_LLMS.get(llm_key) # Re-fetch current_llm after agent creation
        except ValueError as ve:
            logger.error(f"Could not create agent {agent_key} to update its LLM: {ve}")
            return None


    # Determine the model to use for update
    # If new_model_name is provided, use it. Otherwise, use current LLM's model or profile's model.
    agent_profiles = app_config.get("agent_profiles", {})
    profile_data = agent_profiles.get(profile_name, {}) # Get profile data again
    
    target_model_name = new_model_name if new_model_name else \
                        (current_llm.model_name if current_llm else \
                         profile_data.get("llm_model_name", app_config.get('llm_model_name')))
    
    target_temperature = new_temperature if new_temperature is not None else \
                         (current_llm.temperature if current_llm else \
                          profile_data.get("temperature", 0.7))


    if not current_llm or (new_model_name and new_model_name != current_llm.model_name):
        logger.info(f"Creating/Replacing LLM for {llm_key} with model: {target_model_name}, temp: {target_temperature}")
        updated_llm = LLMInterface(
            ollama_url=app_config.get('ollama_url'),
            model_name=target_model_name,
            request_timeout=app_config.get('ollama_request_timeout', 120),
            temperature=target_temperature
        )
        SESSION_LLMS[llm_key] = updated_llm
        if agent_instance: # agent_instance should exist if we reached here after the check
            agent_instance.llm = updated_llm
        logger.info(f"Agent {agent_key}'s LLM instance updated to new instance with model {target_model_name}.")
        return updated_llm
    elif new_temperature is not None and current_llm and new_temperature != current_llm.temperature:
        logger.info(f"Updating temperature for LLM {llm_key} (model: {current_llm.model_name}) to {new_temperature}")
        current_llm.temperature = new_temperature
        return current_llm
    
    logger.debug(f"LLM for {llm_key} (model: {current_llm.model_name if current_llm else 'N/A'}) did not require significant changes or no update parameters were sufficient.")
    return current_llm

# --- FastAPI Lifespan & App (largely unchanged) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_config_global, memory_manager_global, tool_registry_global
    logger.info("Starting WITS-NEXUS v2 Web Application (Lifespan)...")
    try:
        app_config_global = load_app_config('config.yaml')
        
        memory_manager_global = MemoryManager(
            config=app_config_global,
            memory_file_path=app_config_global.memory_manager.memory_file_path
        )
        await memory_manager_global.initialize_db_async()

        tool_registry_global = ToolRegistry()
        tool_registry_global.register_tool(CalculatorTool())
        tool_registry_global.register_tool(DateTimeTool())
        file_tool_base = app_config_global.file_tool_base_path or 'data/user_files'
        if not os.path.exists(file_tool_base):
            os.makedirs(file_tool_base, exist_ok=True)
        tool_registry_global.register_tool(FileTool(base_path=file_tool_base))

        app.state.app_config = app_config_global
        app.state.memory_manager = memory_manager_global
        app.state.tool_registry = tool_registry_global
        logger.info("WITS-NEXUS global components initialized.")
    except Exception as e:
        logger.error(f"Critical error during WITS-NEXUS initialization: {e}", exc_info=True)
        app.state.app_config = None
    yield
    logger.info("Shutting down WITS-NEXUS v2 Web Application...")
    # Example: Close LLM sessions if your LLMInterface has a close method
    # for llm_instance in SESSION_LLMS.values():
    #    if hasattr(llm_instance, 'close_async'): await llm_instance.close_async()
    logger.info("Shutdown complete.")

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


# --- API Routes (Ensure they use the locally defined helper functions) ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/session/agent")
async def select_agent_for_session_endpoint(agent_select: AgentSelectRequest, request: Request): # Renamed for clarity
    app_cfg = request.app.state.app_config
    if not app_cfg:
        raise HTTPException(status_code=503, detail="Application config not available.")
    
    agent_profiles = app_cfg.get("agent_profiles", {})
    if agent_select.agent_profile_name not in agent_profiles:
        raise HTTPException(status_code=404, detail=f"Agent profile '{agent_select.agent_profile_name}' not found.")

    SESSION_CURRENT_AGENT_PROFILE[agent_select.session_id] = agent_select.agent_profile_name
    logger.info(f"Session '{agent_select.session_id}' switched to agent profile: {agent_select.agent_profile_name}")
    
    try:
        # Call the local get_or_create_agent_for_session
        await get_or_create_agent_for_session(
            agent_select.session_id,
            agent_select.agent_profile_name,
            app_cfg, # from app.state
            request.app.state.memory_manager, # from app.state
            request.app.state.tool_registry   # from app.state
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    return {"message": f"Agent profile for session '{agent_select.session_id}' set to '{agent_select.agent_profile_name}'.",
            "session_id": agent_select.session_id,
            "agent_profile_name": agent_select.agent_profile_name}

@app.get("/api/agents")
async def list_agent_profiles_api(request: Request):
    app_cfg = request.app.state.app_config
    if not app_cfg:
        raise HTTPException(status_code=503, detail="Application config not available.")
    profiles = app_cfg.get("agent_profiles", {})
    return [
        {"name": name, "display_name": data.get("display_name", name), "description": data.get("description", "")}
        for name, data in profiles.items()
    ]

# stream_agent_process_updated calling the local get_or_create_agent_for_session
async def stream_agent_process_updated(goal: str, session_id: str, request: Request): # request is FastAPI Request
    app_cfg = request.app.state.app_config
    memory_mgr = request.app.state.memory_manager
    global_tools_reg = request.app.state.tool_registry # Corrected name

    if not app_cfg or not memory_mgr or not global_tools_reg:
        yield '{"type": "error", "content": "Core application components not available."}\n'
        return

    active_profile_name = SESSION_CURRENT_AGENT_PROFILE.get(session_id)
    if not active_profile_name:
        active_profile_name = app_cfg.get("default_agent_profile_name", "general_orchestrator")
        SESSION_CURRENT_AGENT_PROFILE[session_id] = active_profile_name
        logger.info(f"No agent profile for session '{session_id}', using default: {active_profile_name}")

    try:
        agent_instance = await get_or_create_agent_for_session( # Calls local version
            session_id, active_profile_name, app_cfg, memory_mgr, global_tools_reg
        )
    except ValueError as e: # Catch specific error from agent creation
        logger.error(f"ValueError creating agent for streaming: {e}", exc_info=True)
        yield f'{{"type": "error", "content": "Failed to initialize agent: {str(e)}"}}\n'
        return
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error getting agent for streaming: {e}", exc_info=True)
        yield f'{{"type": "error", "content": "Unexpected error initializing agent: {str(e)}"}}\n'
        return

    if not agent_instance:
        yield '{"type": "error", "content": "Agent instance is None after creation attempt."}\n'
        return
    
    # ... (The rest of the ReAct streaming loop remains the same as in the previous version of main.py)
    current_iteration = 0
    max_iterations = agent_instance.max_iterations
    previous_actions = []
    final_answer_generated = False
    
    yield f'{{"type": "info", "content": "Using agent: {active_profile_name} (Class: {agent_instance.__class__.__name__})"}}\n'
    if hasattr(agent_instance.llm, 'model_name'):
        yield f'{{"type": "info", "content": "Agent LLM Model: {agent_instance.llm.model_name}, Temp: {agent_instance.llm.temperature}"}}\n'

    try:
        while current_iteration < max_iterations:
            current_iteration += 1
            yield f'{{"type": "iteration_update", "iteration": {current_iteration}, "max_iterations": {max_iterations}}}\n'
            prompt_context = await agent_instance._build_llm_prompt_async(goal, previous_actions)
            llm_response_json_str = await agent_instance.llm.get_completion_async(prompt_context, json_mode=True)
            if not llm_response_json_str:
                yield '{"type": "error", "content": "LLM returned empty response."}\n'; break
            parsed_response = await agent_instance._parse_llm_response_async(llm_response_json_str)
            if not parsed_response:
                err_msg = f"Failed to parse LLM response: {llm_response_json_str[:200]}..."
                yield f'{{"type": "error", "content": "{err_msg}"}}\n'
                previous_actions.append({"thought": "Error parsing.", "action_type": "error", "action_input": {}, "observation": err_msg})
                continue
            yield f'{{"type": "thought", "content": "{parsed_response.thought}"}}\n'
            if parsed_response.action.type == "final_answer":
                answer = parsed_response.action.input.get('answer', 'No answer provided.')
                yield f'{{"type": "final_answer", "content": "{answer}"}}\n'
                if hasattr(agent_instance, 'memory_manager') and agent_instance.memory_manager:
                    await agent_instance.memory_manager.add_interaction_async(goal, answer, session_id)
                final_answer_generated = True; break
            elif parsed_response.action.type == "tool_call":
                tool_name = parsed_response.action.input.get("tool_name")
                tool_args = parsed_response.action.input.get("tool_args", {})
                yield f'{{"type": "action", "tool_name": "{tool_name}", "tool_args": {repr(tool_args)}}}\n'
                try:
                    observation = await agent_instance.tool_registry.use_tool_async(tool_name, tool_args)
                    yield f'{{"type": "observation", "content": "{str(observation)[:1000]}"}}\n'
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name} by agent {active_profile_name}: {e}", exc_info=True)
                    observation = f"Error executing tool {tool_name}: {str(e)}"
                    yield f'{{"type": "error", "tool_name": "{tool_name}", "content": "{str(e)}"}}\n'
                previous_actions.append({"thought": parsed_response.thought, "action_type": "tool_call", "action_input": {"tool_name": tool_name, "tool_args": tool_args}, "observation": observation})
            else:
                unknown_action_msg = f"Unknown action type: {parsed_response.action.type}"
                yield f'{{"type": "error", "content": "{unknown_action_msg}"}}\n'
                previous_actions.append({"thought": parsed_response.thought, "action_type": "unknown", "action_input": {}, "observation": unknown_action_msg})
            if current_iteration >= max_iterations and not final_answer_generated:
                yield '{"type": "max_iterations_reached", "content": "Max iterations reached."}\n'; break
        if not final_answer_generated and current_iteration >= max_iterations:
            yield '{"type": "info", "content": "Process finished (max iterations)."}\n'
        elif final_answer_generated:
            yield '{"type": "info", "content": "Process finished (final answer)."}\n'
    except Exception as e:
        logger.error(f"Error during agent processing for goal '{goal}' with agent '{active_profile_name}': {e}", exc_info=True)
        yield f'{{"type": "error", "content": "Internal server error during agent processing: {str(e)}"}}\n'


@app.post("/api/chat_stream")
async def chat_stream_endpoint(chat_request: ChatRequest, request: Request):
    if chat_request.agent_profile_name:
        app_cfg = request.app.state.app_config
        agent_profiles = app_cfg.get("agent_profiles", {})
        if chat_request.agent_profile_name in agent_profiles:
            SESSION_CURRENT_AGENT_PROFILE[chat_request.session_id] = chat_request.agent_profile_name
            logger.info(f"Session '{chat_request.session_id}' will use agent profile from request: {chat_request.agent_profile_name}")
        else:
            logger.warning(f"Requested agent profile '{chat_request.agent_profile_name}' not found. Session will use its current or default.")
    return StreamingResponse(
        stream_agent_process_updated(chat_request.goal, chat_request.session_id, request),
        media_type="application/x-ndjson"
    )

@app.post("/api/config/parameters")
async def update_llm_parameters_endpoint(params: ParameterUpdateRequest, request: Request):
    app_cfg = request.app.state.app_config
    if not app_cfg:
        raise HTTPException(status_code=503, detail="Application config not available.")

    active_profile_name = SESSION_CURRENT_AGENT_PROFILE.get(params.session_id)
    if not active_profile_name:
        raise HTTPException(status_code=400, detail=f"No active agent profile for session '{params.session_id}'. Select an agent.")

    logger.info(f"Updating LLM params for session '{params.session_id}', profile '{active_profile_name}' "
                f"to model: {params.model}, temp: {params.temperature}")
    try:
        # Calls the local update_llm_for_session_agent function
        updated_llm = await update_llm_for_session_agent(
            session_id=params.session_id,
            profile_name=active_profile_name,
            app_config=app_cfg, # from app.state
            new_model_name=params.model,
            new_temperature=params.temperature
        )
        if updated_llm:
            return {"message": f"LLM for agent '{active_profile_name}' (session '{params.session_id}') "
                               f"updated. Effective model: {updated_llm.model_name}, temp: {updated_llm.temperature}."}
        else:
             return {"message": (f"LLM for agent '{active_profile_name}' (session '{params.session_id}') "
                                f"update check complete. No changes made or agent not found for update.")}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error updating LLM for session {params.session_id}, profile {active_profile_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected error updating LLM parameters.")


@app.get("/api/session/parameters")
async def get_session_llm_parameters_api(session_id: str, request: Request):
    app_cfg = request.app.state.app_config
    if not app_cfg: raise HTTPException(status_code=503, detail="App config not available.")
    
    active_profile_name = SESSION_CURRENT_AGENT_PROFILE.get(session_id)
    current_model_name = app_cfg.get("llm_model_name") 
    current_temperature = 0.7 

    if active_profile_name:
        agent_key = f"{session_id}_{active_profile_name}"
        llm_key = f"{session_id}_{active_profile_name}_llm"
        
        # Check if LLM instance exists in our session cache
        if llm_key in SESSION_LLMS:
            llm_instance = SESSION_LLMS[llm_key]
            current_model_name = llm_instance.model_name
            current_temperature = llm_instance.temperature
        else: # If not, it might be created on first use or profile specifies it
            profile_data = app_cfg.get("agent_profiles", {}).get(active_profile_name, {})
            current_model_name = profile_data.get("llm_model_name", current_model_name)
            current_temperature = profile_data.get("temperature", current_temperature) # if temp is in profile
            
    return {
        "session_id": session_id,
        "active_agent_profile": active_profile_name,
        "model": current_model_name,
        "temperature": current_temperature,
        "available_models": app_cfg.get("available_ollama_models", [])
    }

# (Memory API endpoints /api/memory/search, /api/memory/clear remain as previously defined in main.py)
@app.get("/api/memory/search")
async def search_memory_api(query: str, session_id: str, request: Request):
    memory_mgr_instance = request.app.state.memory_manager
    if not memory_mgr_instance:
        raise HTTPException(status_code=503, detail="Memory Manager not available.")
    results = await memory_mgr_instance.search_memory_async(query, session_id=session_id, k=5)
    return {"query": query, "results": results}

@app.post("/api/memory/clear")
async def clear_memory_api(data: Dict[str, str], request: Request):
    session_id = data.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required in the request body.")
    memory_mgr_instance = request.app.state.memory_manager
    if not memory_mgr_instance:
        raise HTTPException(status_code=503, detail="Memory Manager not available.")
    if hasattr(memory_mgr_instance, 'clear_session_memory_async'):
        await memory_mgr_instance.clear_session_memory_async(session_id)
        msg = f"Memory for session '{session_id}' cleared (if supported by manager)."
    elif hasattr(memory_mgr_instance, 'storage') and isinstance(memory_mgr_instance.storage, dict):
        if session_id in memory_mgr_instance.storage: memory_mgr_instance.storage[session_id] = []
        logger.warning(f"Simple dict memory clear for session '{session_id}'. Vector DB component needs separate handling.")
        msg = f"Basic dictionary memory for session '{session_id}' cleared."
    else:
        logger.warning(f"Memory manager for session '{session_id}' does not have a recognized clear method.")
        msg = f"No specific clear method found for session '{session_id}' memory."
    return {"message": msg, "session_id": session_id}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir="app")