"""Main FastAPI application server for WITS NEXUS v2."""
from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form # Added File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse # Added StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, AsyncGenerator # Added AsyncGenerator
import json # For streaming data
import os
import uuid
import aiofiles
from datetime import datetime

# Import the agent service
from app.services.agent_service import agent_service_instance
from core.config import load_app_config, AppConfig 
from app.utils import get_or_create_agent_for_session, update_llm_for_session_agent # Import helpers

# Import agent and core components (adjust paths as necessary)
from agents.orchestrator_agent import OrchestratorAgent, StreamData # Import StreamData
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager
from core.tool_registry import ToolRegistry
from core.faiss_utils import create_gpu_index # Added import
import numpy as np # Added import

# Import individual tools
from tools.calculator_tool import CalculatorTool
from tools.datetime_tool import DateTimeTool
from tools.web_search_tool import WebSearchTool
from tools.file_tools import ReadFileTool, WriteFileTool, ListFilesTool
from tools.project_file_tools import ProjectFileReaderTool
from tools.git_tools import GitTool

# Import debug router
from app.routes.debug_routes import debug_router

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# Load application configuration
app_config: AppConfig = load_app_config()

# --- Pydantic Models for API Responses ---
class AgentProfileResponse(BaseModel):
    name: str
    display_name: str
    description: str

class SessionAgentSelectionRequest(BaseModel):
    session_id: str
    agent_profile_name: str

class SessionAgentSelectionResponse(BaseModel):
    message: str
    session_id: str
    selected_agent: str

class LLMParams(BaseModel):
    model: str
    temperature: float

class SessionLLMParamsResponse(BaseModel):
    session_id: str
    active_agent_profile: Optional[str] = None
    model: str
    temperature: float
    available_models: List[str]

class UpdateLLMParamsRequest(BaseModel):
    session_id: str
    model: str
    temperature: float

class UpdateLLMParamsResponse(BaseModel):
    message: str
    session_id: str
    updated_params: LLMParams

class ChatRequest(BaseModel):
    session_id: str
    message: str  # Changed from 'goal' to 'message' for better conversational UX
    goal: Optional[str] = None  # Keep for backward compatibility
    agent_profile_name: Optional[str] = None # Added agent_profile_name
    # context: Optional[Dict[str, Any]] = None # If frontend sends additional context

# --- Pydantic Models for Memory Operations ---
class MemorySearchRequest(BaseModel):
    session_id: str
    query: str
    k: Optional[int] = 5 # Number of results to return

class MemorySearchResultItem(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float
    # Add other relevant fields from MemorySegment if needed, e.g., id, type, source, timestamp

class MemorySearchResponse(BaseModel):
    session_id: str
    query: str
    results: List[MemorySearchResultItem]

class MemoryClearRequest(BaseModel):
    session_id: str

class MemoryClearResponse(BaseModel):
    session_id: str
    message: str

# --- Global instances for core components (to be initialized) ---
# These should ideally be managed with a dependency injection system or a more structured setup.
# For now, global instances for simplicity, initialized on startup.

global_memory_manager: Optional[MemoryManager] = None
global_tool_registry: Optional[ToolRegistry] = None

@app.on_event("startup")
async def startup_event():
    global global_memory_manager, global_tool_registry
    
    # Initialize Memory Manager
    global_memory_manager = MemoryManager(config=app_config) 
    await global_memory_manager.initialize_db() 
    print(f"MemoryManager initialized and database loaded. Path: {app_config.memory_manager.memory_file_path}")

    # Initialize Tool Registry and load tools
    global_tool_registry = ToolRegistry(config=app_config)
    
    # Instantiate and register tools
    calculator = CalculatorTool()
    global_tool_registry.register_tool(calculator)
    
    datetime_tool = DateTimeTool()
    global_tool_registry.register_tool(datetime_tool)
    
    # File tools require config for output_directory
    read_file_tool = ReadFileTool(config=app_config.model_dump()) # Pass config as dict
    global_tool_registry.register_tool(read_file_tool)
    
    write_file_tool = WriteFileTool(config=app_config.model_dump()) # Pass config as dict
    global_tool_registry.register_tool(write_file_tool)

    list_files_tool = ListFilesTool(config=app_config.model_dump()) # Pass config as dict
    global_tool_registry.register_tool(list_files_tool)

    # ProjectFileReaderTool might need config if it has specific settings
    project_file_reader = ProjectFileReaderTool(config=app_config.model_dump()) # Pass config as dict
    global_tool_registry.register_tool(project_file_reader)

    # GitTool might need config
    git_tool = GitTool(config=app_config.model_dump()) # Pass config as dict
    global_tool_registry.register_tool(git_tool)
    
    if app_config.internet_access:
        web_search_tool = WebSearchTool(config=app_config.model_dump()) # Pass config as dict
        global_tool_registry.register_tool(web_search_tool)
        print(f"[FastAPI App] Internet access is enabled. Web search tool registered.")
    
    print(f"[FastAPI App] Registered tools: {[tool.name for tool in global_tool_registry.get_all_tools()]}")

    # Ensure session agents dictionary is initialized (if not already)
    # This is handled by app.utils.SESSION_AGENTS
    print("FastAPI application startup complete.")

# Include the debug router
app.include_router(debug_router)

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/debug", response_class=HTMLResponse)
async def debug_dashboard(request: Request):
    return templates.TemplateResponse("debug_dashboard.html", {"request": request})

@app.get("/api/agents", response_model=List[AgentProfileResponse])
async def get_agents():
    """
    Fetches available agent profiles.
    """
    try:
        profiles = agent_service_instance.get_agent_profiles()
        return profiles
    except Exception as e:
        # Log the exception e
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agent profiles: {str(e)}")

# Placeholder for active sessions and their selected agents/parameters
# In a real app, this would be a more robust session management system (e.g., Redis, database)
active_sessions: Dict[str, Dict[str, Any]] = {}

@app.post("/api/session/agent", response_model=SessionAgentSelectionResponse)
async def select_agent_for_session(payload: SessionAgentSelectionRequest):
    """
    Sets the active agent for a given session.
    """
    session_id = payload.session_id
    agent_name = payload.agent_profile_name

    if not session_id or not agent_name:
        raise HTTPException(status_code=400, detail="session_id and agent_profile_name are required.")

    # Validate if agent_name is a valid profile
    available_profiles = agent_service_instance.get_agent_profiles()
    if not any(p['name'] == agent_name for p in available_profiles):
        raise HTTPException(status_code=404, detail=f"Agent profile '{agent_name}' not found.")

    if session_id not in active_sessions:
        active_sessions[session_id] = {}
    
    active_sessions[session_id]['selected_agent_profile'] = agent_name
    active_sessions[session_id].pop('model', None) # Clear specific model/temp when agent changes
    active_sessions[session_id].pop('temperature', None)
    
    # When an agent is selected, we might want to load its default LLM params
    # For now, just confirm selection. LLM params will be fetched by a separate call.
    
    print(f"Agent '{agent_name}' selected for session '{session_id}'") # Server log
    return SessionAgentSelectionResponse(
        message=f"Agent '{agent_name}' selected for session.",
        session_id=session_id,
        selected_agent=agent_name
    )

@app.get("/api/session/parameters", response_model=SessionLLMParamsResponse)
async def get_session_llm_parameters(session_id: str):
    """
    Gets the current LLM model and temperature for the session's active agent.
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required.")

    session_data = active_sessions.get(session_id, {})
    selected_agent_profile_name = session_data.get('selected_agent_profile')
    
    # Default LLM settings from global config
    model_name = app_config.models.default
    temperature = app_config.default_temperature if app_config.default_temperature is not None else 0.7

    # If an agent is selected, try to get its specific LLM config
    if selected_agent_profile_name and app_config.agent_profiles:
        agent_profile_config = app_config.agent_profiles.get(selected_agent_profile_name)
        if agent_profile_config:
            model_name = agent_profile_config.llm_model_name or model_name
            temperature = agent_profile_config.temperature if agent_profile_config.temperature is not None else temperature

    # Override with session-specific settings if they exist (user changed them in UI)
    model_name = session_data.get('model', model_name)
    temperature = session_data.get('temperature', temperature)

    available_models = list(app_config.models.model_dump().values())
    available_models = [m for m in available_models if m is not None]
    if model_name not in available_models:
        available_models.append(model_name)
    available_models = sorted(list(set(available_models)))

    return SessionLLMParamsResponse(
        session_id=session_id,
        active_agent_profile=selected_agent_profile_name,
        model=model_name,
        temperature=temperature,
        available_models=available_models
    )

@app.post("/api/config/parameters", response_model=UpdateLLMParamsResponse)
async def update_llm_parameters_for_session(payload: UpdateLLMParamsRequest):
    """
    Updates LLM parameters (model, temperature) for the current session.
    These settings will override the agent's default config for this session.
    """
    session_id = payload.session_id
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required.")

    if session_id not in active_sessions:
        active_sessions[session_id] = {} 

    active_sessions[session_id]['model'] = payload.model
    active_sessions[session_id]['temperature'] = payload.temperature
    
    # Also update the actual LLM instance for the agent if it exists
    selected_agent_profile = active_sessions[session_id].get('selected_agent_profile')
    if selected_agent_profile and global_memory_manager and global_tool_registry: # Ensure globals are initialized
        # This function from app.utils will update or create the LLM instance
        # and associate it with the agent in its session store.
        await update_llm_for_session_agent(
            session_id=session_id,
            profile_name=selected_agent_profile,
            app_config=app_config,
            new_model_name=payload.model,
            new_temperature=payload.temperature
        )
        print(f"LLM instance for agent '{selected_agent_profile}' in session '{session_id}' updated.")
    
    print(f"LLM params stored for session '{session_id}': Model={payload.model}, Temp={payload.temperature}")

    return UpdateLLMParamsResponse(
        message="LLM parameters updated successfully for the session.",
        session_id=session_id,
        updated_params=LLMParams(model=payload.model, temperature=payload.temperature)
    )

# --- Memory Endpoints ---
@app.get("/api/memory/search", response_model=MemorySearchResponse)
async def search_memory(session_id: str, query: str, k: Optional[int] = 5):
    if not session_id or not query:
        raise HTTPException(status_code=400, detail="session_id and query are required.")
    if not global_memory_manager:
        raise HTTPException(status_code=500, detail="MemoryManager not initialized.")

    try:
        # Assuming MemoryManager has a method like search_memory_by_text
        # search_memory_by_text should return a list of tuples or objects
        # (score, segment_id, segment_content, segment_metadata)
        # We need to adapt this to MemorySearchResultItem
        
        # Placeholder for actual search logic - this needs to be implemented in MemoryManager
        # For now, let's assume search_memory_by_text exists and returns appropriate data
        # search_results = await global_memory_manager.search_memory_by_text(
        # query_text=query, 
        # session_id=session_id, # If memory is session-specific
        # k=k
        # )

        # Simulating a call to a method that might exist or need to be created in MemoryManager
        # This method would handle embedding the query and searching the FAISS index.
        
        # For now, let's use the existing _search_similar and adapt
        query_embedding = global_memory_manager._generate_embedding(query)
        if query_embedding is None:
            raise HTTPException(status_code=500, detail="Could not generate query embedding.")

        similar_segments_info = global_memory_manager._search_similar(query_embedding, k=k)
        
        results: List[MemorySearchResultItem] = []
        for score, segment_id in similar_segments_info:
            # Retrieve the full segment from MemoryManager.segments
            segment = next((s for s in global_memory_manager.segments if s.id == segment_id), None)
            if segment:
                # Ensure content is a string. It might be a dict if tool_output etc.
                content_text = ""
                if segment.content.text:
                    content_text = segment.content.text
                elif segment.content.tool_output:
                    content_text = str(segment.content.tool_output) # Convert to string if not already
                elif segment.content.tool_name: # Fallback if no direct text/output
                    content_text = f"Tool: {segment.content.tool_name}"


                results.append(MemorySearchResultItem(
                    content=content_text,
                    metadata=segment.metadata,
                    score=score
                ))
        
        return MemorySearchResponse(session_id=session_id, query=query, results=results)

    except Exception as e:
        # Log the exception e
        print(f"Error during memory search: {e}") # Replace with proper logging
        raise HTTPException(status_code=500, detail=f"Error during memory search: {str(e)}")


@app.post("/api/memory/clear", response_model=MemoryClearResponse)
async def clear_memory(payload: MemoryClearRequest):
    session_id = payload.session_id
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required.")
    if not global_memory_manager:
        raise HTTPException(status_code=500, detail="MemoryManager not initialized.")

    try:
        # Assuming MemoryManager has a method like clear_session_memory
        # await global_memory_manager.clear_session_memory(session_id=session_id)
        
        # For now, if memory is not strictly session-partitioned in the MemoryManager's
        # main 'segments' list or FAISS index in a way that allows easy clearing per session,
        # this might be a more complex operation.
        # The current MemoryManager seems to have a global segment list.
        # A true session-specific clear would require filtering segments by session_id
        # and removing them from self.segments and the FAISS index.

        # Placeholder: If we want to clear ALL memory (not session specific as per current MM design)
        # This is a drastic action and likely not what's intended by session_id based clearing.
        # global_memory_manager.segments.clear()
        # global_memory_manager.id_to_faiss_idx.clear()
        # global_memory_manager.faiss_idx_to_id.clear()
        # if global_memory_manager.index:
        #     global_memory_manager.index.reset() # Clears the FAISS index
        # await global_memory_manager._save_to_disk() # Save the empty state

        # A more realistic approach for session-based clearing if segments have session_id in metadata:
        segments_to_keep = []
        ids_to_remove_from_faiss = []
        for segment in global_memory_manager.segments:
            if segment.metadata.get("session_id") == session_id:
                ids_to_remove_from_faiss.append(segment.id)
            else:
                segments_to_keep.append(segment)
        
        if not ids_to_remove_from_faiss:
            return MemoryClearResponse(session_id=session_id, message="No memory segments found for this session to clear.")

        global_memory_manager.segments = segments_to_keep
        
        # Removing from FAISS is more complex as it requires knowing the FAISS internal indices.
        # FAISS typically supports removing by ID if the index is an IndexIDMap.
        # The current setup uses a direct index and manual ID mapping (id_to_faiss_idx, faiss_idx_to_id).
        # A simple approach is to rebuild the index without the removed IDs.
        # This is inefficient for frequent clears but safer with the current mapping.

        if global_memory_manager.index and ids_to_remove_from_faiss:
            # Rebuild FAISS index without the cleared segments
            # This is a simplified approach. A more robust solution would involve
            # using FAISS's remove_ids if an IndexIDMap is used, or careful management
            # of the existing index and mappings.
            
            # For now, let's log that this part is complex.
            # A full rebuild:
            if global_memory_manager.vector_dim and global_memory_manager.embedding_model:
                global_memory_manager.index = create_gpu_index(
                    global_memory_manager.vector_dim, 
                    global_memory_manager.RES, # Access RES via instance
                    global_memory_manager.CUDA_DEVICE # Access CUDA_DEVICE via instance
                )
                global_memory_manager.id_to_faiss_idx.clear()
                global_memory_manager.faiss_idx_to_id.clear()
                for seg in global_memory_manager.segments:
                    if seg.embedding:
                        emb_np = np.array(seg.embedding, dtype=np.float32)
                        global_memory_manager._add_to_index(emb_np, seg.id)
                message = f"Memory cleared for session {session_id}. FAISS index rebuilt."
            else:
                # If index cannot be rebuilt (e.g. model not loaded), just clear segments list
                message = f"Memory segments cleared for session {session_id}. FAISS index may still contain old data if not rebuilt."


        await global_memory_manager._save_to_disk() # Save changes

        return MemoryClearResponse(session_id=session_id, message=message)
    except Exception as e:
        # Log the exception e
        print(f"Error clearing memory: {e}") # Replace with proper logging
        raise HTTPException(status_code=500, detail=f"Error clearing memory: {str(e)}")

# --- File Upload Endpoint ---
class FileUploadResponse(BaseModel):
    message: str
    file_path: Optional[str] = None
    file_name: str
    file_size: int

@app.post("/api/upload", response_model=FileUploadResponse)
async def upload_file(request: Request, session_id: str = Form(...), file: UploadFile = File(...)):
    if not app_config.web_interface.enable_file_uploads:
        raise HTTPException(status_code=403, detail="File uploads are disabled.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    # Ensure user_files directory exists (relative to project root)
    # The AppConfig validator for output_directory might not cover this specific path.
    # file_tool_base_path is already resolved to an absolute path by AppConfig.
    user_files_dir = app_config.file_tool_base_path 
    
    # It's good practice to ensure the directory exists, though AppConfig might handle some.
    # For user-specific subdirectories, create them here.
    session_specific_upload_dir = os.path.join(user_files_dir, session_id)
    os.makedirs(session_specific_upload_dir, exist_ok=True)

    # Sanitize filename to prevent directory traversal issues
    # Using werkzeug.utils.secure_filename is a good practice if available, 
    # but a simple custom sanitizer for now.
    original_filename = file.filename if file.filename else "unnamed_file"
    safe_filename = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in original_filename)
    if not safe_filename: # Handle cases where filename becomes empty after sanitization
        safe_filename = f"uploaded_file_{uuid.uuid4().hex[:8]}"

    file_size = 0
    file_path = ""
    try:
        # Save the uploaded file to the designated directory
        # Using aiofiles for async file handling
        file_path = os.path.join(session_specific_upload_dir, safe_filename)
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
            file_size = len(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Ensure the full path is returned for client use if needed, relative to some base or as an identifier
    # For security, don't return absolute server paths directly if not necessary.
    # Here, we return a path that the agent might use (e.g., relative to a known workspace for tools)
    # For now, let's assume the agent/tools know how to handle paths based on 'user_files_dir'
    
    # The agent will likely receive the 'safe_filename' and assume it's in its designated user files area.
    # If the agent needs a more specific path, this response might need adjustment.
    # For example, if tools operate relative to 'user_files_dir'.
    
    # Let's return a path that might be useful for the agent, e.g., session_id/safe_filename
    # This assumes tools know to look in user_files_dir/session_id/safe_filename
    agent_usable_path = os.path.join(session_id, safe_filename)

    return FileUploadResponse(
        message=f"File '{original_filename}' uploaded successfully as '{safe_filename}'.",
        file_path=agent_usable_path, # Path for agent to potentially use
        file_name=safe_filename,
        file_size=file_size
    )

# --- Chat Streaming Endpoint ---
@app.post("/api/chat_stream") # No response_model here for StreamingResponse
async def chat_stream(payload: ChatRequest):
    """
    Handles chat requests, streams responses from the agent.
    """
    if not global_memory_manager or not global_tool_registry:
        # This check should ideally be more robust, perhaps using FastAPI dependencies
        # to ensure services are available.
        print("Error: Core services (MemoryManager or ToolRegistry) not available.")
        raise HTTPException(status_code=503, detail="Core services not available. Please try again shortly.")

    session_id = payload.session_id
    
    # Support both 'message' and 'goal' fields for backward compatibility
    user_message = payload.message if hasattr(payload, 'message') and payload.message is not None else payload.goal
    
    # agent_profile_name will be None if not sent by client, get_or_create_agent_for_session handles defaults.
    agent_profile_name = payload.agent_profile_name 

    if not session_id or not user_message:
        raise HTTPException(status_code=400, detail="session_id and message are required.")
        
    try:
        # Get or create the agent for the session.
        # This utility function handles LLM setup based on session/profile config.
        agent = await get_or_create_agent_for_session(
            session_id=session_id,
            profile_name=agent_profile_name if agent_profile_name else "", 
            app_config=app_config,
            global_memory_manager=global_memory_manager,
            global_tool_registry=global_tool_registry
        )
    except Exception as e:
        print(f"Error creating/getting agent for session {session_id}: {e}")
        # Log the full error server-side (e.g., using app.logger if configured)
        raise HTTPException(status_code=500, detail=f"Failed to initialize agent: {str(e)}")

    async def stream_generator() -> AsyncGenerator[str, None]:
        try:
            # The agent's run method is an async generator yielding StreamData objects
            # Support both older user_goal param and newer user_input_or_task param
            if hasattr(agent, "run") and callable(agent.run):
                run_params = {}
                # Check if run method accepts user_input_or_task parameter
                if hasattr(agent.run, "__code__") and "user_input_or_task" in agent.run.__code__.co_varnames:
                    run_params["user_input_or_task"] = user_message
                else:
                    # Fall back to older parameter name
                    run_params["user_goal"] = user_message
                
                async for data_chunk in agent.run(**run_params, context=None):
                    yield f"{data_chunk.model_dump_json()}\\n"
            else:
                error_stream_data = StreamData(type="error", content="Agent has no run method.", error_details="Implementation error")
                yield f"{error_stream_data.model_dump_json()}\\n"
        except Exception as e:
            error_detail = f"Error during agent execution: {str(e)}"
            # Log the full error server-side
            print(f"ERROR in chat_stream for session {session_id}, agent {agent.agent_name if agent else 'Unknown'}: {error_detail}")
            # Yield a final error message to the client, using the StreamData model
            error_stream_data = StreamData(type="error", content="Agent execution failed.", error_details=str(e))
            yield f"{error_stream_data.model_dump_json()}\\n"
        finally:
            print(f"Stream finished for session {session_id}, message: '{user_message[:50]}...'")
            # Perform any cleanup for this stream if necessary

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

@app.get("/api/health")
async def health_check():
    """
    Simple health check endpoint to verify the API is running.
    """
    return {
        "status": "healthy", 
        "message": "WITS-NEXUS v2 API is active",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "memory_manager": global_memory_manager is not None,
            "tool_registry": global_tool_registry is not None,
            "agent_service": agent_service_instance is not None
        }
    }

# Ensure StreamData is imported at the top of app/main.py
# from agents.orchestrator_agent import StreamData (this will be added separately if not present)