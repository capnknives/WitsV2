#!/usr/bin/env python3.10
# run.py - Main entry point for WITS-NEXUS v2

# ============================================================================
# IMPORTANT: This application requires Python 3.10 specifically.
# For best results, run using the startup scripts:
#   - PowerShell: .\start_wits.ps1
#   - Command Prompt: start.bat
# Or activate the conda environment: conda activate faiss_gpu_env2
# ============================================================================

# Check Python version
import sys

if sys.version_info.major != 3 or sys.version_info.minor != 10:
    print("=" * 80)
    print(f"WARNING: This application requires Python 3.10 specifically.")
    print(f"Current Python version: {sys.version_info.major}.{sys.version_info.minor}")
    print("You may encounter errors or reduced functionality.")
    print("Please use the startup scripts or activate the proper conda environment.")
    print("=" * 80)

import asyncio
import argparse
import os
import sys
import time
import logging
from datetime import datetime
import uvicorn
from typing import List, Dict # Added List and Dict
import uuid # Add to imports

# Ensure the project root is in sys.path to allow imports from core, agents, tools
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.debug_utils import setup_logging, DebugInfo, log_debug_info

from core.config import load_app_config, AppConfig
from core.llm_interface import LLMInterface
from core.tool_registry import ToolRegistry
from core.memory_manager import MemoryManager
from agents.wits_control_center_agent import WitsControlCenterAgent
from agents.orchestrator_agent import OrchestratorAgent

# Import specialized agents
from agents.specialized.engineer_agent import EngineerAgent
from agents.specialized.scribe_agent import ScribeAgent
from agents.specialized.researcher_agent import ResearcherAgent
from agents.specialized.analyst_agent import AnalystAgent
from agents.specialized.plotter_agent import PlotterAgent
from agents.specialized.character_agent import CharacterDevelopmentAgent
from agents.specialized.worldbuilder_agent import WorldbuilderAgent
from agents.specialized.prose_agent import ProseGenerationAgent
from agents.specialized.editor_agent import EditorAgent

# Import all the tools we've implemented
from tools.calculator_tool import CalculatorTool
from tools.datetime_tool import DateTimeTool
from tools.web_search_tool import WebSearchTool
from tools.file_tools import ReadFileTool, WriteFileTool, ListFilesTool
from tools.project_file_tools import ProjectFileReaderTool
from tools.git_tools import GitTool

async def start_wits_cli(config: AppConfig):
    logger.info(f"Starting WITS CLI using configuration: {{config.app_name}}")
    
    # 1. Initialize LLMInterface
    default_temp = 0.7 # Define a default float temperature
    if hasattr(config, 'default_temperature') and isinstance(config.default_temperature, (float, int)):
        default_temp = float(config.default_temperature)
    
    llm_interface = LLMInterface(
        model_name=config.models.default,  # Or config.models.orchestrator / planner as appropriate
        temperature=default_temp, # Ensure this is a float
        ollama_url=config.ollama_url,
        request_timeout=config.ollama_request_timeout
    )

    # 2. Initialize ToolRegistry and load tools
    tool_registry = ToolRegistry()
    
    # Register all implemented tools
    # Calculator tool
    calculator = CalculatorTool()
    tool_registry.register_tool(calculator)
    
    # DateTime tool
    datetime_tool = DateTimeTool()
    tool_registry.register_tool(datetime_tool)
    
    # File tools
    read_file_tool = ReadFileTool(config.model_dump())
    tool_registry.register_tool(read_file_tool)
    
    write_file_tool = WriteFileTool(config.model_dump())
    tool_registry.register_tool(write_file_tool)
    
    # Project tools for self-improvement capabilities
    project_file_reader = ProjectFileReaderTool(config.model_dump())
    tool_registry.register_tool(project_file_reader)
    
    git_tool = GitTool(config.model_dump())
    tool_registry.register_tool(git_tool)
    
    list_files_tool = ListFilesTool(config.model_dump())
    tool_registry.register_tool(list_files_tool)
    
    # Web search tool (if internet access is enabled)
    if config.internet_access:
        web_search_tool = WebSearchTool(config.model_dump())
        tool_registry.register_tool(web_search_tool)
        print(f"[WITS CLI] Internet access is enabled. Web search tool registered.")
    
    # Git tool (if git integration is enabled)
    if config.git_integration.enabled:
        git_tool = GitTool(config.model_dump())
        tool_registry.register_tool(git_tool)
        print(f"[WITS CLI] Git integration is enabled. Git tool registered.")
    
    print(f"[WITS CLI] Registered tools: {[tool.name for tool in tool_registry.get_all_tools()]}")

    # 3. Initialize MemoryManager
    memory_manager = MemoryManager(config)

    # 4. Initialize Specialized Agents (if any defined in config)
    
    # Initialize EngineerAgent
    engineer_tool_registry = ToolRegistry()
    engineer_tool_registry.register_tool(project_file_reader)
    engineer_tool_registry.register_tool(git_tool)
    engineer_tool_registry.register_tool(read_file_tool)
    engineer_tool_registry.register_tool(write_file_tool)
    
    engineer_agent = EngineerAgent(
        agent_name="WITS_Engineer",
        config=config,
        llm_interface=llm_interface,
        memory_manager=memory_manager,
        tool_registry=engineer_tool_registry
    )
    
    # Initialize ScribeAgent
    scribe_agent = ScribeAgent(
        agent_name="WITS_Scribe",
        config=config,
        llm_interface=llm_interface,
        memory_manager=memory_manager
    )
    
    # Initialize AnalystAgent
    analyst_agent = AnalystAgent(
        agent_name="WITS_Analyst",
        config=config,
        llm_interface=llm_interface,
        memory_manager=memory_manager
    )
    
    # Initialize ResearcherAgent
    researcher_agent = ResearcherAgent(
        agent_name="WITS_Researcher",
        config=config,
        llm_interface=llm_interface,
        memory_manager=memory_manager
    )
    
    # Initialize additional specialized agents
    plotter_agent = PlotterAgent(
        agent_name="WITS_Book_Plotter",
        config=config,
        llm_interface=llm_interface,
        memory_manager=memory_manager,
        tool_registry=tool_registry
    )
    
    character_dev_agent = CharacterDevelopmentAgent(
        agent_name="WITS_Book_Character_Developer",
        config=config,
        llm_interface=llm_interface,
        memory_manager=memory_manager,
        tool_registry=tool_registry
    )
    
    worldbuilder_agent = WorldbuilderAgent(
        agent_name="WITS_Book_Worldbuilder",
        config=config,
        llm_interface=llm_interface,
        memory_manager=memory_manager,
        tool_registry=tool_registry
    )
    
    prose_generator_agent = ProseGenerationAgent(
        agent_name="WITS_Book_Prose_Generator",
        config=config,
        llm_interface=llm_interface,
        memory_manager=memory_manager,
        tool_registry=tool_registry
    )
    
    editor_agent = EditorAgent(
        agent_name="WITS_Book_Editor",
        config=config,
        llm_interface=llm_interface,
        memory_manager=memory_manager,
        tool_registry=tool_registry
    )
    
    # Register all specialized agents
    specialized_agents = {
        "engineer": engineer_agent,
        "scribe": scribe_agent,
        "researcher": researcher_agent,
        "analyst": analyst_agent,
        "book_plotter": plotter_agent,
        "book_character_dev": character_dev_agent,
        "book_worldbuilder": worldbuilder_agent,
        "book_prose_generator": prose_generator_agent,
        "book_editor": editor_agent,
    }
    
    # 5. Initialize OrchestratorAgent
    orchestrator = OrchestratorAgent(
        agent_name="WITS_Orchestrator", 
        config=config,  # Pass AppConfig
        llm_interface=llm_interface, 
        memory_manager=memory_manager,
        tools=tool_registry,
        delegation_targets=specialized_agents
        # ethics_manager=ethics_manager # Pass if orchestrator uses it directly
    )

    # 6. Initialize WitsControlCenterAgent
    wits_control_center = WitsControlCenterAgent(
        agent_name="WitsControlCenterAgent",
        config=config,
        llm_interface=llm_interface,
        memory_manager=memory_manager,
        orchestrator_delegate=orchestrator, # Pass the orchestrator instance
        specialized_agents=specialized_agents # Pass the dict of specialized agents
    )
    logger.info("WitsControlCenterAgent initialized for CLI.")

    current_session_id = f"cli_session_{uuid.uuid4().hex[:12]}" # Generate a unique session ID
    logger.info(f"CLI Session ID: {current_session_id}")
    current_conversation_history = [] # Initialize history for this session

    print(f"Welcome to {config.app_name} (CLI Mode). Type 'exit' or 'quit' to end.")
    print(f"Session ID: {current_session_id}")

    while True:
        try:
            raw_user_input = input("WITS v2 >> ").strip()
            if raw_user_input.lower() in ["exit", "quit", "shutdown"]:
                logger.info(f"Exiting {config.app_name} CLI.")
                break
            if not raw_user_input:
                continue
            
            # Add user input to the session's conversation history for WCCA context
            # WCCA itself will save this to persistent memory with the session_id
            current_conversation_history.append({"role": "user", "content": raw_user_input})

            logger.info(f"CLI User Input for session '{current_session_id}': {{raw_user_input}}")

            assistant_response_parts = []
            async for data_packet in wits_control_center.run(
                raw_user_input=raw_user_input,
                conversation_history=current_conversation_history[:-1], # Pass history *before* current input
                session_id=current_session_id
            ):
                if data_packet.type == "clarification_request_to_user":
                    print(f"WITS: {data_packet.content}")
                    assistant_response_parts.append(data_packet.content)
                elif data_packet.type == "final_answer":
                    print(f"WITS: {data_packet.content}")
                    assistant_response_parts.append(data_packet.content)
                elif data_packet.type == "info":
                    logger.info(f"WCCA Info for session '{current_session_id}': {data_packet.content}")
                    # Optionally print info to console if desired, e.g., for debugging
                    # print(f"[INFO] {data_packet.content}") 
                elif data_packet.type == "error":
                    logger.error(f"WCCA Error for session '{current_session_id}': {data_packet.content}")
                    print(f"[ERROR] {data_packet.content}")
                    assistant_response_parts.append(f"[ERROR] {data_packet.content}") # Also add error to history
                else:
                    # Handle other stream types if necessary, or log them
                    logger.debug(f"WCCA Stream '{data_packet.type}' for session '{current_session_id}': {data_packet.content}")
            
            # After WCCA stream is complete, if there were assistant responses, add them to history
            if assistant_response_parts:
                full_assistant_response = " ".join(assistant_response_parts)
                current_conversation_history.append({"role": "assistant", "content": full_assistant_response})
            
            # Optional: Trim conversation history if it gets too long for WCCA prompt context
            # MAX_HISTORY_TURNS_FOR_WCCA_PROMPT = 5 # Example limit (5 pairs of user/assistant)
            # if len(current_conversation_history) > MAX_HISTORY_TURNS_FOR_WCCA_PROMPT * 2:
            #     current_conversation_history = current_conversation_history[-(MAX_HISTORY_TURNS_FOR_WCCA_PROMPT*2):]


        except KeyboardInterrupt:
            print("\nExiting WITS CLI (Keyboard Interrupt). Goodbye!")
            break
        except Exception as e:
            logger.exception(f"An unexpected error occurred in WITS CLI loop for session '{current_session_id}': {e}")
            print(f"An unexpected error occurred: {e}")

def start_wits_web_app(config: AppConfig):
    print("Starting WITS-NEXUS v2 Web App...")
    # This will be implemented similarly to your v1 app.py, but using the new v2 components
    print(f"Web app (FastAPI) configured to run via Uvicorn.")
    print(f"Configured web host: {config.web_interface.host}, port: {config.web_interface.port}")
    try:
        # Import the FastAPI app instance from app.main
        # from app.main import app as fastapi_app # No longer needed to import app directly for uvicorn.run string usage
        web_cfg = config.web_interface
        print(f"Attempting to start Uvicorn server for FastAPI app on http://{web_cfg.host}:{web_cfg.port}")
        uvicorn.run("app.main:app", host=web_cfg.host, port=web_cfg.port, reload=web_cfg.debug) # Use uvicorn.run
    except ImportError:
        print("[WEB_ERROR] Could not find app.main or the FastAPI app instance within it. Ensure it's set up correctly.")
    except Exception as e:
        print(f"[WEB_ERROR] Failed to start web application with Uvicorn: {e}")

def main_entry():
    parser = argparse.ArgumentParser(description="WITS-NEXUS v2: Modular AI System with MCP Orchestrator")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--mode", type=str, choices=["cli", "web"], default=None, 
                        help="Force run mode (cli or web). Overrides config.yaml if set.")
    args = parser.parse_args()

    # Load configuration using the Pydantic-based loader
    # The config path is resolved relative to core/config.py for consistency
    config_file_abs_path = os.path.join(PROJECT_ROOT, args.config)
    app_config = load_app_config(config_file_abs_path)

    # Determine run mode
    run_mode = args.mode
    if run_mode is None:  # If not forced by CLI arg, check config
        run_mode = "web" if app_config.web_interface.enabled else "cli"

    if run_mode == "web":
        start_wits_web_app(app_config)
    else:  # Default to CLI
        asyncio.run(start_wits_cli(app_config))

if __name__ == "__main__":
    main_entry()
