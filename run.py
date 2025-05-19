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

# Core imports
import asyncio
import logging
import os
import signal
from datetime import datetime
import uuid
import json
from pathlib import Path
import yaml

# Application imports
from core.config import AppConfig
from core.debug_utils import setup_logging
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager
from core.tool_registry import ToolRegistry
from core.schemas import StreamData

# Agent imports
from agents.wits_control_center_agent import WitsControlCenterAgent
from agents.orchestrator_agent import OrchestratorAgent
from agents.specialized.editor_agent import EditorAgent

# Tool imports
from tools.calculator_tool import CalculatorTool
from tools.datetime_tool import DateTimeTool
from tools.web_search_tool import WebSearchTool
from tools.file_tools import ReadFileTool, WriteFileTool, ListFilesTool
from tools.project_file_tools import ProjectFileReaderTool
from tools.git_tools import GitTool

# Global logger
logger = logging.getLogger('WITS')

def configure_console_logging():
    """Configure console logging with enhanced visibility for debugging."""
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set root logger to DEBUG to catch all messages
    
    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler with INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create detailed formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(console_handler)
    
    # Configure specific loggers for key components
    component_loggers = [
        "WITS.agents",
        "WITS.OrchestratorAgent",
        "WITS.LLMInterface",
        "WITS.Tools",
        "WITS.ProseGenerationAgent"
    ]
    
    for logger_name in component_loggers:
        component_logger = logging.getLogger(logger_name)
        component_logger.setLevel(logging.DEBUG)  # Set component loggers to DEBUG
        # Don't add handler - they'll inherit from root
    
    # Log startup message to verify logging is working
    logger.info("Logging system initialized - Console handler configured")

async def start_wits_cli(config: AppConfig):
    """Start WITS in CLI mode."""
    logger.info(f"Starting WITS CLI using configuration: {config.app_name}")

    # Initialize session ID at the start for proper error handling scope
    current_session_id = str(uuid.uuid4())
    
    try:
        # Initialize LLM Interface with default model and temperature
        llm_interface = LLMInterface(
            model_name=config.models.default,
            temperature=config.default_temperature if config.default_temperature is not None else 0.7
        )
        
        # Initialize Memory Manager
        memory_manager = MemoryManager(config)
        await memory_manager.initialize_db()
        
        # Initialize Tool Registry
        tool_registry = ToolRegistry()
        
        # Initialize specialized agents (empty for now)
        specialized_agents = {}
        
        # Initialize OrchestratorAgent with empty delegation targets for now
        orchestrator_agent = OrchestratorAgent(
            agent_name="OrchestratorAgent",
            config=config,
            llm_interface=llm_interface,
            memory_manager=memory_manager,
            tool_registry=tool_registry,
            delegation_targets={}  # Empty dict for now
        )
        
        # Initialize the control center agent with all dependencies
        wcca = WitsControlCenterAgent(
            agent_name="WitsControlCenterAgent",
            config=config,
            llm_interface=llm_interface,
            memory_manager=memory_manager,
            orchestrator_delegate=orchestrator_agent,
            specialized_agents=specialized_agents
        )
        
        logger.info("WitsControlCenterAgent initialized for CLI.")
        logger.info(f"CLI Session ID: {current_session_id}")
        
        while True:
            # Get user input
            raw_user_input = input("\nUser: ").strip()
            
            # Check for exit command
            if raw_user_input.lower() in ['exit', 'quit', 'q']:
                logger.info(f"Exiting {config.app_name} CLI.")
                break
                
            logger.info(f"CLI User Input for session '{current_session_id}': {raw_user_input}")
            
            # Process input through WCCA using the run method
            try:
                async for data_packet in wcca.run(raw_user_input, [], current_session_id):
                    # Handle different types of responses
                    if data_packet.type == "info":
                        logger.info(f"WCCA Info for session '{current_session_id}': {data_packet.content}")
                        print(f"System: {data_packet.content}")
                    elif data_packet.type == "error":
                        logger.error(f"WCCA Error for session '{current_session_id}': {data_packet.content}")
                        print(f"Error: {data_packet.content}")
                    else:
                        logger.debug(f"WCCA Stream '{data_packet.type}' for session '{current_session_id}': {data_packet.content}")
                        print(data_packet.content)
                    
            except Exception as e:
                logger.exception(f"An error occurred while processing input for session '{current_session_id}': {e}")
                print(f"Error: {str(e)}")
                
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Shutting down gracefully...")
        logger.info(f"Received interrupt signal for session '{current_session_id}'")
    except Exception as e:
        logger.exception(f"An unexpected error occurred in WITS CLI loop for session '{current_session_id}': {e}")
        print(f"An unexpected error occurred: {e}")
    finally:
        # Just log shutdown, no need to call close()
        logger.info(f"WITS CLI shutting down for session '{current_session_id}'")

def start_wits_web_app(config: AppConfig):
    """Start WITS in web mode."""
    try:
        web_cfg = config.web_interface
        if not web_cfg.enabled:
            logger.warning("Web interface is disabled in config. Please enable it to use web mode.")
            return
        
        logger.info(f"Starting WITS Web Application on {web_cfg.host}:{web_cfg.port}")
        # Web app is started through FastAPI/uvicorn in app/main.py
        # The app will be configured through the AppConfig passed to it
        
    except Exception as e:
        logger.exception("Failed to start web interface")
        raise

def main_entry():
    """Main entry point for WITS"""
    import sys
    import yaml
    from pathlib import Path
    
    try:
        config_path = Path('config.yaml')
        if not config_path.exists():
            print("Error: config.yaml not found")
            sys.exit(1)
            
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            config = AppConfig(**config_dict)
        
        # Setup logging
        configure_console_logging()
        logger.info("Starting WITS...")
        
        # Start the web server if configured
        web_cfg = config.web_interface
        if web_cfg.enabled:
            start_wits_web_app(config)
        else:
            # Start the event loop and run CLI
            asyncio.run(start_wits_cli(config))
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.exception(f"Failed to start WITS: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main_entry()
