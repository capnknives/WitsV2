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
import uuid
import importlib

# Application imports
from core.config import AppConfig, load_app_config
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager
from agents.orchestrator_agent import OrchestratorAgent
from agents.wits_control_center_agent import WitsControlCenterAgent

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
    current_session_id = str(uuid.uuid4())
    
    try:
        memory_manager = MemoryManager(config.memory_manager) # Pass MemoryManagerConfig
        await memory_manager.initialize_db()
        
        delegation_targets_map = {}
        main_orchestrator_instance = None
        
        main_orchestrator_profile_name = "book_writing_orchestrator" 

        if config.agent_profiles:
            for agent_name, agent_profile in config.agent_profiles.items():
                if not agent_profile.agent_class:
                    logger.warning(f"Agent class not defined for agent profile: {agent_name}. Skipping initialization.")
                    continue

                if agent_name == main_orchestrator_profile_name:
                    logger.info(f"Profile for main orchestrator '{agent_name}' found. Will be initialized after delegates.")
                    continue

                try:
                    module_name, class_name = agent_profile.agent_class.rsplit('.', 1)
                    AgentClass = getattr(importlib.import_module(module_name), class_name)
                    
                    agent_llm_model = agent_profile.llm_model_name or config.models.default
                    agent_llm_temp = agent_profile.temperature if agent_profile.temperature is not None else config.default_temperature
                    if agent_llm_temp is None: 
                        agent_llm_temp = 0.7 

                    agent_llm_interface = LLMInterface(
                        model_name=agent_llm_model,
                        temperature=float(agent_llm_temp),
                    )

                    # Common parameters for BaseAgent and its derivatives
                    agent_constructor_params = {
                        'agent_name': agent_name,
                        'config': agent_profile, # Pass the specific agent_profile here
                        'llm_interface': agent_llm_interface,
                        # memory_manager is added below based on agent needs or if specified
                    }

                    # Add agent_specific_params from the profile
                    if agent_profile.agent_specific_params:
                        agent_constructor_params.update(agent_profile.agent_specific_params)

                    # Handle OrchestratorAgent's specific needs if this delegate is one
                    if issubclass(AgentClass, OrchestratorAgent):
                        agent_constructor_params['memory_manager'] = memory_manager
                        agent_constructor_params['delegation_targets'] = {} # An orchestrator delegate would have its own (likely empty) targets
                        # max_iterations is expected by OrchestratorAgent, ensure it's present if defined in profile
                        if hasattr(agent_profile, 'max_iterations') and agent_profile.max_iterations is not None:
                            agent_constructor_params['max_iterations'] = agent_profile.max_iterations
                        elif 'max_iterations' not in agent_constructor_params: # Default if not in profile or specific_params
                            agent_constructor_params['max_iterations'] = 5 # A sensible default for delegate orchestrators
                    else:
                        # For non-Orchestrator agents, remove max_iterations if it was added from agent_specific_params
                        agent_constructor_params.pop('max_iterations', None)
                        # Pass memory_manager if the agent's __init__ signature likely expects it (e.g., EngineerAgent)
                        # This is a heuristic. A more robust way would be to inspect __init__ signature.
                        # For now, we know EngineerAgent needs it. BaseAgent makes it optional.
                        if AgentClass.__name__ == "EngineerAgent": # Specific check for EngineerAgent
                            agent_constructor_params['memory_manager'] = memory_manager
                        elif 'memory_manager' in agent_constructor_params and agent_constructor_params['memory_manager'] is True: # If specified in agent_specific_params
                             agent_constructor_params['memory_manager'] = memory_manager
                        # else: memory_manager is not passed by default to BaseAgent derivatives unless specified


                    # Ensure 'tool_registry' is passed if the agent expects it (e.g. EngineerAgent)
                    if AgentClass.__name__ == "EngineerAgent" and 'tool_registry' not in agent_constructor_params:
                         # Assuming tool_registry might be globally available or configured elsewhere if needed by EngineerAgent
                         # For now, if not in specific_params, it won't be passed, which matches EngineerAgent's current __init__
                         pass


                    agent_instance = AgentClass(**agent_constructor_params)
                    delegation_targets_map[agent_name] = agent_instance
                    logger.info(f"Initialized specialized agent: {agent_name} (Class: {class_name})")

                except Exception as e:
                    logger.error(f"Failed to initialize specialized agent {agent_name}: {e}", exc_info=True)
        
        orchestrator_profile_data = config.agent_profiles.get(main_orchestrator_profile_name)
        if orchestrator_profile_data and orchestrator_profile_data.agent_class:
            try:
                module_name, class_name = orchestrator_profile_data.agent_class.rsplit('.', 1)
                OrchestratorClass = getattr(importlib.import_module(module_name), class_name)

                orchestrator_llm_model = orchestrator_profile_data.llm_model_name or config.models.orchestrator 
                orchestrator_llm_temp = orchestrator_profile_data.temperature if orchestrator_profile_data.temperature is not None else config.default_temperature
                if orchestrator_llm_temp is None: 
                    orchestrator_llm_temp = 0.7

                orchestrator_llm_interface = LLMInterface(
                    model_name=orchestrator_llm_model,
                    temperature=float(orchestrator_llm_temp),
                )
                
                max_iter = orchestrator_profile_data.max_iterations

                orchestrator_constructor_params = {
                    'agent_name': main_orchestrator_profile_name,
                    'config': orchestrator_profile_data, 
                    'llm_interface': orchestrator_llm_interface,
                    'memory_manager': memory_manager, 
                    'delegation_targets': delegation_targets_map,
                    'max_iterations': max_iter 
                }
                
                if orchestrator_profile_data.agent_specific_params:
                    for key, value in orchestrator_profile_data.agent_specific_params.items():
                        if key not in orchestrator_constructor_params or key == 'max_iterations': 
                            if value is not None: 
                                orchestrator_constructor_params[key] = value
                
                main_orchestrator_instance = OrchestratorClass(**orchestrator_constructor_params)
                logger.info(f"Main orchestrator '{main_orchestrator_profile_name}' initialized successfully with class {class_name}.")
                logger.info(f"Delegation targets for '{main_orchestrator_profile_name}': {list(delegation_targets_map.keys())}")

            except Exception as e:
                logger.error(f"Failed to initialize main orchestrator '{main_orchestrator_profile_name}': {e}", exc_info=True)
                if not main_orchestrator_instance:
                    logger.critical("CRITICAL: Main orchestrator could not be initialized. WITS CLI cannot function.")
                    return 
        else:
            logger.error(f"Profile for main orchestrator '{main_orchestrator_profile_name}' not found or agent_class missing in config.yaml.")
            logger.critical("CRITICAL: Main orchestrator profile missing. WITS CLI cannot function.")
            return

        # Initialize WitsControlCenterAgent
        # Need to find where WCCA config is in AppConfig. Let's assume it's under a direct key for now
        # and will be confirmed/corrected after checking core/config.py
        wcca_profile_name = "wits_control_center" # Assuming a profile name for WCCA
        wcca_agent_profile = config.agent_profiles.get(wcca_profile_name)

        if wcca_agent_profile and wcca_agent_profile.agent_class:
            try:
                module_name, class_name = wcca_agent_profile.agent_class.rsplit('.', 1)
                WCCAClass = getattr(importlib.import_module(module_name), class_name)

                wcca_llm_model = wcca_agent_profile.llm_model_name or config.models.default
                wcca_llm_temp = wcca_agent_profile.temperature if wcca_agent_profile.temperature is not None else config.default_temperature
                if wcca_llm_temp is None: 
                     wcca_llm_temp = 0.7

                wcca_llm_interface = LLMInterface(
                    model_name=wcca_llm_model,
                    temperature=float(wcca_llm_temp),
                )

                wcca_constructor_params = {
                    'agent_name': wcca_profile_name,
                    'config': wcca_agent_profile, # WCCA (as BaseAgent child) expects its profile as 'config'
                    'llm_interface': wcca_llm_interface,
                    'memory_manager': memory_manager,
                    'orchestrator_delegate': main_orchestrator_instance # This is the key fix for WCCA
                }
                if wcca_agent_profile.agent_specific_params:
                    wcca_constructor_params.update(wcca_agent_profile.agent_specific_params)

                wcca = WCCAClass(**wcca_constructor_params)
                logger.info("WitsControlCenterAgent initialized for CLI.")
            except Exception as e:
                logger.error(f"Failed to initialize WitsControlCenterAgent '{wcca_profile_name}': {e}", exc_info=True)
                logger.critical(f"CRITICAL: WitsControlCenterAgent could not be initialized. WITS CLI cannot function.")
                return
        else:
            logger.error(f"Profile for WitsControlCenterAgent '{wcca_profile_name}' not found or agent_class missing in config.yaml.")
            logger.critical(f"CRITICAL: WitsControlCenterAgent profile missing. WITS CLI cannot function.")
            return
        
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
        logger.info(f"Received interrupt signal for session '{current_session_id}'") # Corrected logging
    except Exception as e:
        logger.exception(f"An unexpected error occurred in WITS CLI loop for session '{current_session_id}': {e}")
        print(f"An unexpected error occurred: {e}")
    finally:
        logger.info(f"WITS CLI shutting down for session '{current_session_id}'") # Corrected logging

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
    # from core.config import load_app_config # This line can be removed or kept as a comment

    try:
        config_path = Path('config.yaml')
        if not config_path.exists():
            # Logger might not be configured yet if this fails early
            print("Error: config.yaml not found") 
            sys.exit(1)
            
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
            
        # Convert the loaded YAML data to an AppConfig object
        config = AppConfig(**config_data) # Pydantic validation happens here
            
        # Setup logging
        configure_console_logging() # Configure logging ASAP
        logger.info("Starting WITS...") # Now logger is configured
        
        # Start the web server if configured
        web_cfg = config.web_interface
        if web_cfg.enabled:
            # Assuming start_wits_web_app is defined and handles web server startup
            # The web app should be started in a non-blocking way, allowing CLI to run concurrently if needed
            logger.info("Web interface is enabled. Starting web app...")
            # start_wits_web_app(config) # Uncomment this line if web app should start here
        else:
            logger.info("Web interface is disabled. Starting CLI mode.")
            # Start the event loop and run CLI
            asyncio.run(start_wits_cli(config))
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        # Use logger if available, otherwise print
        if logging.getLogger('WITS').hasHandlers():
            logger.exception(f"Failed to start WITS: {e}")
        else:
            print(f"Failed to start WITS (logging not fully configured): {e}")
        sys.exit(1)

if __name__ == "__main__":
    main_entry()
