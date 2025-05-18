# run.py - Main entry point for WITS-NEXUS v2
import asyncio
import argparse
import os
import sys
import time
import logging
from datetime import datetime
import uvicorn # Added import for Uvicorn

# Ensure the project root is in sys.path to allow imports from core, agents, tools
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.debug_utils import setup_logging, DebugInfo, log_debug_info

from core.config import load_app_config, AppConfig
from core.llm_interface import LLMInterface
from core.tool_registry import ToolRegistry
from core.memory_manager import MemoryManager
# from core.ethics import EthicsManager # Placeholder for ethics integration

from agents.orchestrator_agent import OrchestratorAgent

# Import specialized agents
from agents.specialized.engineer_agent import EngineerAgent
from agents.specialized.scribe_agent import ScribeAgent
from agents.specialized.analyst_agent import AnalystAgent
from agents.specialized.researcher_agent import ResearcherAgent

# Import all the tools we've implemented
from tools.calculator_tool import CalculatorTool
from tools.datetime_tool import DateTimeTool
from tools.web_search_tool import WebSearchTool
from tools.file_tools import ReadFileTool, WriteFileTool, ListFilesTool
from tools.project_file_tools import ProjectFileReaderTool
from tools.git_tools import GitTool

async def start_wits_cli(config: AppConfig):
    # Initialize logging
    logger = setup_logging(config.model_dump())
    logger.info(f"Initializing {config.app_name} CLI...")

    # 1. Initialize LLMInterface
    llm_interface = LLMInterface(config)  # Pass the whole AppConfig object

    # 2. Initialize MemoryManager
    memory_manager = MemoryManager(config)  # Pass AppConfig

    # 3. Initialize ToolRegistry and register tools
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

    # 4. Initialize EthicsManager (placeholder)
    # ethics_manager = EthicsManager(config)

    # 5. Initialize specialized agents
    
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
    
    # Register all specialized agents
    specialized_agents = {
        "engineer_agent": engineer_agent,
        "scribe_agent": scribe_agent,
        "analyst_agent": analyst_agent,
        "researcher_agent": researcher_agent
    }
    
    orchestrator = OrchestratorAgent(
        agent_name="WITS_Orchestrator", 
        config=config,  # Pass AppConfig
        llm_interface=llm_interface, 
        memory_manager=memory_manager,
        tool_registry=tool_registry,
        specialized_agents=specialized_agents
        # ethics_manager=ethics_manager # Pass if orchestrator uses it directly
    )

    # Log initialization of specialized agents
    logger.info(f"Specialized agents initialized: {', '.join(specialized_agents.keys())}")
    print(f"[WITS CLI] Specialized agents initialized: {list(specialized_agents.keys())}")
    print(f"[WITS CLI] EngineerAgent initialized with tools: {[tool.name for tool in engineer_tool_registry.get_all_tools()]}")

    print(f"\n{config.app_name} CLI (Orchestrator Model: {config.models.orchestrator}) is ready.")
    print("Type your goal or 'exit' to quit.")

    while True:
        try:
            user_goal = input("WITS v2 >> ").strip()
            if user_goal.lower() in ["exit", "quit", "shutdown"]:
                logger.info(f"Exiting {config.app_name} CLI.")
                break
            if not user_goal:
                continue
            
            logger.info(f"Processing goal: '{user_goal}'")
            start_time = time.time()
            
            # Log debug info for goal start
            debug_info = DebugInfo(
                timestamp=datetime.now().isoformat(),
                component="CLI",
                action="process_goal",
                details={
                    "goal": user_goal,
                    "orchestrator_model": config.models.orchestrator
                },
                duration_ms=0,
                success=True
            )
            log_debug_info(logger, debug_info)
            
            # The orchestrator's run method should handle the ReAct loop internally
            final_result = await orchestrator.run(user_goal)
            
            # Log completion with timing
            completion_time = (time.time() - start_time) * 1000  # ms
            debug_info = DebugInfo(
                timestamp=datetime.now().isoformat(),
                component="CLI",
                action="goal_completed",
                details={
                    "goal": user_goal,
                    "result_length": len(final_result)
                },
                duration_ms=completion_time,
                success=True
            )
            log_debug_info(logger, debug_info)
            
            print(f"\n[WITS CLI] Final Output:\n------------------------------------\n{final_result}\n------------------------------------")
            logger.debug(f"Goal completed in {completion_time:.2f}ms")

        except KeyboardInterrupt:
            logger.warning(f"Exiting {config.app_name} CLI due to user interrupt.")
            break
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Log error details
            debug_info = DebugInfo(
                timestamp=datetime.now().isoformat(),
                component="CLI",
                action="goal_error",
                details={
                    "goal": user_goal,
                    "error_type": type(e).__name__
                },
                duration_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e)
            )
            log_debug_info(logger, debug_info)

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
