# run.py - Main entry point for WITS-NEXUS v2
import asyncio
import argparse
import os
import sys

# Ensure the project root is in sys.path to allow imports from core, agents, tools
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config import load_app_config, AppConfig
from core.llm_interface import LLMInterface
from core.tool_registry import ToolRegistry
from core.memory_manager import MemoryManager
# from core.ethics import EthicsManager # Placeholder for ethics integration

from agents.orchestrator_agent import OrchestratorAgent

# Import all the tools we've implemented
from tools.calculator_tool import CalculatorTool
from tools.datetime_tool import DateTimeTool
from tools.web_search_tool import WebSearchTool
from tools.file_tools import ReadFileTool, WriteFileTool, ListFilesTool

async def start_wits_cli(config: AppConfig):
    print(f"Initializing {config.app_name} CLI...")

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
    
    list_files_tool = ListFilesTool(config.model_dump())
    tool_registry.register_tool(list_files_tool)
    
    # Web search tool (if internet access is enabled)
    if config.internet_access:
        web_search_tool = WebSearchTool(config.model_dump())
        tool_registry.register_tool(web_search_tool)
        print(f"[WITS CLI] Internet access is enabled. Web search tool registered.")
    
    print(f"[WITS CLI] Registered tools: {[tool.name for tool in tool_registry.get_all_tools()]}")

    # 4. Initialize EthicsManager (placeholder)
    # ethics_manager = EthicsManager(config)

    # 5. Initialize OrchestratorAgent
    orchestrator = OrchestratorAgent(
        agent_name="WITS_Orchestrator", 
        config=config,  # Pass AppConfig
        llm_interface=llm_interface, 
        memory_manager=memory_manager,
        tool_registry=tool_registry
        # ethics_manager=ethics_manager # Pass if orchestrator uses it directly
    )

    print(f"\n{config.app_name} CLI (Orchestrator Model: {config.models.orchestrator}) is ready.")
    print("Type your goal or 'exit' to quit.")

    while True:
        try:
            user_goal = input("WITS v2 >> ").strip()
            if user_goal.lower() in ["exit", "quit", "shutdown"]:
                print(f"Exiting {config.app_name} CLI.")
                break
            if not user_goal:
                continue
            
            print(f"\n[WITS CLI] Processing goal: '{user_goal}'...")
            
            # The orchestrator's run method should handle the ReAct loop internally
            final_result = await orchestrator.run(user_goal) 
            
            print(f"\n[WITS CLI] Final Orchestrator Output:\n------------------------------------\n{final_result}\n------------------------------------")

        except KeyboardInterrupt:
            print(f"\nExiting {config.app_name} CLI due to user interrupt.")
            break
        except Exception as e:
            print(f"[WITS_CLI_ERROR] An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            # Optionally, allow continuing after an error or break
            # break 

def start_wits_web_app(config: AppConfig):
    print("Starting WITS-NEXUS v2 Web App (Placeholder)...")
    # This will be implemented similarly to your v1 app.py, but using the new v2 components
    # For now, this function will just be a placeholder
    print("Web app functionality to be built in app/main.py using Flask or FastAPI.")
    print(f"Configured web host: {config.web_interface.host}, port: {config.web_interface.port}")
    # try:
    #     from app.main import app as flask_app 
    #     web_cfg = config.web_interface
    #     print(f"Attempting to start Flask server on http://{web_cfg.host}:{web_cfg.port}")
    #     flask_app.run(host=web_cfg.host, port=web_cfg.port, debug=web_cfg.debug)
    # except ImportError:
    #     print("[WEB_ERROR] Could not import Flask app from app.main. Ensure it's set up.")
    # except Exception as e:
    #     print(f"[WEB_ERROR] Failed to start web application: {e}")

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
