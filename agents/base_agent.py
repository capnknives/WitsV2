# filepath: c:\WITS\wits_nexus_v2\agents\base_agent.py.new
# agents/base_agent.py
import logging # Added for logger
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List # Added List for get_available_tools
# from core.config import AppConfig # If passing full config object
# from core.llm_interface import LLMInterface
# from core.memory_manager import MemoryManager

class BaseAgent(ABC):
    def __init__(self, agent_name: str, config: Any, llm_interface: Any, memory_manager: Optional[Any] = None, tool_registry: Optional[Any] = None): # Modified memory_manager to be optional
        self.agent_name = agent_name
        # config is expected to be an AgentProfileConfig instance
        self.agent_profile = config 
        self.llm = llm_interface
        self.memory = memory_manager
        self.tool_registry = tool_registry # Added tool_registry initialization
        self.logger = logging.getLogger(f"WITS.agents.{self.agent_name}") # Added logger initialization
        self.logger.info(f"[{self.agent_name}] Initialized.") # Changed print to self.logger.info
        
    def _get_agent_specific_config(self, agent_profile_config: Any) -> Dict[str, Any]:
        # This method might be redundant if agent_profile is directly used.
        # Kept for now for compatibility if any subclass was using self.agent_config
        if hasattr(agent_profile_config, "llm_model_name"):
            return {"model_name": agent_profile_config.llm_model_name or "unknown"}
        return {"model_name": "unknown"}

    @property
    def config(self) -> Any:
        # Provides access to the agent's specific profile configuration.
        return self.agent_profile

    @abstractmethod
    async def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> Any: # Changed return type to Any for more flexibility with AsyncGenerator
        """Main execution method for the agent."""
        pass

    def get_name(self) -> str:
        return self.agent_name

    def get_description(self) -> str:
        """Returns a description of the agent's capabilities."""
        return f"This is the {self.agent_name}. It can perform general tasks."

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Returns a list of tools/capabilities the agent offers, for LLM prompting."""
        # Base agents might not have specific "tools" in the same way an orchestrator does.
        # This can be overridden by subclasses if they expose distinct functionalities.
        return []
