# filepath: c:\WITS\wits_nexus_v2\agents\base_agent.py.new
# agents/base_agent.py
import logging # Added for logger
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List # Added List for get_available_tools
# from core.config import AppConfig # If passing full config object
# from core.llm_interface import LLMInterface
# from core.memory_manager import MemoryManager

class BaseAgent(ABC):
    def __init__(self, agent_name: str, config: Any, llm_interface: Any, memory_manager: Any, tool_registry: Optional[Any] = None): # Added tool_registry
        self.agent_name = agent_name
        self.config_full = config # Full app config
        self.agent_config = self._get_agent_specific_config(config) # e.g., config.models.scribe
        self.llm = llm_interface
        self.memory = memory_manager
        self.tool_registry = tool_registry # Added tool_registry initialization
        self.logger = logging.getLogger(f"WITS.agents.{self.agent_name}") # Added logger initialization
        self.logger.info(f"[{self.agent_name}] Initialized.") # Changed print to self.logger.info
        
    def _get_agent_specific_config(self, full_config: Any) -> Dict[str, Any]:
        # Helper to extract model name or other agent-specific settings
        
        # Check if this is an AgentProfileConfig (from app/utils.py)
        if hasattr(full_config, "llm_model_name"):
            model_name = full_config.llm_model_name
            return {"model_name": model_name}
            
        # Handle case when it\'s the full AppConfig
        if hasattr(full_config, "models"):
            agent_model_key = self.agent_name.lower().replace("_agent", "") # e.g. orchestrator_agent -> orchestrator
            model_name = getattr(full_config.models, agent_model_key, full_config.models.default)
            return {"model_name": model_name}
            
        # Default case - use a generic empty config with default model
        return {"model_name": "llama3"}    @abstractmethod
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
