# filepath: c:\WITS\wits_nexus_v2\agents\base_agent.py.new
# agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
# from core.config import AppConfig # If passing full config object
# from core.llm_interface import LLMInterface
# from core.memory_manager import MemoryManager

class BaseAgent(ABC):
    def __init__(self, agent_name: str, config: Any, llm_interface: Any, memory_manager: Any): # Use AppConfig for type hinting later
        self.agent_name = agent_name
        self.config_full = config # Full app config
        self.agent_config = self._get_agent_specific_config(config) # e.g., config.models.scribe
        self.llm = llm_interface
        self.memory = memory_manager
        print(f"[{self.agent_name}] Initialized.")
        
    def _get_agent_specific_config(self, full_config: Any) -> Dict[str, Any]:
        # Helper to extract model name or other agent-specific settings
        
        # Check if this is an AgentProfileConfig (from app/utils.py)
        if hasattr(full_config, "llm_model_name"):
            model_name = full_config.llm_model_name
            return {"model_name": model_name}
            
        # Handle case when it's the full AppConfig
        if hasattr(full_config, "models"):
            agent_model_key = self.agent_name.lower().replace("_agent", "") # e.g. orchestrator_agent -> orchestrator
            model_name = getattr(full_config.models, agent_model_key, full_config.models.default)
            return {"model_name": model_name}
            
        # Default case - use a generic empty config with default model
        return {"model_name": "llama3"}


    @abstractmethod
    async def run(self, user_input_or_task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Main execution method for the agent."""
        pass

    def get_name(self) -> str:
        return self.agent_name
