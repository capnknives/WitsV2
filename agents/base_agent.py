# filepath: c:\WITS\wits_nexus_v2\agents\base_agent.py.new
# The parent of all our awesome agents! Time to define some ground rules ^_^
import logging # Gotta have those sweet, sweet logs
from abc import ABC, abstractmethod # Making sure everyone follows the rules >.>
from typing import Any, Dict, Optional, List # Types make the world go round! \o/
# from core.config import AppConfig # We'll import this when we need it
# from core.llm_interface import LLMInterface # Our connection to the magic AI brain
# from core.memory_manager import MemoryManager # Where memories go to... hopefully not die x.x

class BaseAgent(ABC):
    def __init__(self, agent_name: str, config: Any, llm_interface: Any, memory_manager: Optional[Any] = None, tool_registry: Optional[Any] = None): # Memory is optional, don't panic! >.>
        self.agent_name = agent_name  # What shall we call you? ^_^
        # config has all our settings (hopefully they're good ones lol)
        self.agent_profile = config 
        self.llm = llm_interface  # Our trusty brain! =D
        self.memory = memory_manager  # Where we keep all the important stuff... I think? O.o
        self.tool_registry = tool_registry  # All the cool tools we can use! \o/
        self.logger = logging.getLogger(f"WITS.agents.{self.agent_name}")  # Gotta have those logs, right? =P
        self.logger.info(f"[{self.agent_name}] is alive! Time to do awesome things ^_^")
        
    def _get_agent_specific_config(self, agent_profile_config: Any) -> Dict[str, Any]:
        # This might not be needed anymore but hey, better safe than sorry >.>
        # Just in case someone out there is still using self.agent_config 
        # (I'm looking at you, legacy code! XD)
        if hasattr(agent_profile_config, "llm_model_name"):
            return {"model_name": agent_profile_config.llm_model_name or "unknown"}  # At least we tried! =P
        return {"model_name": "unknown"}  # No model name? This should be fun x.x    @property
    def config(self) -> Any:
        # Just a fancy way to get our config (I love properties! ^_^)
        return self.agent_profile

    @abstractmethod
    async def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """This is where agents do their thing! Every agent needs to implement this =D
        If you forget, Python will yell at you x.x"""
        pass

    def get_name(self) -> str:
        # My name is... (what?)... my name is... (who?)... ^_^
        return self.agent_name

    def get_description(self) -> str:
        """Returns a description of the agent's capabilities."""
        return f"This is the {self.agent_name}. It can perform general tasks."

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Returns a list of tools/capabilities the agent offers, for LLM prompting."""
        # Base agents might not have specific "tools" in the same way an orchestrator does.
        # This can be overridden by subclasses if they expose distinct functionalities.
        return []
