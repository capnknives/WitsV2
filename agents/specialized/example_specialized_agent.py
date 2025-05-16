# agents/specialized/example_specialized_agent.py
from typing import Any, Dict
from agents.base_agent import BaseAgent

class ExampleSpecializedAgent(BaseAgent):
    """An example of a specialized agent."""
    def __init__(self, agent_name: str, config: Dict[str, Any], llm_interface: Any, memory: Any):
        super().__init__(agent_name, config, llm_interface, memory)

    async def run(self, task_description: str, context: Dict[str, Any] = None) -> str:
        # Specialized logic for this agent
        print(f"[{self.agent_name}] Received task: {task_description}")
        # Example: Call LLM for its specific purpose
        # response = await self.llm.generate(prompt=f"Complete this specialized task: {task_description}")
        return f"[{self.agent_name}] Task '{task_description[:50]}...' processed. (Placeholder response)"
