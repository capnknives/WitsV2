# core/autonomy/agent_factory.py
"""
Agent Factory for WITS Nexus v2.
Enables creating and managing autonomous agents dynamically.
It's like a character creator for AI agents! \o/
"""

import logging
import importlib
import asyncio
from typing import Dict, Any, Optional, List, Union, Type, Callable, AsyncGenerator
import uuid

from pydantic import BaseModel, Field
from core.memory_manager import MemoryManager
from core.llm_interface import LLMInterface
from core.tool_registry import ToolRegistry
from agents.base_agent import BaseAgent
from core.schemas import StreamData

class AgentConfig(BaseModel):
    """Configuration for a dynamically created agent."""
    agent_name: str = Field(..., description="Name for the agent")
    agent_type: str = Field(..., description="Type of agent to create")
    model_name: Optional[str] = Field(None, description="LLM model to use")
    temperature: float = Field(0.7, description="LLM temperature")
    tools: List[str] = Field(default_factory=list, description="Tools the agent can use")
    description: Optional[str] = Field(None, description="Description of the agent's purpose")
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters for the agent")
    
    class Config:
        """Model configuration."""
        arbitrary_types_allowed = True

class AgentStatus(BaseModel):
    """Status information about an agent."""
    agent_id: str = Field(..., description="Unique ID of the agent")
    agent_name: str = Field(..., description="Name of the agent")
    agent_type: str = Field(..., description="Type of agent")
    is_running: bool = Field(False, description="Whether the agent is currently running")
    model_name: str = Field(..., description="LLM model being used")
    tasks_completed: int = Field(0, description="Number of tasks this agent has completed")
    created_at: str = Field(..., description="When this agent was created")
    last_active: Optional[str] = Field(None, description="When this agent was last active")
    description: Optional[str] = Field(None, description="Description of the agent's purpose")
    
    class Config:
        """Model configuration."""
        arbitrary_types_allowed = True

class AgentFactory:
    """
    Factory for creating and managing agents dynamically.
    Your one-stop shop for all your agent creation needs! =D
    """
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        tool_registry: ToolRegistry,
        default_model_name: str = "llama3",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the agent factory.
        
        Args:
            memory_manager: Memory manager for agent memory
            tool_registry: Tool registry for agent tools
            default_model_name: Default LLM model to use
            config: Additional configuration
        """
        self.memory_manager = memory_manager
        self.tool_registry = tool_registry
        self.default_model_name = default_model_name
        self.config = config or {}
        self.logger = logging.getLogger("WITS.Autonomy.AgentFactory")
        
        # Registry of available agent types
        self.agent_types: Dict[str, Type[BaseAgent]] = {}
        
        # Active agent instances
        self.agents: Dict[str, BaseAgent] = {}
        
        # Agent status tracking
        self.agent_status: Dict[str, AgentStatus] = {}
        
        # Initialize with built-in agent types
        self._register_built_in_agent_types()
    
    def _register_built_in_agent_types(self) -> None:
        """Register built-in agent types from the agents directory."""
        try:
            # Try to import and register common agent types
            from agents.base_orchestrator_agent import BaseOrchestratorAgent
            self.register_agent_type("orchestrator", BaseOrchestratorAgent)
            
            # Import specialized agents if available
            try:
                from agents.specialized.engineer_agent import EngineerAgent
                self.register_agent_type("engineer", EngineerAgent)
            except ImportError:
                self.logger.debug("EngineerAgent not found, skipping registration")
            
            # Try to import other specialized agents
            specialized_agents = [
                ("scribe", "agents.specialized.scribe_agent", "ScribeAgent"),
                ("analyst", "agents.specialized.analyst_agent", "AnalystAgent"),
                ("researcher", "agents.specialized.researcher_agent", "ResearcherAgent")
            ]
            
            for agent_type, module_path, class_name in specialized_agents:
                try:
                    module = importlib.import_module(module_path)
                    agent_class = getattr(module, class_name)
                    self.register_agent_type(agent_type, agent_class)
                except (ImportError, AttributeError):
                    self.logger.debug(f"{class_name} not found, skipping registration")
        
        except Exception as e:
            self.logger.error(f"Error registering built-in agent types: {str(e)}")
    
    def register_agent_type(self, type_name: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register a new agent type.
        
        Args:
            type_name: Name for the agent type
            agent_class: Agent class to register
        """
        if not issubclass(agent_class, BaseAgent):
            raise TypeError(f"Agent class must be a subclass of BaseAgent")
        
        self.agent_types[type_name] = agent_class
        self.logger.info(f"Registered agent type: {type_name}")
    
    def get_available_agent_types(self) -> List[str]:
        """
        Get a list of available agent types.
        
        Returns:
            List[str]: Names of available agent types
        """
        return list(self.agent_types.keys())
    
    def get_agent_class(self, agent_type: str) -> Optional[Type[BaseAgent]]:
        """
        Get the agent class for a given type.
        
        Args:
            agent_type: Type of agent to get
            
        Returns:
            Optional[Type[BaseAgent]]: The agent class or None if not found
        """
        return self.agent_types.get(agent_type)
    
    async def create_agent(
        self,
        config: AgentConfig,
        custom_tools: Optional[List[str]] = None
    ) -> str:
        """
        Create a new agent according to the provided configuration.
        
        Args:
            config: Configuration for the new agent
            custom_tools: Optional list of specific tool names to provide to the agent
            
        Returns:
            str: ID of the created agent
        """
        # Check if the agent type exists
        agent_class = self.agent_types.get(config.agent_type)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {config.agent_type}")
        
        # Create a unique ID for this agent
        agent_id = str(uuid.uuid4())
        
        # Get tools for the agent
        tool_names = config.tools or custom_tools or []
        agent_tools = {}
        for tool_name in tool_names:
            tool = self.tool_registry.get_tool(tool_name)
            if tool:
                agent_tools[tool_name] = tool
        
        # Create LLM interface for the agent
        llm = LLMInterface(
            model_name=config.model_name or self.default_model_name,
            temperature=config.temperature
        )
        
        # Prepare constructor arguments based on the agent class
        constructor_params = {
            'agent_name': config.agent_name,
            'config': self.config,
            'llm_interface': llm,
            'memory_manager': self.memory_manager,
        }
        
        # Add tool_registry if the agent class accepts it
        if 'tool_registry' in agent_class.__init__.__annotations__:
            constructor_params['tool_registry'] = self.tool_registry
        
        # Add any additional parameters from the config
        constructor_params.update(config.params or {})
        
        try:
            # Create the agent instance
            agent = agent_class(**constructor_params)
            
            # Store the agent
            self.agents[agent_id] = agent
            
            # Record agent status
            from datetime import datetime
            self.agent_status[agent_id] = AgentStatus(
                agent_id=agent_id,
                agent_name=config.agent_name,
                agent_type=config.agent_type,
                is_running=False,
                model_name=config.model_name or self.default_model_name,
                tasks_completed=0,
                created_at=datetime.now().isoformat(),
                description=config.description
            )
            
            self.logger.info(f"Created agent '{config.agent_name}' of type '{config.agent_type}' with ID {agent_id}")
            return agent_id
        
        except Exception as e:
            self.logger.error(f"Error creating agent '{config.agent_name}': {str(e)}")
            raise
    
    async def run_agent(
        self,
        agent_id: str,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[StreamData, None]:
        """
        Run an agent with a specific goal.
        
        Args:
            agent_id: ID of the agent to run
            goal: Goal for the agent to achieve
            context: Additional context for the agent
            
        Yields:
            AsyncGenerator[StreamData, None]: Stream of data from the agent execution
        """
        # Check if the agent exists
        agent = self.agents.get(agent_id)
        if not agent:
            yield StreamData(
                type="error",
                content=f"Agent with ID {agent_id} not found"
            )
            return
        
        # Update status
        status = self.agent_status.get(agent_id)
        if status:
            from datetime import datetime
            status.is_running = True
            status.last_active = datetime.now().isoformat()
        
        try:
            # Run the agent
            self.logger.info(f"Running agent {agent_id} with goal: {goal}")
            context = context or {}
            
            # Stream results from the agent
            async for data in agent.run(goal, context):
                yield data
                
            # Update status after successful run
            if status:
                status.tasks_completed += 1
                status.is_running = False
                from datetime import datetime
                status.last_active = datetime.now().isoformat()
        
        except Exception as e:
            self.logger.error(f"Error running agent {agent_id}: {str(e)}")
            yield StreamData(
                type="error",
                content=f"Error running agent: {str(e)}"
            )
            
            # Update status on error
            if status:
                status.is_running = False
                from datetime import datetime
                status.last_active = datetime.now().isoformat()
    
    def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        """
        Get the status of an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Optional[AgentStatus]: Status of the agent or None if not found
        """
        return self.agent_status.get(agent_id)
    
    def list_agents(self) -> List[AgentStatus]:
        """
        List all currently managed agents.
        
        Returns:
            List[AgentStatus]: Status of all managed agents
        """
        return list(self.agent_status.values())
    
    async def delete_agent(self, agent_id: str) -> bool:
        """
        Delete an agent.
        
        Args:
            agent_id: ID of the agent to delete
            
        Returns:
            bool: Whether the deletion was successful
        """
        # Check if the agent exists
        if agent_id not in self.agents:
            return False
        
        # Check if the agent is running
        status = self.agent_status.get(agent_id)
        if status and status.is_running:
            return False  # Can't delete a running agent
        
        # Remove the agent
        del self.agents[agent_id]
        if agent_id in self.agent_status:
            del self.agent_status[agent_id]
        
        self.logger.info(f"Deleted agent with ID {agent_id}")
        return True

class AgentManagerResponse(BaseModel):
    """Response model for agent management operations."""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Message about the operation")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data about the operation")

class AgentManager:
    """
    High-level manager for agent operations.
    Think of it as an AI resource manager! ^_^
    """
    
    def __init__(self, agent_factory: AgentFactory):
        """
        Initialize the agent manager.
        
        Args:
            agent_factory: Factory for creating agents
        """
        self.agent_factory = agent_factory
        self.logger = logging.getLogger("WITS.Autonomy.AgentManager")
    
    async def create_agent(self, agent_config: Dict[str, Any]) -> AgentManagerResponse:
        """
        Create a new agent.
        
        Args:
            agent_config: Configuration for the new agent
            
        Returns:
            AgentManagerResponse: Result of the operation
        """
        try:
            # Convert dict to AgentConfig
            config = AgentConfig(**agent_config)
            
            # Create the agent
            agent_id = await self.agent_factory.create_agent(config)
            
            return AgentManagerResponse(
                success=True,
                message=f"Created agent '{config.agent_name}' with ID {agent_id}",
                data={"agent_id": agent_id}
            )
        
        except Exception as e:
            self.logger.error(f"Error creating agent: {str(e)}")
            return AgentManagerResponse(
                success=False,
                message=f"Failed to create agent: {str(e)}",
                data=None
            )
    
    async def run_agent_task(
        self,
        agent_id: str,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Union[StreamData, AgentManagerResponse], None]:
        """
        Run a task with a specific agent.
        
        Args:
            agent_id: ID of the agent to run
            goal: Goal for the agent to achieve
            context: Additional context for the agent
            
        Yields:
            AsyncGenerator[Union[StreamData, AgentManagerResponse], None]: Stream of data or response
        """
        try:
            # Run the agent
            async for data in self.agent_factory.run_agent(agent_id, goal, context):
                yield data
        
        except Exception as e:
            self.logger.error(f"Error running agent {agent_id}: {str(e)}")
            yield AgentManagerResponse(
                success=False,
                message=f"Error running agent: {str(e)}",
                data=None
            )
    
    async def list_agents(self) -> AgentManagerResponse:
        """
        List all agents.
        
        Returns:
            AgentManagerResponse: Result with list of agents
        """
        try:
            agents = self.agent_factory.list_agents()
            return AgentManagerResponse(
                success=True,
                message=f"Found {len(agents)} agents",
                data={"agents": [agent.model_dump() for agent in agents]}
            )
        
        except Exception as e:
            self.logger.error(f"Error listing agents: {str(e)}")
            return AgentManagerResponse(
                success=False,
                message=f"Error listing agents: {str(e)}",
                data=None
            )
    
    async def get_agent_status(self, agent_id: str) -> AgentManagerResponse:
        """
        Get the status of an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            AgentManagerResponse: Result with agent status
        """
        try:
            status = self.agent_factory.get_agent_status(agent_id)
            if not status:
                return AgentManagerResponse(
                    success=False,
                    message=f"Agent with ID {agent_id} not found",
                    data=None
                )
            
            return AgentManagerResponse(
                success=True,
                message=f"Agent status retrieved",
                data={"status": status.model_dump()}
            )
        
        except Exception as e:
            self.logger.error(f"Error getting agent status: {str(e)}")
            return AgentManagerResponse(
                success=False,
                message=f"Error getting agent status: {str(e)}",
                data=None
            )
    
    async def delete_agent(self, agent_id: str) -> AgentManagerResponse:
        """
        Delete an agent.
        
        Args:
            agent_id: ID of the agent to delete
            
        Returns:
            AgentManagerResponse: Result of the operation
        """
        try:
            success = await self.agent_factory.delete_agent(agent_id)
            if not success:
                return AgentManagerResponse(
                    success=False,
                    message=f"Failed to delete agent with ID {agent_id}",
                    data=None
                )
            
            return AgentManagerResponse(
                success=True,
                message=f"Deleted agent with ID {agent_id}",
                data=None
            )
        
        except Exception as e:
            self.logger.error(f"Error deleting agent: {str(e)}")
            return AgentManagerResponse(
                success=False,
                message=f"Error deleting agent: {str(e)}",
                data=None
            )
    
    async def get_available_agent_types(self) -> AgentManagerResponse:
        """
        Get a list of available agent types.
        
        Returns:
            AgentManagerResponse: Result with list of agent types
        """
        try:
            agent_types = self.agent_factory.get_available_agent_types()
            return AgentManagerResponse(
                success=True,
                message=f"Found {len(agent_types)} agent types",
                data={"agent_types": agent_types}
            )
        
        except Exception as e:
            self.logger.error(f"Error getting agent types: {str(e)}")
            return AgentManagerResponse(
                success=False,
                message=f"Error getting agent types: {str(e)}",
                data=None
            )
