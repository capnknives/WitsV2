# tools/agent_management_tool.py
"""
Agent Management Tool for WITS Nexus v2.
Enables creating, managing, and controlling AI agents.
It's like having your own AI assistant army! \o/
"""

import logging
import asyncio
from typing import ClassVar, Type, Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

from tools.base_tool import ToolResponse
from core.autonomy.enhanced_tool_base import EnhancedTool
from core.autonomy.agent_factory import AgentFactory, AgentConfig, AgentStatus, AgentManager

# --- Agent Creation Tool ---

class CreateAgentArgs(BaseModel):
    """Arguments for creating a new agent."""
    agent_name: str = Field(..., description="Name for the new agent")
    agent_type: str = Field(..., description="Type of agent to create ('chat', 'task', 'assistant', etc.)")
    model_name: Optional[str] = Field(None, description="LLM model to use")
    temperature: float = Field(0.7, description="LLM temperature")
    tools: List[str] = Field(default_factory=list, description="Tools the agent can use")
    description: Optional[str] = Field(None, description="Description of the agent's purpose")
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters for the agent")

class CreateAgentResponse(BaseModel):
    """Response from agent creation."""
    success: bool = Field(..., description="Whether the agent was created successfully")
    agent_id: Optional[str] = Field(None, description="ID of the created agent")
    agent_name: Optional[str] = Field(None, description="Name of the created agent")
    message: str = Field(..., description="Creation status message")
    error: Optional[str] = Field(None, description="Error message if creation failed")

class AgentCreationTool(EnhancedTool):
    """
    Create new AI agents with customized capabilities.
    I can help you build your own AI team! \o/
    """
    
    name: ClassVar[str] = "create_agent"
    description: ClassVar[str] = "Create a new AI agent with customized capabilities."
    args_schema: ClassVar[Type[BaseModel]] = CreateAgentArgs
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        super().__init__()
        self.config = config or {}
        self.logger = logging.getLogger("WITS.AgentCreationTool")
        
        # The actual AgentFactory and AgentManager should be injected by the application
        self.agent_factory = None
        self.agent_manager = None
    
    def set_agent_factory(self, agent_factory: AgentFactory):
        """Set the agent factory for creating agents."""
        self.agent_factory = agent_factory
    
    def set_agent_manager(self, agent_manager: AgentManager):
        """Set the agent manager for managing agents."""
        self.agent_manager = agent_manager
    
    async def _execute_impl(self, args: CreateAgentArgs) -> ToolResponse[CreateAgentResponse]:
        """Implementation of agent creation."""
        try:
            if not self.agent_factory:
                return ToolResponse[CreateAgentResponse](
                    status_code=500,
                    error_message="Agent factory is not available",
                    output=CreateAgentResponse(
                        success=False,
                        message="Agent factory is not available",
                        error="Agent factory is not available"
                    )
                )
                
            if not self.agent_manager:
                return ToolResponse[CreateAgentResponse](
                    status_code=500,
                    error_message="Agent manager is not available",
                    output=CreateAgentResponse(
                        success=False,
                        message="Agent manager is not available",
                        error="Agent manager is not available"
                    )
                )
            
            # Create agent config
            agent_config = AgentConfig(
                agent_name=args.agent_name,
                agent_type=args.agent_type,
                model_name=args.model_name,
                temperature=args.temperature,
                tools=args.tools,
                description=args.description,
                params=args.params
            )
            
            # Create the agent
            agent_id = await self.agent_manager.create_agent(agent_config)
            
            if not agent_id:
                return ToolResponse[CreateAgentResponse](
                    status_code=500,
                    error_message="Failed to create agent for unknown reason",
                    output=CreateAgentResponse(
                        success=False,
                        message="Failed to create agent for unknown reason",
                        error="Failed to create agent for unknown reason"
                    )
                )
            
            return ToolResponse[CreateAgentResponse](
                status_code=200,
                output=CreateAgentResponse(
                    success=True,
                    agent_id=agent_id,
                    agent_name=args.agent_name,
                    message=f"Agent '{args.agent_name}' created successfully with ID: {agent_id}"
                )
            )
            
        except Exception as e:
            self.logger.error(f"Error creating agent: {str(e)}")
            return ToolResponse[CreateAgentResponse](
                status_code=500,
                error_message=f"Error creating agent: {str(e)}",
                output=CreateAgentResponse(
                    success=False,
                    message=f"Error creating agent: {str(e)}",
                    error=f"Error creating agent: {str(e)}"
                )
            )

# --- Agent Management Tool ---

class ManageAgentArgs(BaseModel):
    """Arguments for managing an existing agent."""
    agent_id: str = Field(..., description="ID of the agent to manage")
    action: str = Field(..., description="Action to perform: 'start', 'stop', 'pause', 'resume', 'delete'")
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters for the action")

class ManageAgentResponse(BaseModel):
    """Response from agent management action."""
    success: bool = Field(..., description="Whether the action was successful")
    agent_id: str = Field(..., description="ID of the agent that was managed")
    action: str = Field(..., description="Action that was performed")
    message: str = Field(..., description="Status message")
    error: Optional[str] = Field(None, description="Error message if the action failed")

class AgentManagementTool(EnhancedTool):
    """
    Manage existing AI agents (start, stop, pause, resume, delete).
    I'm the AI agent wrangler! Keeping your AI team in check! \o/
    """
    
    name: ClassVar[str] = "manage_agent"
    description: ClassVar[str] = "Manage an existing AI agent (start, stop, pause, resume, delete)."
    args_schema: ClassVar[Type[BaseModel]] = ManageAgentArgs
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        super().__init__()
        self.config = config or {}
        self.logger = logging.getLogger("WITS.AgentManagementTool")
        
        # The actual AgentManager should be injected by the application
        self.agent_manager = None
    
    def set_agent_manager(self, agent_manager: AgentManager):
        """Set the agent manager for managing agents."""
        self.agent_manager = agent_manager
    
    async def _execute_impl(self, args: ManageAgentArgs) -> ToolResponse[ManageAgentResponse]:
        """Implementation of agent management."""
        try:
            if not self.agent_manager:
                return ToolResponse[ManageAgentResponse](
                    status_code=500,
                    error_message="Agent manager is not available",
                    output=ManageAgentResponse(
                        success=False,
                        agent_id=args.agent_id,
                        action=args.action,
                        message="Agent manager is not available",
                        error="Agent manager is not available"
                    )
                )
            
            # Check if agent exists
            agent_status = await self.agent_manager.get_agent_status(args.agent_id)
            if not agent_status:
                return ToolResponse[ManageAgentResponse](
                    status_code=404,
                    error_message=f"Agent with ID {args.agent_id} not found",
                    output=ManageAgentResponse(
                        success=False,
                        agent_id=args.agent_id,
                        action=args.action,
                        message=f"Agent with ID {args.agent_id} not found",
                        error=f"Agent with ID {args.agent_id} not found"
                    )
                )
            
            result = False
            message = "Unknown action"
            
            # Perform the requested action
            if args.action == 'start':
                result = await self.agent_manager.start_agent(args.agent_id)
                message = f"Agent {args.agent_id} started" if result else f"Failed to start agent {args.agent_id}"
            
            elif args.action == 'stop':
                result = await self.agent_manager.stop_agent(args.agent_id)
                message = f"Agent {args.agent_id} stopped" if result else f"Failed to stop agent {args.agent_id}"
            
            elif args.action == 'pause':
                result = await self.agent_manager.pause_agent(args.agent_id)
                message = f"Agent {args.agent_id} paused" if result else f"Failed to pause agent {args.agent_id}"
            
            elif args.action == 'resume':
                result = await self.agent_manager.resume_agent(args.agent_id)
                message = f"Agent {args.agent_id} resumed" if result else f"Failed to resume agent {args.agent_id}"
            
            elif args.action == 'delete':
                result = await self.agent_manager.delete_agent(args.agent_id)
                message = f"Agent {args.agent_id} deleted" if result else f"Failed to delete agent {args.agent_id}"
            
            else:
                return ToolResponse[ManageAgentResponse](
                    status_code=400,
                    error_message=f"Unknown action: {args.action}",
                    output=ManageAgentResponse(
                        success=False,
                        agent_id=args.agent_id,
                        action=args.action,
                        message=f"Unknown action: {args.action}",
                        error=f"Unknown action: {args.action}"
                    )
                )
            
            if not result:
                return ToolResponse[ManageAgentResponse](
                    status_code=500,
                    error_message=message,
                    output=ManageAgentResponse(
                        success=False,
                        agent_id=args.agent_id,
                        action=args.action,
                        message=message,
                        error=f"Action {args.action} failed"
                    )
                )
            
            return ToolResponse[ManageAgentResponse](
                status_code=200,
                output=ManageAgentResponse(
                    success=True,
                    agent_id=args.agent_id,
                    action=args.action,
                    message=message
                )
            )
            
        except Exception as e:
            self.logger.error(f"Error managing agent: {str(e)}")
            return ToolResponse[ManageAgentResponse](
                status_code=500,
                error_message=f"Error managing agent: {str(e)}",
                output=ManageAgentResponse(
                    success=False,
                    agent_id=args.agent_id,
                    action=args.action,
                    message=f"Error managing agent: {str(e)}",
                    error=f"Error managing agent: {str(e)}"
                )
            )

# --- Agent Query Tool ---

class QueryAgentArgs(BaseModel):
    """Arguments for querying agent information."""
    query_type: str = Field(..., description="Type of query: 'list', 'status', 'info'")
    agent_id: Optional[str] = Field(None, description="ID of specific agent (None for 'list' query)")
    include_inactive: bool = Field(False, description="Whether to include inactive agents in listings")

class QueryAgentResponse(BaseModel):
    """Response from agent query."""
    success: bool = Field(..., description="Whether the query was successful")
    query_type: str = Field(..., description="Type of query performed")
    agents: Optional[List[AgentStatus]] = Field(None, description="List of agent information (for 'list' query)")
    agent_info: Optional[AgentStatus] = Field(None, description="Specific agent information (for 'status' or 'info' query)")
    message: str = Field(..., description="Query result message")
    error: Optional[str] = Field(None, description="Error message if the query failed")

class AgentQueryTool(EnhancedTool):
    """
    Get information about available agents and their status.
    I'm the AI information desk! Know who's who in your AI team! \o/
    """
    
    name: ClassVar[str] = "query_agent"
    description: ClassVar[str] = "Get information about available agents and their status."
    args_schema: ClassVar[Type[BaseModel]] = QueryAgentArgs
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        super().__init__()
        self.config = config or {}
        self.logger = logging.getLogger("WITS.AgentQueryTool")
        
        # The actual AgentManager should be injected by the application
        self.agent_manager = None
    
    def set_agent_manager(self, agent_manager: AgentManager):
        """Set the agent manager for managing agents."""
        self.agent_manager = agent_manager
    
    async def _execute_impl(self, args: QueryAgentArgs) -> ToolResponse[QueryAgentResponse]:
        """Implementation of agent query."""
        try:
            if not self.agent_manager:
                return ToolResponse[QueryAgentResponse](
                    status_code=500,
                    error_message="Agent manager is not available",
                    output=QueryAgentResponse(
                        success=False,
                        query_type=args.query_type,
                        message="Agent manager is not available",
                        error="Agent manager is not available"
                    )
                )
            
            # List all agents
            if args.query_type == 'list':
                agents = await self.agent_manager.list_agents(include_inactive=args.include_inactive)
                
                if agents is None:
                    return ToolResponse[QueryAgentResponse](
                        status_code=500,
                        error_message="Failed to retrieve agent list",
                        output=QueryAgentResponse(
                            success=False,
                            query_type=args.query_type,
                            message="Failed to retrieve agent list",
                            error="Failed to retrieve agent list"
                        )
                    )
                
                return ToolResponse[QueryAgentResponse](
                    status_code=200,
                    output=QueryAgentResponse(
                        success=True,
                        query_type=args.query_type,
                        agents=agents,
                        message=f"Found {len(agents)} agents"
                    )
                )
            
            # Get specific agent status
            elif args.query_type in ['status', 'info']:
                if not args.agent_id:
                    return ToolResponse[QueryAgentResponse](
                        status_code=400,
                        error_message="Agent ID is required for status/info query",
                        output=QueryAgentResponse(
                            success=False,
                            query_type=args.query_type,
                            message="Agent ID is required for status/info query",
                            error="Agent ID is required for status/info query"
                        )
                    )
                
                agent_info = await self.agent_manager.get_agent_status(args.agent_id)
                
                if not agent_info:
                    return ToolResponse[QueryAgentResponse](
                        status_code=404,
                        error_message=f"Agent with ID {args.agent_id} not found",
                        output=QueryAgentResponse(
                            success=False,
                            query_type=args.query_type,
                            message=f"Agent with ID {args.agent_id} not found",
                            error=f"Agent with ID {args.agent_id} not found"
                        )
                    )
                
                return ToolResponse[QueryAgentResponse](
                    status_code=200,
                    output=QueryAgentResponse(
                        success=True,
                        query_type=args.query_type,
                        agent_info=agent_info,
                        message=f"Retrieved information for agent {args.agent_id}"
                    )
                )
            
            else:
                return ToolResponse[QueryAgentResponse](
                    status_code=400,
                    error_message=f"Unknown query type: {args.query_type}",
                    output=QueryAgentResponse(
                        success=False,
                        query_type=args.query_type,
                        message=f"Unknown query type: {args.query_type}",
                        error=f"Unknown query type: {args.query_type}"
                    )
                )
            
        except Exception as e:
            self.logger.error(f"Error querying agents: {str(e)}")
            return ToolResponse[QueryAgentResponse](
                status_code=500,
                error_message=f"Error querying agents: {str(e)}",
                output=QueryAgentResponse(
                    success=False,
                    query_type=args.query_type,
                    message=f"Error querying agents: {str(e)}",
                    error=f"Error querying agents: {str(e)}"
                )
            )

# --- Agent Task Tool ---

class AgentTaskArgs(BaseModel):
    """Arguments for sending tasks to agents."""
    agent_id: str = Field(..., description="ID of the agent to send the task to")
    task: str = Field(..., description="Task description or question to send to the agent")
    priority: Optional[str] = Field("normal", description="Priority level: 'low', 'normal', 'high', 'critical'")
    wait_for_response: bool = Field(False, description="Whether to wait for a response or just queue the task")
    timeout_seconds: Optional[int] = Field(30, description="Timeout in seconds when waiting for response")

class AgentTaskResponse(BaseModel):
    """Response from agent task execution."""
    success: bool = Field(..., description="Whether the task was sent successfully")
    agent_id: str = Field(..., description="ID of the agent the task was sent to")
    task_id: Optional[str] = Field(None, description="ID of the created task")
    response: Optional[str] = Field(None, description="Agent's response if wait_for_response=True")
    message: str = Field(..., description="Status message")
    error: Optional[str] = Field(None, description="Error message if the task failed")

class AgentTaskTool(EnhancedTool):
    """
    Send tasks to agents and optionally get their responses.
    I'm the messenger between you and your AI team! \o/
    """
    
    name: ClassVar[str] = "agent_task"
    description: ClassVar[str] = "Send tasks to agents and optionally get their responses."
    args_schema: ClassVar[Type[BaseModel]] = AgentTaskArgs
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        super().__init__()
        self.config = config or {}
        self.logger = logging.getLogger("WITS.AgentTaskTool")
        
        # The actual AgentManager should be injected by the application
        self.agent_manager = None
    
    def set_agent_manager(self, agent_manager: AgentManager):
        """Set the agent manager for managing agents."""
        self.agent_manager = agent_manager
    
    async def _execute_impl(self, args: AgentTaskArgs) -> ToolResponse[AgentTaskResponse]:
        """Implementation of agent task execution."""
        try:
            if not self.agent_manager:
                return ToolResponse[AgentTaskResponse](
                    status_code=500,
                    error_message="Agent manager is not available",
                    output=AgentTaskResponse(
                        success=False,
                        agent_id=args.agent_id,
                        message="Agent manager is not available",
                        error="Agent manager is not available"
                    )
                )
            
            # Check if agent exists
            agent_status = await self.agent_manager.get_agent_status(args.agent_id)
            if not agent_status:
                return ToolResponse[AgentTaskResponse](
                    status_code=404,
                    error_message=f"Agent with ID {args.agent_id} not found",
                    output=AgentTaskResponse(
                        success=False,
                        agent_id=args.agent_id,
                        message=f"Agent with ID {args.agent_id} not found",
                        error=f"Agent with ID {args.agent_id} not found"
                    )
                )
            
            # Check if agent is running
            if not agent_status.is_running:
                return ToolResponse[AgentTaskResponse](
                    status_code=400,
                    error_message=f"Agent {args.agent_id} is not running",
                    output=AgentTaskResponse(
                        success=False,
                        agent_id=args.agent_id,
                        message=f"Agent {args.agent_id} is not running",
                        error=f"Agent {args.agent_id} is not running"
                    )
                )
            
            # Send the task to the agent
            if args.wait_for_response:
                # Execute and wait for response
                task_id, response = await self.agent_manager.execute_task_sync(
                    args.agent_id, 
                    args.task, 
                    priority=args.priority,
                    timeout_seconds=args.timeout_seconds or 30
                )
                
                if not task_id:
                    return ToolResponse[AgentTaskResponse](
                        status_code=500,
                        error_message="Failed to execute task",
                        output=AgentTaskResponse(
                            success=False,
                            agent_id=args.agent_id,
                            message="Failed to execute task",
                            error="Failed to execute task"
                        )
                    )
                
                return ToolResponse[AgentTaskResponse](
                    status_code=200,
                    output=AgentTaskResponse(
                        success=True,
                        agent_id=args.agent_id,
                        task_id=task_id,
                        response=response,
                        message="Task executed successfully"
                    )
                )
                
            else:
                # Just queue the task
                task_id = await self.agent_manager.queue_task(
                    args.agent_id,
                    args.task,
                    priority=args.priority
                )
                
                if not task_id:
                    return ToolResponse[AgentTaskResponse](
                        status_code=500,
                        error_message="Failed to queue task",
                        output=AgentTaskResponse(
                            success=False,
                            agent_id=args.agent_id,
                            message="Failed to queue task",
                            error="Failed to queue task"
                        )
                    )
                
                return ToolResponse[AgentTaskResponse](
                    status_code=200,
                    output=AgentTaskResponse(
                        success=True,
                        agent_id=args.agent_id,
                        task_id=task_id,
                        message="Task queued successfully"
                    )
                )
            
        except Exception as e:
            self.logger.error(f"Error executing agent task: {str(e)}")
            return ToolResponse[AgentTaskResponse](
                status_code=500,
                error_message=f"Error executing agent task: {str(e)}",
                output=AgentTaskResponse(
                    success=False,
                    agent_id=args.agent_id,
                    message=f"Error executing agent task: {str(e)}",
                    error=f"Error executing agent task: {str(e)}"
                )
            )
