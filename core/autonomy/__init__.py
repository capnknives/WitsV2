# core/autonomy/__init__.py
"""
WITS Nexus v2 Enhanced AI Autonomy Module

This module provides components to enhance AI autonomy in tool usage, 
allowing the AI to learn from examples and improve over time! \o/
"""

from .tool_example_repository import ToolExampleRepository, ToolExampleUsage
from .tool_simulator import ToolSimulator, SimulationResult
from .enhanced_json_handler import EnhancedJSONHandler
from .example_prompt_templates import EnhancedPromptTemplate
from .autonomy_enhancer import AutonomyEnhancer, ToolUsageStats
from .enhanced_tool_base import EnhancedTool
from .mcp_tool_adapter import MCPToolAdapter, MCPToolDefinition, MCPToolManager
from .code_modifier import CodeModifier, CodeModificationResult, PythonCodeAnalyzer
from .agent_factory import AgentFactory, AgentManager, AgentConfig, AgentStatus

__all__ = [
    # Base enhanced autonomy
    'ToolExampleRepository',
    'ToolExampleUsage',
    'ToolSimulator',
    'SimulationResult',
    'EnhancedJSONHandler',
    'EnhancedPromptTemplate',
    'AutonomyEnhancer',
    'ToolUsageStats',
    'EnhancedTool',
    
    # MCP Tool adapter
    'MCPToolAdapter',
    'MCPToolDefinition',
    'MCPToolManager',
    
    # Code modification
    'CodeModifier',
    'CodeModificationResult',
    'PythonCodeAnalyzer',
    
    # Agent management
    'AgentFactory',
    'AgentManager',
    'AgentConfig',
    'AgentStatus',
]
