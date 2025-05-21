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

__all__ = [
    'ToolExampleRepository',
    'ToolExampleUsage',
    'ToolSimulator',
    'SimulationResult',
    'EnhancedJSONHandler',
    'EnhancedPromptTemplate',
    'AutonomyEnhancer',
    'ToolUsageStats',
    'EnhancedTool',
]
