# core/autonomy/tool_simulator.py
"""
Simulator for testing tool execution in a safe environment before actually calling tools.
Think of it as a practice area where our AI can learn how tools work without breaking anything! \o/
"""
import logging
import json
from typing import Dict, Any, Tuple, List, Optional, Type
from pydantic import BaseModel, ValidationError

from .tool_example_repository import ToolExampleRepository, ToolExampleUsage
from core.tool_registry import ToolRegistry
from tools.base_tool import BaseTool, ToolException

class SimulationResult(BaseModel):
    """Result of a tool simulation including validation feedback and predictions."""
    valid: bool = False
    error_message: Optional[str] = None
    predicted_output: Optional[Dict[str, Any]] = None
    suggested_args: Optional[Dict[str, Any]] = None
    similar_examples: List[Dict[str, Any]] = []
    explanation: str = "No simulation data available"

class ToolSimulator:
    """
    A simulator for testing tool execution before actual runs.
    Our AI's training ground for tool mastery! =D
    """
    
    def __init__(self, example_repository: ToolExampleRepository, tool_registry: ToolRegistry):
        """
        Initialize the tool simulator with repositories.
        
        Args:
            example_repository: Where we keep successful examples
            tool_registry: The catalog of available tools
        """
        self.example_repository = example_repository
        self.tool_registry = tool_registry
        self.logger = logging.getLogger("WITS.Autonomy.ToolSimulator")
        
    async def simulate_tool_execution(self, tool_name: str, args: Dict[str, Any]) -> SimulationResult:
        """
        Simulate executing a tool and return expected output.
        It's like a rehearsal before the big performance! ^_^
        
        Args:
            tool_name: Name of the tool to simulate
            args: Arguments to pass to the tool
            
        Returns:
            SimulationResult: The simulation result with validation and predictions
        """
        # First check if the tool exists
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return SimulationResult(
                valid=False, 
                error_message=f"Tool '{tool_name}' does not exist in the registry.",
                explanation="The tool you're trying to use doesn't exist. Check the available tools and their names."
            )
        
        # Validate the arguments
        is_valid, error_message = await self.validate_tool_args(tool_name, args)
        
        # Find similar examples for prediction
        similar_examples = self.example_repository.find_similar_examples(tool_name, args)
        
        # Prepare result
        result = SimulationResult(
            valid=is_valid,
            error_message=error_message if not is_valid else None,
            similar_examples=[example.model_dump() for example in similar_examples if example.success][:3]  # Top 3 successful examples
        )
        
        # If arguments are invalid, try to suggest corrections
        if not is_valid:
            suggested_args = self.suggest_corrections(tool_name, args, error_message)
            result.suggested_args = suggested_args
            
            # Add explanation based on the error
            result.explanation = f"The arguments provided are invalid: {error_message}. "
            
            if suggested_args:
                result.explanation += "I've suggested some corrections based on successful examples."
            else:
                result.explanation += "Try checking the tool's required parameters and their types."
        
        # If we have similar successful examples, use the most recent one to predict output
        successful_examples = [ex for ex in similar_examples if ex.success]
        if successful_examples:
            result.predicted_output = successful_examples[0].result
            result.explanation = f"Based on {len(successful_examples)} similar successful examples, " + \
                               f"I predict the tool will execute successfully. " + \
                               f"Example explanation: {successful_examples[0].explanation}"
        
        return result
    
    async def validate_tool_args(self, tool_name: str, args: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate tool arguments before execution.
        Let's check if we've got all the ingredients for our recipe! ^_^
        
        Args:
            tool_name: Name of the tool to validate args for
            args: Arguments to validate
            
        Returns:
            Tuple[bool, str]: Validation result and error message if invalid
        """
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return False, f"Tool '{tool_name}' does not exist in the registry."
        
        try:
            # Get the tool's args schema class
            args_schema_class = tool.args_schema
            
            # Validate against the schema
            args_schema_class(**args)
            return True, ""
            
        except ValidationError as e:
            # Format validation errors in a friendlier way
            error_details = []
            for error in e.errors():
                path = ".".join(str(p) for p in error["loc"])
                error_details.append(f"{path}: {error['msg']}")
                
            error_message = "; ".join(error_details)
            self.logger.debug(f"Validation error for tool '{tool_name}': {error_message}")
            return False, error_message
            
        except Exception as e:
            error_message = f"Unexpected error during validation: {str(e)}"
            self.logger.error(error_message)
            return False, error_message
    
    def suggest_corrections(self, tool_name: str, args: Dict[str, Any], error: str) -> Optional[Dict[str, Any]]:
        """
        Suggest corrections to make tool usage work.
        It's like autocorrect, but for tool arguments! \o/
        
        Args:
            tool_name: Name of the tool to suggest corrections for
            args: The problematic arguments
            error: The error message from validation
            
        Returns:
            Optional[Dict[str, Any]]: Suggested corrected arguments, if possible
        """
        # First, get successful examples for this tool
        successful_examples = self.example_repository.get_successful_examples(tool_name)
        if not successful_examples:
            self.logger.debug(f"No successful examples found for tool '{tool_name}' to suggest corrections")
            return None
            
        # Create a copy of the args that we can modify
        corrected_args = args.copy()
        missing_fields = []
        
        # Check for missing fields in the error message
        if "missing" in error.lower() or "required" in error.lower():
            # Extract field names from common Pydantic validation errors
            for part in error.split(";"):
                if "field required" in part:
                    try:
                        field_name = part.split("'")[1]
                        missing_fields.append(field_name)
                    except IndexError:
                        continue
        
        # If we found missing fields, try to fill them from examples
        for field in missing_fields:
            for example in successful_examples:
                if field in example.args:
                    corrected_args[field] = example.args[field]
                    self.logger.debug(f"Suggested value for missing field '{field}' from example")
                    break
                    
        # Check if we actually made any changes
        if corrected_args == args:
            self.logger.debug("Could not generate meaningful corrections")
            return None
            
        return corrected_args
