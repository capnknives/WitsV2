# core/autonomy/tool_example_repository.py
"""
Repository for storing examples of successful tool usage to help AI learn proper tool usage patterns.
This is like a recipe book for our AI - making it easier to learn from past successes! ^_^
"""
import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class ToolExampleUsage(BaseModel):
    """
    A single example of how a tool was used successfully (or unsuccessfully).
    Think of this as a case study for our AI to learn from! \o/
    """
    tool_name: str = Field(..., description="Name of the tool that was used")
    args: Dict[str, Any] = Field(..., description="Arguments that were passed to the tool")
    result: Any = Field(..., description="Result returned by the tool")
    context: str = Field(..., description="Context in which the tool was used (e.g., user goal)")
    explanation: str = Field(..., description="Explanation of why this example was successful or what went wrong")
    success: bool = Field(..., description="Whether the tool execution was successful")
    timestamp: float = Field(default_factory=time.time, description="When this example was recorded")
    
    def get_similarity_key(self) -> str:
        """Generate a simplified representation for similarity comparison."""
        # Simplified representation focusing on the tool name and arg keys (not values)
        arg_keys = sorted(list(self.args.keys()))
        return f"{self.tool_name}:{','.join(arg_keys)}"

class ToolExampleRepository:
    """
    A repository of tool usage examples that can be referenced by the AI.
    Like a library of "how to use this tool" guides! =D
    """
    
    def __init__(self, storage_path: str = "data/tool_examples"):
        """
        Initialize the repository with a storage path.
        
        Args:
            storage_path: Where to store our treasure trove of examples! ^_^
        """
        self.storage_path = storage_path
        self.examples: Dict[str, List[ToolExampleUsage]] = {}  # Tool name -> list of examples
        self.logger = logging.getLogger("WITS.Autonomy.ToolExampleRepository")
        
        # Make sure our storage home exists! \o/
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Load existing examples if available
        self.load()
        
    def add_example(self, example: ToolExampleUsage) -> None:
        """
        Add a new example to the repository.
        
        Args:
            example: The shiny new example to add to our collection! ^_^
        """
        if example.tool_name not in self.examples:
            self.examples[example.tool_name] = []
            
        self.examples[example.tool_name].append(example)
        self.logger.info(f"Added new example for tool '{example.tool_name}' (success: {example.success})")
        
        # Save after each addition to make sure we don't lose our precious examples! >.>
        self.save()
        
    def get_examples(self, tool_name: str) -> List[ToolExampleUsage]:
        """
        Get all examples for a specific tool.
        
        Args:
            tool_name: Which tool's examples to retrieve
            
        Returns:
            List[ToolExampleUsage]: All examples we have for this tool (might be empty!)
        """
        return self.examples.get(tool_name, [])
    
    def get_successful_examples(self, tool_name: str, limit: int = 3) -> List[ToolExampleUsage]:
        """
        Get successful examples for a specific tool, limited to the most recent ones.
        
        Args:
            tool_name: Which tool's successful examples to retrieve
            limit: Maximum number of examples to return
            
        Returns:
            List[ToolExampleUsage]: The most recent successful examples (might be empty!)
        """
        all_examples = self.examples.get(tool_name, [])
        # Filter for successful examples and sort by recency
        successful = [ex for ex in all_examples if ex.success]
        successful.sort(key=lambda ex: ex.timestamp, reverse=True)  # Most recent first
        return successful[:limit]  # Return the most recent ones up to the limit
    
    def find_similar_examples(self, tool_name: str, args: Dict[str, Any]) -> List[ToolExampleUsage]:
        """
        Find examples with similar arguments for a tool.
        This helps our AI learn from past similar scenarios! =D
        
        Args:
            tool_name: Which tool to search examples for
            args: Arguments to compare against
            
        Returns:
            List[ToolExampleUsage]: Similar examples, if any
        """
        if tool_name not in self.examples:
            return []
            
        # Create a simple similarity key from the current args
        arg_keys = sorted(list(args.keys()))
        current_key = f"{tool_name}:{','.join(arg_keys)}"
        
        # Find examples with the same args structure
        similar_examples = []
        for example in self.examples[tool_name]:
            example_key = example.get_similarity_key()
            if example_key == current_key:
                similar_examples.append(example)
        
        # Sort by success (successful examples first) and then by recency
        similar_examples.sort(key=lambda ex: (not ex.success, -ex.timestamp))
        
        return similar_examples
        
    def save(self) -> None:
        """
        Save repository to disk as JSON.
        Let's make sure our hard work persists! \o/
        """
        try:
            # Convert our examples to a JSON-serializable format
            serializable_data = {}
            for tool_name, examples in self.examples.items():
                serializable_data[tool_name] = [ex.model_dump() for ex in examples]
            
            # Save to a JSON file
            file_path = os.path.join(self.storage_path, "tool_examples.json")
            with open(file_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
                
            self.logger.debug(f"Saved {sum(len(examples) for examples in self.examples.values())} examples to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save tool examples: {str(e)}", exc_info=True)
        
    def load(self) -> None:
        """
        Load repository from disk.
        Time to restore our knowledge base! ^_^
        """
        file_path = os.path.join(self.storage_path, "tool_examples.json")
        
        if not os.path.exists(file_path):
            self.logger.info(f"No existing examples found at {file_path}")
            return
            
        try:
            with open(file_path, 'r') as f:
                serializable_data = json.load(f)
                
            # Convert back to ToolExampleUsage objects
            for tool_name, examples_data in serializable_data.items():
                self.examples[tool_name] = [ToolExampleUsage.model_validate(data) for data in examples_data]
                
            example_count = sum(len(examples) for examples in self.examples.values())
            self.logger.info(f"Loaded {example_count} examples from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load tool examples from {file_path}: {str(e)}", exc_info=True)
            # Start with an empty repository if loading fails
            self.examples = {}
