# core/autonomy/example_prompt_templates.py
"""
Enhanced prompt templates for the LLM to better use tools with examples and guidance.
These templates help the AI understand how to use tools correctly! ^_^
"""
import json
from typing import Dict, List, Any, Optional
from .tool_example_repository import ToolExampleUsage

class EnhancedPromptTemplate:
    """
    Collection of enhanced prompt templates for AI tool usage.
    Our guidebook for teaching AIs how to use tools properly! \o/
    """
    
    @staticmethod
    def tool_usage_template(tool_name: str, description: str, examples: List[ToolExampleUsage]) -> str:
        """
        Generate a template for tool usage with examples.
        
        Args:
            tool_name: Name of the tool
            description: Description of the tool
            examples: List of example usages
            
        Returns:
            str: A formatted prompt template with examples
        """
        # Start with the tool info
        template = f"### Tool: {tool_name}\n\n"
        template += f"**Description**: {description}\n\n"
        
        # Add examples section if we have examples
        if examples:
            template += "**Usage Examples**:\n\n"
            
            for i, example in enumerate(examples):
                template += f"#### Example {i+1}:\n"
                template += f"**Context**: {example.context}\n\n"
                
                # Format the arguments nicely
                args_str = json.dumps(example.args, indent=2)
                template += f"**Arguments**:\n```json\n{args_str}\n```\n\n"
                
                # Format the result (if it's a dict, pretty-print it)
                if isinstance(example.result, dict):
                    result_str = json.dumps(example.result, indent=2)
                    template += f"**Result**:\n```json\n{result_str}\n```\n\n"
                else:
                    template += f"**Result**: {example.result}\n\n"
                    
                # Add the explanation
                template += f"**Explanation**: {example.explanation}\n\n"
                template += "---\n\n"
                
        # Add a reminder of proper usage format
        template += "**Usage Format**:\n"
        template += "```json\n"
        template += "{\n"
        template += f'  "tool_name": "{tool_name}",\n'
        template += '  "args": {\n    "param1": "value1",\n    "param2": "value2"\n  }\n'
        template += "}\n"
        template += "```\n"
        
        return template
    
    @staticmethod
    def error_correction_template(tool_name: str, error: str, args: Dict[str, Any], 
                                suggested_fix: Optional[Dict[str, Any]]) -> str:
        """
        Generate a template for error correction guidance.
        
        Args:
            tool_name: Name of the tool
            error: Error message
            args: Arguments that caused the error
            suggested_fix: Suggested fixed arguments, if available
            
        Returns:
            str: A formatted prompt template with error guidance
        """
        template = f"### Error Correction for Tool: {tool_name}\n\n"
        template += "**Error Message**:\n```\n" + error + "\n```\n\n"
        
        # Show the problematic arguments
        args_str = json.dumps(args, indent=2)
        template += "**Problematic Arguments**:\n```json\n" + args_str + "\n```\n\n"
        
        # Show the suggested fix if available
        if suggested_fix:
            fix_str = json.dumps(suggested_fix, indent=2)
            template += "**Suggested Fix**:\n```json\n" + fix_str + "\n```\n\n"
            
            # Highlight the differences
            template += "**What Changed**:\n"
            for key in set(list(args.keys()) + list(suggested_fix.keys())):
                if key not in args:
                    template += f"- Added missing key: `{key}` with value: `{suggested_fix[key]}`\n"
                elif key not in suggested_fix:
                    template += f"- Removed problematic key: `{key}`\n"
                elif args[key] != suggested_fix[key]:
                    template += f"- Changed `{key}` from `{args[key]}` to `{suggested_fix[key]}`\n"
        
        # Add general guidance
        template += "\n**General Guidance**:\n"
        template += "1. Make sure all required parameters are provided\n"
        template += "2. Check that parameter types are correct (strings, numbers, booleans, etc.)\n"
        template += "3. Avoid extra or unknown parameters\n"
        
        return template
    
    @staticmethod
    def tools_overview_template(tools_info: List[Dict[str, Any]], 
                              with_examples: bool = False,
                              example_repository = None) -> str:
        """
        Generate an enhanced tools overview with optional examples.
        
        Args:
            tools_info: List of tool information dictionaries
            with_examples: Whether to include examples
            example_repository: Repository to get examples from (required if with_examples=True)
            
        Returns:
            str: A formatted prompt template with tools overview
        """
        template = "# Available Tools\n\n"
        template += "The following tools are available for you to use:\n\n"
        
        for tool_info in tools_info:
            tool_name = tool_info.get("name", "Unknown Tool")
            description = tool_info.get("description", "No description available")
            parameters = tool_info.get("parameters", {})
            
            template += f"## {tool_name}\n\n"
            template += f"**Description**: {description}\n\n"
            
            # Add parameters info
            template += "**Parameters**:\n"
            if "properties" in parameters:
                for param_name, param_info in parameters["properties"].items():
                    param_desc = param_info.get("description", "No description")
                    param_type = param_info.get("type", "any")
                    template += f"- `{param_name}` ({param_type}): {param_desc}\n"
            
            template += "\n"
            
            # Add examples if requested and available
            if with_examples and example_repository:
                examples = example_repository.get_successful_examples(tool_name, limit=1)
                if examples:
                    example = examples[0]
                    template += "**Example Usage**:\n"
                    args_str = json.dumps(example.args, indent=2)
                    template += f"```json\n{{\n  \"tool_name\": \"{tool_name}\",\n  \"args\": {args_str}\n}}\n```\n\n"
            
            template += "---\n\n"
        
        # Add general usage instructions
        template += "## How to Use Tools\n\n"
        template += "To use a tool, provide a JSON object with the following format:\n\n"
        template += "```json\n"
        template += "{\n"
        template += '  "tool_name": "name_of_the_tool",\n'
        template += '  "args": {\n    "param1": "value1",\n    "param2": "value2"\n  }\n'
        template += "}\n"
        template += "```\n\n"
        template += "Make sure to:\n"
        template += "1. Use the exact tool name as listed above\n"
        template += "2. Include all required parameters with correct types\n"
        template += "3. Avoid adding extra parameters that aren't specified\n"
        
        return template
    
    @staticmethod
    def learning_reminder_template() -> str:
        """
        Generate a reminder for the AI about learning from examples.
        
        Returns:
            str: A formatted reminder about learning from examples
        """
        return (
            "# Learning From Examples\n\n"
            "Remember to learn from the examples provided:\n\n"
            "1. **Pattern Recognition**: Study the patterns in successful examples\n"
            "2. **Error Analysis**: Understand what went wrong in failed examples\n"
            "3. **Adaptability**: Apply lessons from similar examples to new situations\n"
            "4. **Self-Improvement**: Learn from your own successful interactions\n\n"
            "When you successfully use a tool, that interaction can become a new example for future reference.\n"
        )
