# core/autonomy/enhanced_json_handler.py
"""
Enhanced JSON handling tools to improve error recovery and formatting for tool calls.
Think of it as a translator that helps the AI communicate better with tools! ^_^
"""
import json
import re
import logging
from typing import Dict, Any, Optional, List, Union, Type
from pydantic import BaseModel, ValidationError

class EnhancedJSONHandler:
    """
    Enhanced JSON handler for improved error recovery and formatting.
    Our AI's personal JSON fixer-upper! \o/
    """
    
    def __init__(self):
        """Initialize the JSON handler with a logger."""
        self.logger = logging.getLogger("WITS.Autonomy.EnhancedJSONHandler")
    
    def extract_tool_call_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract tool call JSON from text that may contain conversational content.
        
        Args:
            text: The text that may contain a JSON object for a tool call
            
        Returns:
            Optional[Dict[str, Any]]: The extracted tool call as a dict, or None if extraction fails
        """
        # Try to find JSON between braces with more specific patterns for tool calls
        tool_call_pattern = r'\{\s*"tool_name"\s*:.*?"args"\s*:.*?\}'
        match = re.search(tool_call_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                # If direct parsing fails, try to fix common issues
                json_text = self._fix_common_json_issues(match.group(0))
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse tool call JSON after fixing common issues")
                    return None
        
        # If no specific tool call pattern found, try more generic JSON extraction
        json_pattern = r'(\{[\s\S]*\})'
        match = re.search(json_pattern, text)
        if match:
            try:
                json_obj = json.loads(match.group(1))
                # Check if it looks like a tool call
                if "tool_name" in json_obj and "args" in json_obj:
                    return json_obj
                else:
                    self.logger.warning("Found JSON but it doesn't match the tool call format")
                    return None
            except json.JSONDecodeError:
                # Try fixing before giving up
                json_text = self._fix_common_json_issues(match.group(1))
                try:
                    json_obj = json.loads(json_text)
                    if "tool_name" in json_obj and "args" in json_obj:
                        return json_obj
                    else:
                        return None
                except json.JSONDecodeError:
                    return None
        
        return None
    
    def fix_tool_args_json(self, json_str: str, schema_model: Type[BaseModel]) -> Union[Dict[str, Any], None]:
        """
        Try to fix malformed JSON for tool arguments based on a schema.
        
        Args:
            json_str: The potentially malformed JSON string
            schema_model: The Pydantic model defining the schema
            
        Returns:
            Union[Dict[str, Any], None]: Fixed JSON as dict, or None if unfixable
        """
        try:
            # First attempt to parse as is
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix common issues
            fixed_json = self._fix_common_json_issues(json_str)
            try:
                json_dict = json.loads(fixed_json)
                
                # Now validate against the schema to fill in defaults
                try:
                    validated = schema_model.model_validate(json_dict)
                    return validated.model_dump()
                except ValidationError:
                    # Return what we have even if it doesn't fully validate
                    return json_dict
            except json.JSONDecodeError:
                # Still failed after fixing
                self.logger.warning("Failed to parse JSON after attempted fixes")
                return None
    
    def _fix_common_json_issues(self, json_str: str) -> str:
        """
        Fix common JSON formatting issues.
        
        Args:
            json_str: The potentially malformed JSON string
            
        Returns:
            str: The JSON string with common issues fixed
        """
        # Remove any markdown code block markers
        json_str = re.sub(r'```json|```', '', json_str).strip()
        
        # Fix unquoted keys
        json_str = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', json_str)
        
        # Fix single quotes used instead of double quotes
        # This is complex because we need to avoid replacing within already properly quoted strings
        # Simplified approach that works for many cases
        json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
        
        # Fix trailing commas in objects and arrays
        json_str = re.sub(r',(\s*})', r'\1', json_str)
        json_str = re.sub(r',(\s*])', r'\1', json_str)
        
        # Balance braces if needed
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
        
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        if open_brackets > close_brackets:
            json_str += ']' * (open_brackets - close_brackets)
        
        return json_str
    
    def suggest_json_corrections(self, json_str: str, schema_model: Type[BaseModel]) -> Dict[str, Any]:
        """
        Analyze and suggest corrections for malformed JSON based on a schema.
        
        Args:
            json_str: The potentially malformed JSON string
            schema_model: The Pydantic model defining the schema
            
        Returns:
            Dict[str, Any]: Corrections and guidance information
        """
        corrections = {
            "fixed_json": None,
            "issues_found": [],
            "suggested_structure": schema_model.model_json_schema(),
            "guidance": "Please check the JSON structure against the suggested schema."
        }
        
        # Try to fix the JSON
        try:
            # First attempt parsing as is
            json.loads(json_str)
            corrections["fixed_json"] = json_str
            corrections["issues_found"].append("No JSON parsing issues detected")
        except json.JSONDecodeError as e:
            # Found parsing issues
            corrections["issues_found"].append(f"JSON parsing error: {str(e)}")
            
            # Try to fix it
            fixed_json = self._fix_common_json_issues(json_str)
            try:
                json.loads(fixed_json)
                corrections["fixed_json"] = fixed_json
                corrections["issues_found"].append("Fixed common JSON formatting issues")
            except json.JSONDecodeError as e2:
                corrections["issues_found"].append(f"Could not fix JSON: {str(e2)}")
        
        # Check against schema if we have valid JSON
        if corrections["fixed_json"]:
            try:
                json_dict = json.loads(corrections["fixed_json"])
                
                # Validate against the schema
                try:
                    schema_model.model_validate(json_dict)
                    corrections["issues_found"].append("JSON is valid against the schema")
                except ValidationError as e:
                    # Add validation errors to issues
                    for error in e.errors():
                        path = ".".join(str(p) for p in error["loc"])
                        corrections["issues_found"].append(f"Schema validation error: {path}: {error['msg']}")
                    
                    # Create a corrected version using schema defaults where possible
                    try:
                        # Create a version with defaults where possible
                        dummy = schema_model.model_construct()
                        defaults = {k: v for k, v in dummy.model_dump().items() 
                                  if k not in json_dict and v is not None}
                        
                        # Add missing fields with defaults
                        for key, value in defaults.items():
                            if key not in json_dict:
                                json_dict[key] = value
                        
                        corrections["fixed_json"] = json.dumps(json_dict)
                        corrections["issues_found"].append("Added missing fields with default values")
                    except Exception as e:
                        corrections["issues_found"].append(f"Error creating defaults: {str(e)}")
            except Exception as e:
                corrections["issues_found"].append(f"Unexpected error in schema validation: {str(e)}")
        
        return corrections
    
    def extract_tool_args(self, text: str, tool_name: str, schema_model: Type[BaseModel]) -> Optional[Dict[str, Any]]:
        """
        Extract and validate tool arguments from text.
        
        Args:
            text: The text that might contain tool arguments
            tool_name: The name of the tool
            schema_model: The Pydantic model for the tool's args schema
            
        Returns:
            Optional[Dict[str, Any]]: Validated tool arguments or None if extraction fails
        """
        # Try to extract a JSON object that might contain tool args
        json_pattern = r'(\{[\s\S]*\})'
        match = re.search(json_pattern, text)
        
        if not match:
            self.logger.warning(f"No JSON-like structure found in text for tool '{tool_name}'")
            return None
            
        json_str = match.group(1)
        
        try:
            # First try direct parsing
            args_dict = json.loads(json_str)
            
            # Check if this is a nested structure with 'args' key
            if isinstance(args_dict, dict) and "args" in args_dict:
                args_dict = args_dict["args"]
                
            # Validate against schema
            try:
                validated_args = schema_model(**args_dict)
                return validated_args.model_dump()
            except ValidationError as e:
                self.logger.warning(f"Validation error for tool '{tool_name}': {str(e)}")
                return None
                
        except json.JSONDecodeError:
            # Try to fix the JSON
            fixed_json = self._fix_common_json_issues(json_str)
            try:
                args_dict = json.loads(fixed_json)
                
                # Check for nested args
                if isinstance(args_dict, dict) and "args" in args_dict:
                    args_dict = args_dict["args"]
                
                # Validate
                try:
                    validated_args = schema_model(**args_dict)
                    return validated_args.model_dump()
                except ValidationError:
                    return None
            except json.JSONDecodeError:
                return None
        
        except Exception as e:
            self.logger.error(f"Unexpected error extracting args for tool '{tool_name}': {str(e)}")
            return None
