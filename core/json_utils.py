import json
import logging
import re
from typing import Dict, Optional, Union, Any

logger = logging.getLogger("WITS.JsonUtils")

def balance_json_braces(json_str: str) -> str:
    """
    Fix common JSON formatting issues by balancing braces.
    Counts open and close braces and adds missing closing braces if needed.
    
    Args:
        json_str: The JSON string to balance
        
    Returns:
        Balanced JSON string with equal numbers of open and close braces
    """
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    
    # If we have more open braces than closing braces, add the missing closing braces
    if open_braces > close_braces:
        missing_braces = open_braces - close_braces
        logger.warning(f"Fixed JSON by adding {missing_braces} missing closing braces")
        return json_str + ("}" * missing_braces)
    elif close_braces > open_braces:
        # This is unusual, but let's handle it anyway
        logger.warning(f"Found {close_braces - open_braces} more closing braces than opening braces in JSON")
    
    return json_str

def extract_json_from_text(text: str) -> str:
    """
    Attempt to extract a JSON object from text that may contain conversational content.
    
    Args:
        text: The text that may contain a JSON object
        
    Returns:
        The extracted JSON string, or the original text if no JSON-like structure is found
    """
    # Try to find JSON between braces
    json_pattern = r'(\{[\s\S]*\})'
    match = re.search(json_pattern, text)
    if match:
        return match.group(1)
        
    # If no complete JSON object found, look for patterns indicating JSON intent
    if text.lstrip().startswith('{') and not text.rstrip().endswith('}'):
        # We have an opening brace but no closing one - likely truncated JSON
        logger.warning("Found opening JSON brace but no closing one. Attempting to balance.")
        return balance_json_braces(text)
    
    # LLM might be starting with conversational text before JSON
    if '{' in text:
        potential_json_start = text.find('{')
        logger.warning(f"Found conversational text before JSON. Extracting from position {potential_json_start}.")
        return balance_json_braces(text[potential_json_start:])
    
    logger.warning("No JSON-like structure found in text. Returning original text.")
    return text

def generate_default_json_structure() -> dict:
    """
    Generate a default JSON structure for the OrchestratorLLMResponse schema
    when parsing completely fails.
    
    Returns:
        A dictionary with the minimal required fields for OrchestratorLLMResponse
    """
    return {
        "thoughts": "JSON parsing failed. System generated fallback structure.",
        "reasoning": "The LLM response could not be parsed as JSON.",
        "plan": ["Use fallback processing logic", "Try again with stronger JSON enforcement"],
        "command": {
            "name": "fallback_command",
            "args": {"reason": "JSON parsing error in LLM response"}
        }
    }
    
def safe_json_loads(json_str: str, session_id: str = "unknown") -> Optional[dict]:
    """
    Safely parse JSON with error handling and fixes for common issues.
    
    Args:
        json_str: The JSON string to parse
        session_id: Optional session ID for logging purposes
        
    Returns:
        Parsed JSON dictionary or None if parsing completely fails
    """
    if not json_str or not json_str.strip():
        logger.error(f"Empty JSON string received for session '{session_id}'")
        return generate_default_json_structure()
    
    # Fix common issues
    if json_str.startswith("```json"):
        json_str = json_str[len("```json"):].strip()
    elif json_str.startswith("```"):
        json_str = json_str[len("```"):].strip()
        
    if json_str.endswith("```"):
        json_str = json_str[:-len("```")].strip()
    
    # Replace problematic escape sequences
    json_str = json_str.replace("\\'", "'")
    
    # Handle conversational responses
    if not json_str.lstrip().startswith('{'):
        logger.warning(f"Response doesn't start with a JSON object for session '{session_id}', attempting to extract JSON.")
        json_str = extract_json_from_text(json_str)
        
        # If we still don't have valid JSON structure, try to search for key JSON patterns
        if not json_str.lstrip().startswith('{'):
            logger.warning(f"Still no valid JSON structure after extraction for session '{session_id}'. Attempting to build minimal structure.")
            return generate_default_json_structure()
    
    # Balance braces as a last resort
    json_str = balance_json_braces(json_str)
    
    try:
        # Try to parse the JSON
        result = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        # Log detailed error information
        logger.error(f"JSON parsing error for session '{session_id}': {e}")
        logger.error(f"Error at position {e.pos}, line {e.lineno}, column {e.colno}")
        
        # Show snippet around the error
        error_start = max(0, e.pos - 20)
        error_end = min(len(json_str), e.pos + 20)
        logger.error(f"JSON snippet at error: {json_str[error_start:error_end]}")
        
        # Return fallback structure
        logger.error(f"Returning fallback JSON structure for session '{session_id}'")
        return generate_default_json_structure()
