import json
import logging

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
    
def safe_json_loads(json_str: str, session_id: str = "unknown") -> dict:
    """
    Safely parse JSON with error handling and fixes for common issues.
    
    Args:
        json_str: The JSON string to parse
        session_id: Optional session ID for logging purposes
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        json.JSONDecodeError if parsing fails after attempted fixes
    """
    # Fix common issues
    if json_str.startswith("```json"):
        json_str = json_str[len("```json"):].strip()
    elif json_str.startswith("```"):
        json_str = json_str[len("```"):].strip()
        
    if json_str.endswith("```"):
        json_str = json_str[:-len("```")].strip()
    
    # Replace problematic escape sequences
    json_str = json_str.replace("\\'", "'")
    
    # Balance braces
    json_str = balance_json_braces(json_str)
    
    try:
        # Try to parse the JSON
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Log detailed error information
        logger.error(f"JSON parsing error for session '{session_id}': {e}")
        logger.error(f"Error at position {e.pos}, line {e.lineno}, column {e.colno}")
        
        # Show snippet around the error
        error_start = max(0, e.pos - 20)
        error_end = min(len(json_str), e.pos + 20)
        logger.error(f"JSON snippet at error: {json_str[error_start:error_end]}")
        
        # Re-raise the exception
        raise
