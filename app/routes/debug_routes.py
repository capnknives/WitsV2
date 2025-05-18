# app/routes/debug_routes.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Any, Optional
import logging
import os
import json
from datetime import datetime, timedelta
import time

debug_router = APIRouter(prefix="/api/debug", tags=["debug"])

# In-memory cache for debug data
debug_data_cache = {
    "metrics": {
        "active_sessions": 0,
        "llm_calls": 0,
        "tool_calls": 0,
        "avg_response_time": 0,
        "memory_segments": 0,
        "last_updated": 0
    },
    "performance": {
        "llm_interface": [],
        "memory_manager": [],
        "tools": [],
        "agents": [],
        "last_updated": 0
    },
    "logs": {}  # Changed to an empty dict to support per-log-type caching
}

# Cache expiration time in seconds
CACHE_EXPIRATION = 15

# Process and parse log entries from files
def parse_log_files(log_type: Optional[str] = None): # Added log_type parameter
    """Parse the log files to extract structured data, optionally filtering by log_type."""
    logs = []
    log_dir = os.path.join(os.getcwd(), "logs")
    
    if not os.path.exists(log_dir):
        return []
    
    # Read from both debug and error logs, combine and sort later if necessary, or handle specific files
    log_files_to_check = []
    if os.path.exists(os.path.join(log_dir, 'wits_debug.log')):
        log_files_to_check.append(os.path.join(log_dir, 'wits_debug.log'))
    if os.path.exists(os.path.join(log_dir, 'wits_error.log')):
        log_files_to_check.append(os.path.join(log_dir, 'wits_error.log'))

    if not log_files_to_check:
        return []
    
    raw_log_entries = []
    for log_file_path in log_files_to_check:
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                # Read last N lines for performance, e.g., 1000 lines per file
                raw_log_entries.extend(f.readlines()[-1000:]) 
        except Exception as e:
            print(f"Error reading log file {log_file_path}: {e}")
            continue
    
    # Sort by timestamp if combining from multiple files (assuming consistent timestamp format)
    # For simplicity, current parsing processes them sequentially. A more robust solution
    # would parse timestamps and sort all entries if true chronological order is needed.

    for line in raw_log_entries: # Process the collected lines
        try:
            parts = line.strip().split(' | ')
            if len(parts) < 3:
                continue
            
            timestamp_str, level_str, component_msg = parts[0], parts[1], ' | '.join(parts[2:])
            component_parts = component_msg.split(' - ', 1)
            component = component_parts[0].strip() if len(component_parts) > 0 else "unknown"
            message = component_parts[1].strip() if len(component_parts) > 1 else component_msg.strip()
            level = level_str.strip().upper()

            # Filter by log_type if specified
            if log_type:
                if log_type == "error" and level != "ERROR":
                    continue
                if log_type == "warning" and level != "WARNING":
                    continue
                # For "all", no level filtering here, or if log_type is not "error" or "warning"
            
            logs.append({
                "timestamp": timestamp_str.strip(),
                "level": level,
                "component": component,
                "message": message
            })
        except Exception as e:
            print(f"Error parsing log line: '{line[:100]}...': {e}") # Log the problematic line
            continue
    
    # Sort logs by timestamp (assuming standard format that sorts chronologically as strings)
    # A more robust sort would parse datetime objects.
    try:
        logs.sort(key=lambda x: x["timestamp"], reverse=True)
    except Exception as e:
        print(f"Could not sort logs by timestamp: {e}")

    return logs[:200] # Return the last 200 combined and filtered log entries

# Extract performance data from logs
def extract_performance_data():
    """Extract performance data from logs for charts and tables."""
    log_dir = os.path.join(os.getcwd(), "logs")
    performance_file = os.path.join(log_dir, "wits_performance.log")
    
    # Initialize performance data structure
    performance_data = {
        "llm_interface": [],
        "memory_manager": [],
        "tools": [],
        "agents": []
    }
    
    if not os.path.exists(performance_file):
        return performance_data
    
    # Process the performance log file
    with open(performance_file, 'r') as f:
        for line in f.readlines()[-500:]:  # Last 500 entries
            try:
                if "Performance:" in line:
                    # Extract timestamp
                    timestamp_part = line.split(' | ')[0]
                    timestamp = timestamp_part.strip()
                    
                    # Extract component and action
                    if "LLMInterface" in line or "WITS.LLM" in line:
                        component = "LLMInterface"
                        category = "llm_interface"
                    elif "MemoryManager" in line or "WITS.Memory" in line:
                        component = "MemoryManager"
                        category = "memory_manager"
                    elif "Tool." in line or "WITS.Tools" in line:
                        component = line.split("Tool.")[1].split(".")[0] if "Tool." in line else "Tool"
                        category = "tools"
                    elif "Agent" in line or "WITS.Agents" in line:
                        component = line.split("Agent.")[1].split(".")[0] if "Agent." in line else "Agent"
                        category = "agents"
                    else:
                        continue
                    
                    # Extract action, duration, and success (simplified parsing)
                    action = "execute" if "execute" in line else "operation"
                    duration = float(line.split("took ")[1].split(" seconds")[0]) * 1000 if "took " in line else 0
                    success = "failed" not in line.lower()
                    
                    # Add to the appropriate category
                    if len(performance_data[category]) < 100:  # Limit entries per category
                        performance_data[category].append({
                            "timestamp": timestamp,
                            "component": component,
                            "action": action,
                            "duration_ms": duration,
                            "success": success,
                            "details": {}
                        })
            except Exception as e:
                print(f"Error parsing performance data: {e}")
                continue
    
    # Sort entries by timestamp (newest first)
    for category in performance_data:
        performance_data[category].sort(key=lambda x: x["timestamp"], reverse=True)
        performance_data[category] = performance_data[category][:50]  # Keep only the latest 50
    
    # Add last_updated timestamp to the performance data itself
    performance_data["last_updated"] = int(time.time())
    return performance_data

# Calculate system metrics
def calculate_metrics():
    """Calculate system-wide metrics."""
    metrics = {
        "active_sessions": 0,
        "llm_calls": 0,
        "tool_calls": 0,
        "avg_response_time": 0,
        "memory_segments": 0
    }
    
    # Extract performance data for calculations
    performance_data = extract_performance_data()
    
    # Count LLM calls and calculate average response time
    llm_calls = performance_data.get("llm_interface", [])
    if llm_calls:
        metrics["llm_calls"] = len(llm_calls)
        metrics["avg_response_time"] = int(sum(call["duration_ms"] for call in llm_calls) / len(llm_calls))
    
    # Count tool calls
    tool_calls = performance_data.get("tools", [])
    metrics["tool_calls"] = len(tool_calls)
    
    # Count active sessions (placeholder, would need session tracking)
    metrics["active_sessions"] = 1
    
    # Get memory segments (simplified, would need proper tracking)
    metrics["memory_segments"] = 0
    memory_logs = performance_data.get("memory_manager", [])
    for log in memory_logs:
        if "segment_count" in log.get("details", {}):
            metrics["memory_segments"] = log["details"]["segment_count"]
            break
    
    return metrics

@debug_router.get("/metrics", response_model=Dict[str, Any])
async def get_debug_metrics():
    """Get system-wide debug metrics."""
    global debug_data_cache
    
    # Check if we need to refresh the cache
    current_time = time.time()
    if (current_time - debug_data_cache["metrics"].get("last_updated", 0)) > CACHE_EXPIRATION: # Added .get for safety
        calculated_metrics = calculate_metrics() # Store calculated metrics
        debug_data_cache["metrics"] = {**calculated_metrics, "last_updated": int(current_time)} # Merge and update timestamp
    
    return debug_data_cache["metrics"]

@debug_router.get("/performance/{component_name}", response_model=List[Dict[str, Any]])
async def get_component_performance_data(component_name: str):
    """Get performance data for a specific component."""
    global debug_data_cache
    current_time = time.time()

    # Check if performance data cache needs refresh
    if (current_time - debug_data_cache["performance"].get("last_updated", 0)) > CACHE_EXPIRATION:
        debug_data_cache["performance"] = extract_performance_data() 
        # extract_performance_data now adds its own "last_updated" key

    component_data = debug_data_cache["performance"].get(component_name)
    if component_data is None:
        # Try replacing hyphen with underscore if JS sends that way e.g. llm-interface
        component_data = debug_data_cache["performance"].get(component_name.replace('-', '_'))

    if component_data is None:
        raise HTTPException(status_code=404, detail=f"Performance data for component '{component_name}' not found.")
    
    return component_data


@debug_router.get("/logs/{log_type}", response_model=Dict[str, Any])
async def get_typed_logs(log_type: str):
    """Get logs filtered by type (all, error, warning)."""
    global debug_data_cache
    current_time = time.time()
    
    # For logs, we might want to parse them more frequently or on demand,
    # as they can change rapidly. Caching strategy might differ from metrics/performance.
    # For now, let's use a similar cache, but keyed by log_type.

    cache_key = f"logs_{log_type}"
    if cache_key not in debug_data_cache["logs"] or \
       (current_time - debug_data_cache["logs"][cache_key].get("last_updated", 0)) > CACHE_EXPIRATION / 2: # Shorter cache for logs
        
        parsed_logs = []
        if log_type == "all":
            parsed_logs = parse_log_files()
        elif log_type == "error":
            parsed_logs = parse_log_files(log_type="error")
        elif log_type == "warning":
            parsed_logs = parse_log_files(log_type="warning")
        else:
            raise HTTPException(status_code=400, detail="Invalid log type. Must be 'all', 'error', or 'warning'.")
            
        debug_data_cache["logs"][cache_key] = {
            "logs": parsed_logs,
            "last_updated": int(current_time)
        }
        
    return debug_data_cache["logs"][cache_key]
