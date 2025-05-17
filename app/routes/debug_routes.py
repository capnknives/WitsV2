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
    "logs": {
        "logs": [],
        "last_updated": 0
    }
}

# Cache expiration time in seconds
CACHE_EXPIRATION = 15

# Process and parse log entries from files
def parse_log_files():
    """Parse the log files to extract structured data."""
    logs = []
    log_dir = os.path.join(os.getcwd(), "logs")
    
    if not os.path.exists(log_dir):
        return []
    
    # Find the most recent debug log file
    log_files = [f for f in os.listdir(log_dir) if f.startswith('wits_debug')]
    if not log_files:
        return []
    
    log_files.sort(reverse=True)
    log_file_path = os.path.join(log_dir, log_files[0])
    
    # Process the log file
    with open(log_file_path, 'r') as f:
        for line in f.readlines()[-1000:]:  # Read the last 1000 lines for performance
            try:
                # Parse log line
                parts = line.strip().split(' | ')
                if len(parts) < 3:
                    continue
                    
                timestamp_str, level, component_msg = parts[:3]
                component_parts = component_msg.split(' - ', 1)
                component = component_parts[0] if len(component_parts) > 0 else "unknown"
                message = component_parts[1] if len(component_parts) > 1 else component_msg
                
                logs.append({
                    "timestamp": timestamp_str,
                    "level": level.strip(),
                    "component": component.strip(),
                    "message": message.strip()
                })
            except Exception as e:
                print(f"Error parsing log line: {e}")
                continue
    
    return logs[-100:]  # Return the last 100 parsed log entries

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
        metrics["avg_response_time"] = sum(call["duration_ms"] for call in llm_calls) / len(llm_calls)
    
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
    if (current_time - debug_data_cache["metrics"]["last_updated"]) > CACHE_EXPIRATION:
        debug_data_cache["metrics"] = calculate_metrics()
        debug_data_cache["metrics"]["last_updated"] = current_time
    
    return debug_data_cache["metrics"]

@debug_router.get("/performance", response_model=Dict[str, Any])
async def get_performance_data():
    """Get performance data for all components."""
    global debug_data_cache
    
    # Check if we need to refresh the cache
    current_time = time.time()
    if (current_time - debug_data_cache["performance"]["last_updated"]) > CACHE_EXPIRATION:
        debug_data_cache["performance"] = extract_performance_data()
        debug_data_cache["performance"]["last_updated"] = current_time
    
    return debug_data_cache["performance"]

@debug_router.get("/logs", response_model=Dict[str, Any])
async def get_debug_logs():
    """Get recent system logs."""
    global debug_data_cache
    
    # Check if we need to refresh the cache
    current_time = time.time()
    if (current_time - debug_data_cache["logs"]["last_updated"]) > CACHE_EXPIRATION:
        debug_data_cache["logs"]["logs"] = parse_log_files()
        debug_data_cache["logs"]["last_updated"] = current_time
    
    return debug_data_cache["logs"]
