# app/services/agent_service.py
from typing import List, Dict, Any
from core.config import AppConfig, load_app_config

class AgentService:
    def __init__(self, config: AppConfig):
        self.config = config

    def get_agent_profiles(self) -> List[Dict[str, Any]]:
        profiles = []
        if self.config.agent_profiles:
            for name, profile_config in self.config.agent_profiles.items():
                # Construct a display name if not explicitly provided (e.g., from agent_class)
                # For now, using the profile key as name and a placeholder description.
                # The frontend expects 'name', 'display_name', 'description'.
                # We'll need to adjust how these are derived or stored in config.yaml.
                
                display_name = name.replace("_", " ").title() # Example: "engineer" -> "Engineer"
                description = f"Agent profile for {display_name}." 
                                
                # Attempt to get more specific details if available in agent_specific_params or profile_config
                if hasattr(profile_config, 'agent_specific_params') and profile_config.agent_specific_params:
                    description = profile_config.agent_specific_params.get('description', description)
                    display_name = profile_config.agent_specific_params.get('display_name', display_name)
                
                # Fallback for description if agent_class is available
                if 'description' not in profile_config.agent_specific_params and hasattr(profile_config, 'agent_class'):
                     description = f"Specialized agent: {profile_config.agent_class}"

                profiles.append({
                    "name": name,
                    "display_name": display_name,
                    "description": description,
                    # "agent_class": profile_config.agent_class, # Not directly needed by UI, but good for backend
                    # "model": profile_config.model, # Not directly needed by UI for selection
                    # "tool_names": profile_config.tool_names # Not directly needed by UI for selection
                })
        
        # Add a default orchestrator if no specific profiles are defined,
        # or ensure the primary orchestrator is always available.
        # This part needs to align with how agents are actually instantiated and selected.
        # For now, let's assume config.yaml will list all selectable agents.
        # If agent_profiles is empty, we might add a default one.
        if not profiles: # Or if we want a guaranteed default
            profiles.append({
                "name": "default_orchestrator",
                "display_name": "Default Orchestrator",
                "description": "Handles general tasks and coordinates other agents."
            })
            
        return profiles

# Global instance of the service, initialized with loaded config
# This approach might need refinement for FastAPI dependency injection
_config = load_app_config() # Loads from config.yaml by default
agent_service_instance = AgentService(config=_config)
