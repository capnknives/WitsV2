# core/ethics.py
from typing import Dict, Any

class EthicsViolationError(Exception):
    pass

class EthicsManager:
    """Handles ethics checks for LLM outputs and agent actions."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ethics_enabled = self.config.get('ethics_enabled', True)
        # Load rules from an overlay file or define static rules
        self.disallowed_phrases = ["example harmful phrase"] # Placeholder
        print(f"[EthicsManager] Initialized. Ethics enabled: {self.ethics_enabled}")

    def check_text_content(self, text: str) -> bool:
        if not self.ethics_enabled:
            return True
        for phrase in self.disallowed_phrases:
            if phrase in text.lower():
                raise EthicsViolationError(f"Content violates ethics policy due to phrase: '{phrase}'")
        return True

    def approve_action(self, agent_name: str, action_type: str, details: Dict[str, Any] = None) -> bool:
        if not self.ethics_enabled:
            return True
        # Implement action approval logic based on config and rules
        # e.g., check self.config.get('allow_code_execution') for 'execute_code' action
        print(f"[EthicsManager] Approving action: Agent='{agent_name}', Type='{action_type}', Details='{details}' (Placeholder approval)")
        return True
