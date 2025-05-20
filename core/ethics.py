# core/ethics.py
# Our friendly neighborhood ethics guardian! Keeping things nice and proper ^_^
from typing import Dict, Any, Optional  # Type hints, because we're responsible like that! =D

class EthicsViolationError(Exception):
    """Oopsie! Someone tried to do something they shouldn't! x.x"""
    pass

class EthicsManager:
    """
    The friendly but firm ethics guardian! \\o/
    
    We're like that responsible friend who makes sure nobody does anything silly:
    - Keeps an eye on what everyone's saying ^_^
    - Makes sure actions are safe and proper =D 
    - Steps in when needed (but tries to be nice about it! >.>)
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Time to set up our ethics compass! Setting our moral GPS =D
        
        Args:
            config: The rules we live by (very important stuff! O.o)
        """
        self.config = config
        self.ethics_enabled = self.config.get('ethics_enabled', True)  # Safety first! \\o/
        # Load our list of no-no words (keeping it clean and friendly! ^_^)
        self.disallowed_phrases = ["example harmful phrase"]  # Just a placeholder! We'll add more x.x
        print(f"[EthicsManager] Ready to keep things nice and proper! Ethics enabled: {self.ethics_enabled} \\o/")

    def check_text_content(self, text: str) -> bool:
        """
        Making sure everyone plays nice with their words! ^_^
        
        Args:
            text: The words we need to check (we read everything! O.o)
            
        Returns:
            bool: True if everything's okay, otherwise... well... x.x
            
        Raises:
            EthicsViolationError: When someone tries to be naughty! >.>
        """
        if not self.ethics_enabled:
            return True  # Living dangerously! But if you insist... =P
            
        for phrase in self.disallowed_phrases:
            if phrase in text.lower():
                raise EthicsViolationError(f"Whoops! Can't say that! Found this no-no phrase: '{phrase}' x.x")
        return True  # All clear! \\o/

    def approve_action(self, agent_name: str, action_type: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Time to check if this action gets a thumbs up! ^_^
        
        Args:
            agent_name: Who wants to do the thing? =D
            action_type: What are they trying to do? O.o
            details: All the juicy details (optional but helpful! \\o/)
            
        Returns:
            bool: True if we say "go for it!", False if we say "better not!" =P
        """
        if not self.ethics_enabled:
            return True  # Running with scissors mode enabled! x.x
            
        # TODO: Add more checks here! Right now we're too trusting >.>
        print(f"[EthicsManager] Checking if this is okay... Agent='{agent_name}', Type='{action_type}', Details='{details}' ^_^")
        return True  # For now, everyone gets a gold star! =D
