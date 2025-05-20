# core/llm_interface.py
# Behold! The magical bridge to our AI friends! ^_^
import ollama  # Our trusty companion in the AI adventure! \o/
import time
from typing import Any, Dict, Optional, Union, List
import json
from datetime import datetime
import logging
from pydantic import ValidationError

from .schemas import OrchestratorLLMResponse  # For keeping our responses neat and tidy =D
from .debug_utils import log_async_execution_time, DebugInfo, log_debug_info  # Debug powers, activate! x.x

class LLMInterface:
    """
    Your friendly AI language model interface! =D
    
    I help you chat with those fancy Ollama LLMs:
    1. Basic chitchat (the simple stuff ^_^)
    2. Fancy JSON outputs (gotta keep it structured \\o/)
    3. Fixing oopsies (retry all the things! >.>)
    4. Watching everything like a hawk (debug mode activated! x.x)
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,  # Which AI friend should we talk to? =P
        temperature: float = 0.7,  # How spicy should our responses be? ^_^
        ollama_url: Optional[str] = None,  # Where's our AI hanging out? 
        request_timeout: Optional[int] = None  # How long before we give up? >.>
    ): 
        """
        Time to make friends with an LLM! Let's get this party started \\o/
        
        Args:
            model_name: Pick your AI companion! (or let it pick itself =P)
            temperature: Creativity dial - 0 is boring, 1 is chaos! ^_^
            ollama_url: The secret hideout of our AI friend
            request_timeout: How patient should we be? (in seconds, because time is hard x.x)
        """
        self._model_name = model_name
        self._temperature = temperature
        self._ollama_url = ollama_url
        self._request_timeout = request_timeout
        
        # Time to start our diary! (aka logging) ^_^
        self.logger = logging.getLogger('WITS.LLMInterface')
        
        # Debug mode: for when things go boom! >.>
        self.debug_enabled = False
        self.debug_config = {"log_prompts": False, "log_responses": False, "log_tokens": False}
        
        self.logger.info(f"Ready to rock with model: {self._model_name or 'default'}, temp: {self._temperature} \\o/")
        if self._ollama_url:
            self.logger.info(f"Using Ollama server at {self._ollama_url}")

    @property
    def model_name(self) -> Optional[str]:
        """Get the current model name (our AI buddy's nickname =P)"""
        return self._model_name
    
    @model_name.setter
    def model_name(self, value: str):
        """Time to give our AI a new name! ^_^"""
        self._model_name = value
    
    @property
    def temperature(self) -> float:
        """How wild is our AI feeling? Temperature check! O.o"""
        return self._temperature
    
    @temperature.setter
    def temperature(self, value: float):
        """
        Adjust the AI's creativity thermostat! 
        0.0 = Serious business mode
        1.0 = Party mode! \\o/
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError("Whoa there! Temperature must be between 0.0 and 1.0 x.x")
        self._temperature = value

    async def generate_text(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Time to make the AI think! Let's see what it comes up with ^_^
        
        Args:
            prompt: The question we're asking our AI friend
            model_name: Which AI to chat with (optional party guest =P)
            options: Extra settings for maximum fun! \\o/
            
        Returns:
            str: What our AI friend had to say! 
        """
        start_time = time.time()
        model_to_use = model_name or self._model_name
        if not model_to_use:
            raise ValueError("Oops! We forgot to pick an AI to talk to! x.x")
        
        effective_options = {
            "temperature": self._temperature  # Setting the creativity meter! ^_^
        }
        
        # Are we in a hurry? Set a timeout! >.>
        if self._request_timeout:
            effective_options["timeout"] = self._request_timeout
        
        # Got any special requests? Add them here! =D
        if options:
            effective_options.update(options)

        # Sneaky debug mode activated! O.o
        if self.debug_enabled and self.debug_config.get('log_prompts', False):
            self.logger.debug(
                f"Here's what we're asking {model_to_use}:\n"
                f"{'='*40}\n{prompt}\n{'='*40}"
            )
        
        try:
            # Time to poke the AI and see what happens! ^_^
            generation_start = time.time()
            response = ollama.generate(
                model=model_to_use,
                prompt=prompt,
                options=effective_options
            )
            generation_time = (time.time() - generation_start) * 1000  # ms
            
            result = response.get('response', '').strip()
            
            if self.debug_enabled:
                # Time to spy on our AI's performance! O.o
                debug_info = DebugInfo(
                    timestamp=datetime.now().isoformat(),
                    component="LLMInterface",
                    action="generate_text",
                    details={
                        "model": model_to_use,
                        "prompt_length": len(prompt),
                        "response_length": len(result),
                    },
                    duration_ms=generation_time,
                    success=True
                )
                log_debug_info(self.logger, debug_info)
                
                # Let's see what our AI friend said! =D
                if self.debug_config.get('log_responses', False):
                    self.logger.debug(
                        f"Our AI buddy {model_to_use} says:\n"
                        f"{'='*40}\n{result}\n{'='*40}"
                    )
                
                # Number crunching time! x.x
                if self.debug_config.get('log_tokens', False) and 'eval_count' in response:
                    self.logger.debug(                        f"Fun stats from {model_to_use}! \\o/\n"
                        f"Brain cells used: {response['eval_count']}\n"
                        f"Think time: {response.get('eval_duration', 'Too fast to measure! =P')}"
                    )
            
            return result
            
        except Exception as e:
            error_msg = f"Uh oh! {model_to_use} got confused: {e} x.x"
            
            if self.debug_enabled:
                # Time to document our oopsie! O.o
                debug_info = DebugInfo(
                    timestamp=datetime.now().isoformat(),
                    component="LLMInterface",
                    action="generate_text",
                    details={
                        "model": model_to_use,
                        "prompt_length": len(prompt),
                        "options": effective_options
                    },
                    duration_ms=(time.time() - start_time) * 1000,
                    success=False,
                    error=str(e)
                )
                log_debug_info(self.logger, debug_info)
            
            self.logger.error(error_msg, exc_info=True)
            return f"Oops! {model_to_use} had a brain freeze >.< Details: {str(e)}"

    async def generate_structured_orchestrator_response(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Union[OrchestratorLLMResponse, str]:
        """
        Time for some organized thinking! Let's get our AI to format things nicely ^_^
        
        We'll help our AI friend:
        1. Parse their thoughts into pretty JSON =D
        2. Give them a few retries if they mess up >.>
        3. Package everything in a nice Pydantic bow! \\o/
        
        Args:
            prompt: The big question for our AI buddy
            model_name: Which AI should think about this? (optional)
            options: Special instructions for extra pizzazz! 
            
        Returns:
            Union[OrchestratorLLMResponse, str]: Either a neat package or an "oops" message x.x
        """
        model_to_use = model_name or self._model_name
        if not model_to_use:
            raise ValueError("Help! We forgot to pick an AI to talk to! x.x")
        
        effective_options = {
            "temperature": self._temperature,
            "format": "json"  # Making sure our AI speaks JSON! ^_^
        }
        
        if self._request_timeout:
            effective_options["timeout"] = self._request_timeout
        
        if options:
            effective_options.update(options)

        print(f"Time to chat with {model_to_use}! (JSON mode activated! \\o/)")
        print(f"Here's a peek at what we're asking:\n{prompt[:300]}...\n...\n{prompt[-300:]}")

        max_retries = 2  # Everyone deserves a second chance! ^_^
        
        raw_json_output = ""  # Initialize for scope
        for attempt in range(max_retries):
            try:
                # Let's see what our AI friend comes up with! =D
                response = ollama.generate(
                    model=model_to_use,
                    prompt=prompt,
                    options=effective_options
                )
                
                raw_json_output = response.get('response', '').strip()
                print(f"Got some fancy JSON from {model_to_use}! \\o/\n{raw_json_output[:200]}...")

                # Time to make it all pretty and organized! ^_^
                parsed_response = OrchestratorLLMResponse.model_validate_json(raw_json_output)
                print(f"Yay! {model_to_use}'s response is all neat and tidy! =D")
                return parsed_response
            
            except json.JSONDecodeError as e_json:
                error_msg = f"Oopsie! {model_to_use} forgot how to JSON (attempt {attempt + 1}/{max_retries}): {e_json} x.x"
                print(f"[ERROR] {error_msg}")
                
                if attempt == max_retries - 1:
                    # Time to give up... >.>
                    formatted_output = raw_json_output[:500] + "..." if len(raw_json_output) > 500 else raw_json_output
                    return f"Error: Our AI friend couldn't write proper JSON after {max_retries} tries! Output: {formatted_output}"
            
            except ValidationError as e_val:
                error_msg = f"Almost there! But the format isn't quite right (attempt {attempt + 1}/{max_retries}): {e_val} O.o"
                print(f"[ERROR] {error_msg}")
                
                if attempt == max_retries - 1:
                    formatted_output = raw_json_output[:500] + "..." if len(raw_json_output) > 500 else raw_json_output
                    return f"Error: {model_to_use} tried their best, but the response format wasn't quite right after {max_retries} attempts! Output: {formatted_output}"
            
            except Exception as e:
                error_msg = f"Unexpected plot twist! (attempt {attempt + 1}/{max_retries}): {e} >.< "
                print(f"[ERROR] {error_msg}")
                
                if attempt == max_retries - 1:
                    return f"Error: {model_to_use} ran into a wall after {max_retries} attempts! Details: {str(e)}"
            
            # Quick breather before trying again! =P
            # await asyncio.sleep(1)  # Uncomment when using asyncio
        
        return "Error: Something went wrong in our AI adventure x.x"

    async def chat_completion_async(
        self,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Time for a fun chat with our AI friend! \\o/
        
        Args:
            messages: Our conversation history with the AI ^_^
            model_name: Which AI to chat with? (optional party guest)
            options: Special chat settings for extra fun! =D
            format: Want JSON? We can do that! 
        """
        model_to_use = model_name or self._model_name
        if not model_to_use:
            raise ValueError("Oops! We need to pick an AI to chat with! x.x")

        effective_options = {
            "temperature": self._temperature  # Setting the fun-meter! ^_^
        }

        if self._request_timeout:
            effective_options["timeout"] = self._request_timeout  # Don't leave us hanging! >.>
        
        if options:
            effective_options.update(options)

        # If we want JSON, let's make sure our AI knows! =P
        if format == "json":
            messages = [
                {
                    "role": "system",
                    "content": "You must respond with valid JSON only. No other text or formatting. ^_^"
                }
            ] + messages
            
        try:
            # Time to have a nice chat! \\o/
            response = ollama.chat(
                model=model_to_use,
                messages=messages,
                options=effective_options
            )
            
            # Package everything up nicely! =D
            return {
                "response": response.message["content"],
                "model": response.model,
                "created_at": response.created_at,
            }
            
        except Exception as e:
            error_msg = f"Oh no! Chat with {model_to_use} went wrong: {e} x.x"
            self.logger.error(error_msg, exc_info=True)
            return {"error": error_msg}
