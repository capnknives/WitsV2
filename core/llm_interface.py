# core/llm_interface.py
import ollama
import time
from typing import Any, Dict, Optional, Union, List
import json
from datetime import datetime
import logging
from pydantic import ValidationError

from .schemas import OrchestratorLLMResponse  # For structured output parsing
from .debug_utils import log_async_execution_time, DebugInfo, log_debug_info

class LLMInterface:
    """
    Interface for interacting with language models through Ollama.
    
    This class handles communication with Ollama LLMs, including:
    1. Basic text generation
    2. Structured JSON outputs for specific agent types
    3. Error handling and retry logic
    4. Detailed debugging and performance monitoring
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        ollama_url: Optional[str] = None,
        request_timeout: Optional[int] = None
    ): 
        """
        Initialize the LLM interface with configuration.
        
        Args:
            model_name: The name of the model to use by default
            temperature: The temperature parameter for generation (0.0 to 1.0)
            ollama_url: The URL of the Ollama server
            request_timeout: Timeout for Ollama requests in seconds
        """
        self._model_name = model_name
        self._temperature = temperature
        self._ollama_url = ollama_url
        self._request_timeout = request_timeout
        
        # Set up logging
        self.logger = logging.getLogger('WITS.LLMInterface')
        
        # Set default debug values
        self.debug_enabled = False
        self.debug_config = {"log_prompts": False, "log_responses": False, "log_tokens": False}
        
        self.logger.info(f"Initialized with model: {self._model_name or 'default'}, temp: {self._temperature}")
        if self._ollama_url:
            self.logger.info(f"Using Ollama server at {self._ollama_url}")

    @property
    def model_name(self) -> Optional[str]:
        """Get the current model name."""
        return self._model_name
    
    @model_name.setter
    def model_name(self, value: str):
        """Set the model name."""
        self._model_name = value
    
    @property
    def temperature(self) -> float:
        """Get the current temperature value."""
        return self._temperature
    
    @temperature.setter
    def temperature(self, value: float):
        """Set the temperature value."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        self._temperature = value

    @log_async_execution_time(logging.getLogger('WITS.LLMInterface'))
    async def generate_text(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate text from the LLM.
        
        Args:
            prompt: The input prompt for the LLM
            model_name: The Ollama model to use (overrides default if specified)
            options: Additional options for the Ollama generation
            
        Returns:
            str: The generated text or error message
        """
        start_time = time.time()
        model_to_use = model_name or self._model_name
        if not model_to_use:
            raise ValueError("No model name specified and no default model set")
        
        effective_options = {
            "temperature": self._temperature
        }
        
        # Set request timeout if provided
        if self._request_timeout:
            effective_options["timeout"] = self._request_timeout
        
        # Override with method-specific options if provided
        if options:
            effective_options.update(options)

        # Log prompt if enabled
        if self.debug_enabled:
            try:
                if self.debug_config.get('log_prompts', False):
                    self.logger.debug(
                        f"Prompt to {model_to_use}:\n"
                        f"{'='*40}\n{prompt}\n{'='*40}"
                    )
            except:
                pass
        
        try:
            # Set up Ollama client with custom URL if provided
            if self._ollama_url:
                ollama.set_url(self._ollama_url)
            
            # Use synchronous ollama client for now
            # Future enhancement: switch to ollama.AsyncClient() with asyncio
            generation_start = time.time()
            response = ollama.generate(
                model=model_to_use,
                prompt=prompt,
                options=effective_options
            )
            generation_time = (time.time() - generation_start) * 1000  # ms
            
            result = response.get('response', '').strip()
            
            # Log performance metrics
            if self.debug_enabled:
                debug_info = DebugInfo(
                    timestamp=datetime.now().isoformat(),
                    component="LLMInterface",
                    action="generate_text",
                    details={
                        "model": model_to_use,
                        "prompt_length": len(prompt),
                        "response_length": len(result),
                        "options": effective_options
                    },
                    duration_ms=generation_time,
                    success=True
                )
                log_debug_info(self.logger, debug_info)
                
                # Log response if enabled
                try:
                    if isinstance(self.debug_config, dict) and self.debug_config.get('log_responses', False):
                        self.logger.debug(
                            f"Response from {model_to_use}:\n"
                            f"{'='*40}\n{result}\n{'='*40}"
                        )
                except:
                    pass
                
                # Log token metrics if enabled
                try:
                    if isinstance(self.debug_config, dict) and self.debug_config.get('log_tokens', False) and 'eval_count' in response:
                        self.logger.debug(
                            f"Token metrics for {model_to_use}:\n"
                            f"Eval count: {response['eval_count']}\n"
                            f"Eval duration: {response.get('eval_duration', 'N/A')}"
                        )
                except:
                    pass
            
            return result
            
        except Exception as e:
            error_msg = f"Error calling model '{model_to_use}': {e}"
            
            # Log error with debug info
            if self.debug_enabled:
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
            return f"Error: LLM call failed for model {model_to_use}. Details: {str(e)}"

    async def generate_structured_orchestrator_response(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Union[OrchestratorLLMResponse, str]:
        """
        Generate a structured JSON response from the LLM for the Orchestrator.
        
        This method handles parsing the LLM output into a Pydantic model and
        includes retry logic for handling JSON parsing errors.
        
        Args:
            prompt: The orchestrator prompt for the LLM
            model_name: The Ollama model to use (overrides orchestrator model if specified)
            options: Additional options for the Ollama generation
            
        Returns:
            Union[OrchestratorLLMResponse, str]: Parsed Pydantic model or error string
        """
        model_to_use = model_name or self._model_name
        if not model_to_use:
            raise ValueError("No model name specified and no default model set")
        
        effective_options = {
            "temperature": self._temperature,
            "format": "json"  # Always set format to JSON for structured outputs
        }
        
        # Set request timeout if provided
        if self._request_timeout:
            effective_options["timeout"] = self._request_timeout
        
        # Override with method-specific options if provided
        if options:
            effective_options.update(options)

        print(f"[LLMInterface] Calling model '{model_to_use}' (Structured JSON Output for Orchestrator).")
        print(f"Prompt Snippet:\n{prompt[:300]}...\n...\n{prompt[-300:]}")

        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                # Call the LLM
                response = ollama.generate(
                    model=model_to_use,
                    prompt=prompt,
                    options=effective_options
                )
                
                raw_json_output = response.get('response', '').strip()
                print(f"[LLMInterface] Raw JSON from model '{model_to_use}':\n{raw_json_output[:200]}...")

                # Attempt to parse the JSON into the Pydantic model
                parsed_response = OrchestratorLLMResponse.model_validate_json(raw_json_output)
                print(f"[LLMInterface] Model '{model_to_use}' response successfully parsed into OrchestratorLLMResponse.")
                return parsed_response
            
            except json.JSONDecodeError as e_json:
                error_msg = f"JSONDecodeError on attempt {attempt + 1}/{max_retries}: {e_json}."
                print(f"[LLMInterface_ERROR] {error_msg}")
                
                # Only return error on final attempt
                if attempt == max_retries - 1:
                    formatted_output = raw_json_output[:500] + "..." if len(raw_json_output) > 500 else raw_json_output
                    return f"Error: LLM output was not valid JSON after {max_retries} attempts. Output: {formatted_output}"
            
            except ValidationError as e_val:
                error_msg = f"Pydantic ValidationError on attempt {attempt + 1}/{max_retries}: {e_val}."
                print(f"[LLMInterface_ERROR] {error_msg}")
                
                # Only return error on final attempt
                if attempt == max_retries - 1:
                    formatted_output = raw_json_output[:500] + "..." if len(raw_json_output) > 500 else raw_json_output
                    return f"Error: LLM output did not match expected OrchestratorLLMResponse schema after {max_retries} attempts. Output: {formatted_output}"
            
            except Exception as e:
                error_msg = f"General error on attempt {attempt + 1}/{max_retries} calling model '{model_to_use}': {e}"
                print(f"[LLMInterface_ERROR] {error_msg}")
                
                # Only return error on final attempt
                if attempt == max_retries - 1:
                    return f"Error: LLM call failed for model {model_to_use} after {max_retries} attempts. Details: {str(e)}"
            
            # If we get here, an error occurred but we have more retries - add a short delay
            # await asyncio.sleep(1) # Uncomment when using asyncio
        
        # Should not reach here if the loop handles all cases correctly
        return "Error: LLM generation failed after all retries."
