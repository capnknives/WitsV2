# core/llm_interface.py
import ollama
from typing import Any, Dict, Optional, Union, List
import json
from pydantic import ValidationError
from .schemas import OrchestratorLLMResponse  # For structured output parsing

class LLMInterface:
    """
    Interface for interacting with language models through Ollama.
    
    This class handles communication with Ollama LLMs, including:
    1. Basic text generation
    2. Structured JSON outputs for specific agent types
    3. Error handling and retry logic
    """
    
    def __init__(self, config: Any): 
        """
        Initialize the LLM interface with configuration.
        
        Args:
            config: Application configuration object
        """
        self.config = config
        self.default_model = self.config.models.default
        print(f"[LLMInterface] Initialized. Default model: {self.default_model}")

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
        model_to_use = model_name or self.default_model
        effective_options = {}
        
        # Use config options if available
        if hasattr(self.config, 'ollama_options'):
            effective_options.update(self.config.ollama_options)
        
        # Override with method-specific options if provided
        if options:
            effective_options.update(options)

        print(f"[LLMInterface] Calling model '{model_to_use}' (Text Output). Prompt length: {len(prompt)}")
        
        try:
            # Use synchronous ollama client for now
            # Future enhancement: switch to ollama.AsyncClient() with asyncio
            response = ollama.generate(
                model=model_to_use,
                prompt=prompt,
                options=effective_options
            )
            
            print(f"[LLMInterface] Model '{model_to_use}' responded.")
            return response.get('response', '').strip()
            
        except Exception as e:
            error_msg = f"Error calling model '{model_to_use}': {e}"
            print(f"[LLMInterface_ERROR] {error_msg}")
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
        model_to_use = model_name or self.config.models.orchestrator
        effective_options = {}
        
        # Use config options if available
        if hasattr(self.config, 'ollama_options'):
            effective_options.update(self.config.ollama_options)
        
        # Override with method-specific options if provided
        if options:
            effective_options.update(options)
        
        # Always set format to JSON for structured outputs
        effective_options["format"] = "json"

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
