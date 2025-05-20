# ====================================================================
# WITS Control Center Agent - THE CORE BRAIN OF THE ENTIRE SYSTEM
# DO NOT STUB OR REMOVE THIS FILE UNDER ANY CIRCUMSTANCES!!!
# This is the central coordinator that handles all user input processing
# and decides whether to respond directly or delegate to specialized agents
# ====================================================================

import json
import logging
import uuid
from typing import AsyncGenerator, List, Dict, Optional, Any
import time

from agents.base_agent import BaseAgent
from agents.book_orchestrator_agent import BookOrchestratorAgent as OrchestratorAgent
from core.config import AppConfig
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager
from core.schemas import StreamData, MemorySegment, MemorySegmentContent
from utils.logging_utils import log_async_execution_time

class WitsControlCenterAgent(BaseAgent):
    """WITS Control Center Agent - The top-level coordinator of the WITS system.
    
    This agent is responsible for:
    1. Parsing user input and determining the appropriate response type
    2. Delegating complex tasks to the OrchestratorAgent
    3. Managing direct responses and clarifications
    4. Maintaining conversation context and memory

    I'm like the manager trying to make sense of everyone's requests xD
    
    !!! CRITICAL COMPONENT !!! 
    This class is the entry point for all user interactions and MUST NEVER be stubbed or deleted.
    If you're thinking of removing functionality from this class, DON'T! The whole system relies on
    this component being fully implemented to properly route and handle user requests.
    """

    def __init__(
        self,
        agent_name: str,
        config: AppConfig,
        llm_interface: LLMInterface,
        memory_manager: MemoryManager,
        orchestrator_delegate: OrchestratorAgent,  # REQUIRED! Don't ever make this optional! >.>
        specialized_agents: Optional[Dict[str, BaseAgent]] = None
    ) -> None:
        """Initialize our awesome control center! Let's get this party started =D
        
        Args:
            agent_name: What they call me in the logs ^_^
            config: All the settings and stuff we need
            llm_interface: Our connection to the language model (please work!)
            memory_manager: Gotta remember things or we're toast x.x
            orchestrator_delegate: The ReAct planning mastermind we delegate to
                                  CRITICAL - This must be properly initialized and passed!
            specialized_agents: The cool specialists we can call on (optional)
                               
        NOTE: This initialization MUST remain complete - don't stub/simplify this or the 
        system will break in mysterious ways! The orchestrator_delegate parameter is 
        especially critical as it's required for task delegation.
        """
        super().__init__(agent_name, config, llm_interface, memory_manager)
        # Ensure memory is set properly
        self.memory = memory_manager
        self.orchestrator_delegate = orchestrator_delegate
        self.specialized_agents = specialized_agents or {}  # Empty dict if None provided =)
        self.logger = logging.getLogger(f"WITS.{self.__class__.__name__}")
        # Use the llm_interface's model name first, fall back to config if needed
        self.control_center_model_name = (
            llm_interface.model_name or 
            getattr(config.models, 'control_center', config.models.default)
        )
        self.logger.info(
            f"WCCA initialized! Using model: {self.control_center_model_name} for decisions. "
            "Time to make some magic happen! \\o/"
        )

    def _build_goal_elicitation_prompt(self, raw_user_input: str, conversation_history: List[Dict[str, str]]) -> str:
        """Build a prompt to figure out what the user wants us to do ^_^
        
        This is where we put on our detective hat and try to understand if the user:
        - Just wants to chat (easy peasy!)
        - Has a specific task (time to delegate!)
        - Needs clarification (confusion intensifies o.O)
        - Has a direct question (I got this! ...maybe)
        
        !!! CRITICAL PROMPT ENGINEERING - DO NOT SIMPLIFY !!!
        This prompt template is carefully crafted to enable accurate intent
        classification. Changes here will directly impact the system's ability
        to understand what the user wants. If you're thinking of shortening or
        simplifying this prompt, don't! The entire user experience depends on it.
        
        Args:
            raw_user_input: What the user just told us
            conversation_history: The saga so far... 
        
        Returns:
            str: A carefully crafted prompt to help our LLM friend understand what's up
        """
        # Convert history to a nice readable format =D
        history_str = "\n".join([f"{turn['role']}: {turn['content']}" for turn in conversation_history])
        
        # Build our super helpful prompt! Don't mess this up brain >.>
        return f"""You are the WITS Control Center, a master AI assistant. Your primary task is to understand 
the user's input and decide the best course of action.

Conversation History (most recent last):
{history_str}

Current User Input:
{raw_user_input}

Based on the input and history, analyze the user's intent.
Respond with a single JSON object containing a "type" field and other relevant fields based on that type.

Possible types:

1. For chat interactions (greetings, social stuff):
   {{"type": "chat_response", "response": "<A friendly chat response>"}}

2. For clear, actionable tasks:
   {{"type": "goal_defined", "goal_statement": "<A structured goal statement>"}}
"""

    async def _determine_user_intent(self, raw_user_input: str, conversation_history: List[Dict[str, str]], session_id: str) -> Dict[str, Any]:
        """Analyze user input to determine whether it's a chat or a task request.
        
        This method sends the user input to our LLM to get a structured understanding
        of what the user wants - simple chat or a complex goal to be orchestrated.
        
        !!! CRITICAL IMPLEMENTATION - DO NOT STUB OR SIMPLIFY !!!
        This is a core decision-making method that determines the entire flow of
        the WITS system. If this method isn't properly implemented, the system 
        will fail to properly route user requests.
        
        Args:
            raw_user_input: What the user typed
            conversation_history: Previous exchanges in this conversation
            session_id: Current session identifier
            
        Returns:
            Dict containing intent analysis with 'type' and other relevant fields
        """
        try:
            # Build a prompt for the LLM to analyze user intent
            prompt = self._build_goal_elicitation_prompt(raw_user_input, conversation_history)
            
            # Call the LLM to get structured intent analysis
            llm_response = await self.llm.chat_completion_async(
                model_name=self.control_center_model_name,
                messages=[{"role": "system", "content": "You are WITS Control Center, determining user intent."},
                          {"role": "user", "content": prompt}]
            )
            
            # Extract the response content
            response_content = llm_response.get('response', '')
            self.logger.debug(f"Raw LLM intent response: {response_content[:500]}...")
            
            # Try to extract JSON from the response
            try:
                # First attempt to find JSON in the response
                json_str = self._extract_json_from_response(response_content)
                parsed_response = json.loads(json_str)
                
                # Validate that we got the expected format
                if "type" not in parsed_response:
                    raise ValueError("Response missing required 'type' field")
                
                # For chat responses, ensure we have the response text
                if parsed_response["type"] == "chat_response" and "response" not in parsed_response:
                    parsed_response["response"] = "I'm sorry, I'm not sure how to respond to that."
                
                # For goal requests, ensure we have a structured goal statement
                elif parsed_response["type"] == "goal_defined" and "goal_statement" not in parsed_response:
                    parsed_response["goal_statement"] = raw_user_input
                    
                return parsed_response
                
            except (json.JSONDecodeError, ValueError) as e:
                # If we can't parse the JSON properly, make a best guess based on the raw input
                self.logger.warning(f"Failed to parse LLM intent response: {str(e)}. Using fallback.")
                
                # Simple heuristic: if it looks like a question or greeting, treat as chat
                if raw_user_input.lower().startswith(('hi', 'hello', 'hey', 'what', 'who', 'why', 'how', 'can you')):
                    return {"type": "chat_response", "response": response_content}
                else:
                    # Otherwise assume it's a task/goal
                    return {"type": "goal_defined", "goal_statement": raw_user_input}
                
        except Exception as e:
            # If anything goes wrong, fall back to just treating the input as a goal
            self.logger.error(f"Error in _determine_user_intent: {str(e)}", exc_info=True)
            return {"type": "goal_defined", "goal_statement": raw_user_input}

    async def handle_llm_response(self, llm_response_json: Dict[str, Any], session_id: Optional[str] = None) -> AsyncGenerator[StreamData, None]:
        """Handle the intent determined by the LLM.
        
        This method takes the intent classification from _determine_user_intent
        and either provides a direct chat response or delegates complex tasks to
        the orchestrator agent.
        
        !!! CRITICAL ROUTER - DO NOT STUB OR DELETE !!!
        This is the crucial routing logic that determines whether the system
        responds directly or delegates to the orchestrator. Without this
        properly implemented, the entire system's decision-making fails.
        If you're thinking of simplifying this logic, DON'T! The branching
        here is essential for proper WITS functionality.
        
        Args:
            llm_response_json: The parsed response from _determine_user_intent
            session_id: Current session identifier (optional)
            
        Yields:
            StreamData: Data packets for UI updates and responses
        """
        try:
            # Ensure we have a valid session ID
            session_id = session_id or str(uuid.uuid4())
            
            # Handle based on the detected intent type
            if llm_response_json.get("type") == "goal_defined":
                goal_statement = llm_response_json.get("goal_statement", "")
                
                if not goal_statement.strip():
                    yield StreamData(
                        type="error", 
                        content="I couldn't understand your request clearly. Could you please rephrase it?"
                    )
                    return
                
                # First, let the user know we're working on their task
                yield StreamData(
                    type="info",
                    content=f"I understand! Working on: {goal_statement}"
                )
                
                # Now delegate to the orchestrator agent with the structured goal
                context = {"session_id": session_id}
                async for data in self.orchestrator_delegate.run(goal_statement, context):
                    # Pass through all orchestrator responses
                    yield data
                
            elif llm_response_json.get("type") == "chat_response":
                # Direct chat response, no need for delegation
                response = llm_response_json.get("response", "I'm not sure how to respond to that.")
                yield StreamData(type="chat", content=response)
                
            else:
                # Unknown intent type - this shouldn't happen with proper LLM responses
                yield StreamData(
                    type="error", 
                    content="I couldn't determine how to handle your request. Could you try again?"
                )
                
        except Exception as e:
            # Catch any errors during processing
            self.logger.error(f"Error in handle_llm_response: {str(e)}", exc_info=True)
            yield StreamData(
                type="error",
                content=f"I encountered an issue while processing your request. Please try again.",
                error_details=str(e)
            )

    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from LLM response text.
        
        LLMs sometimes wrap JSON in markdown code blocks or add extra text.
        This method tries to extract just the JSON part.
        
        !!! CRITICAL UTILITY - DO NOT DELETE OR SIMPLIFY !!!
        This method enables proper parsing of LLM responses which is essential
        for the WCCA to correctly interpret intent classification results.
        Without this, many LLM responses would fail to parse correctly.
        
        Args:
            response_text: The raw LLM response text
            
        Returns:
            str: The extracted JSON string
        """
        if "```json" in response_text:
            # Extract from JSON code block with explicit labeling
            json_block_start = response_text.find("```json") + len("```json")
            json_block_end = response_text.rfind("```")
            return response_text[json_block_start:json_block_end].strip()
        
        elif "```" in response_text:
            # Extract from generic code block
            json_block_start = response_text.find("```") + len("```")
            json_block_end = response_text.rfind("```")
            return response_text[json_block_start:json_block_end].strip()
        
        else:
            # Try to find bare JSON by looking for outermost braces
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1  # Include the closing brace
            
            if json_start != -1 and json_end > json_start:
                return response_text[json_start:json_end].strip()
            
            # If all else fails, just return the original text
            return response_text

    async def _log_decision_to_memory(self, session_id: Optional[str], raw_user_input: str, 
                                    response_text: str, parsed_response: Dict[str, Any],
                                    conversation_history_length: int) -> None:
        """Log decision process to memory for future reference.
        
        !!! CRITICAL FOR SYSTEM MEMORY - DO NOT DELETE !!!
        This method saves information about the decisions made by the WCCA,
        which is essential for maintaining context across conversations and 
        enabling users to review past interactions. Removing this would
        damage the system's ability to learn from past interactions.
        
        Args:
            session_id: Current session identifier
            raw_user_input: Original user input
            response_text: Raw LLM response
            parsed_response: Parsed response as a dictionary
            conversation_history_length: Length of the conversation history
        """
        if not session_id:
            session_id = str(uuid.uuid4())
            
        try:
            decision_details = {
                "raw_user_input": raw_user_input,
                "llm_response_snippet": response_text[:500] if len(response_text) > 500 else response_text,
                "intent_type": parsed_response.get("type", "unknown"),
                "conversation_history_length": conversation_history_length
            }
            
            # Add goal statement if this was a task request
            if parsed_response.get("type") == "goal_defined":
                decision_details["goal_statement"] = parsed_response.get("goal_statement", "")
                
            # Add memory segment using the correct method signature
            segment_id = await self.memory.add_segment(
                segment_type="AGENT_DECISION",
                content_text=json.dumps(decision_details),
                source=self.agent_name,
                tool_name="intent_classifier",
                importance=0.7,  # Decisions are pretty important!
                meta={
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "decision_type": parsed_response.get("type", "unknown")
                }
            )
            
            self.logger.debug(f"Logged decision process (ID: {segment_id})")
            
        except Exception as e:
            self.logger.error(f"Failed to log decision to memory: {str(e)}", exc_info=True)

    @log_async_execution_time(logging.getLogger("WITS.WitsControlCenterAgent"))
    async def run(
        self,
        raw_user_input: str,
        conversation_history: List[Dict[str, str]],
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[StreamData, None]:
        """Process user requests and coordinate responses.
        
        This is the main entry point for the WCCA. It:
        1. Analyzes user input
        2. Determines if it's a simple chat or complex task
        3. Either responds directly or delegates to the orchestrator
        4. Logs the decision process to memory
        
        !!! ABSOLUTELY CRITICAL COMPONENT - DO NOT MODIFY WITHOUT EXTREME CAUTION !!!
        This method is the primary entry point for ALL user interactions in the system.
        Modifying this incorrectly will break the entire application flow.
        If you're thinking about stubbing this out - STOP! The system will completely
        fail to function properly. This is not exaggeration - it's the truth x.x
        
        Args:
            raw_user_input: The user's input text
            conversation_history: Previous conversation turns
            session_id: Optional session identifier
        
        Yields:
            StreamData: Response streams for the user interface
        """
        # Ensure we have a valid session ID
        session_id = session_id or str(uuid.uuid4())
        
        self.logger.info(f"WCCA processing request in session '{session_id}': {raw_user_input}")
        
        try:
            # Add user input to memory - using correct parameter names for add_segment
            await self.memory.add_segment(
                segment_type="USER_INPUT",
                content_text=raw_user_input,
                source="user",
                meta={"session_id": session_id, "timestamp": time.time()}
            )
            
            # Determine user intent - chat or task?
            llm_response = await self._determine_user_intent(
                raw_user_input, 
                conversation_history, 
                session_id
            )
            
            # Log the intent classification response
            self.logger.info(f"Intent classification response: {llm_response}")
            
            # Log the decision process to memory
            await self._log_decision_to_memory(
                session_id, 
                raw_user_input,
                str(llm_response), 
                llm_response, 
                len(conversation_history)
            )
            
            # Handle the response - either chat directly or delegate to orchestrator
            async for data in self.handle_llm_response(llm_response, session_id):
                yield data
            
        except Exception as e:
            # Something went wrong
            self.logger.exception(f"Error in WCCA run method: {str(e)}")
            yield StreamData(
                type="error",
                content="I encountered an issue processing your request. Please try again.",
                error_details=str(e)
            )
            
        # Log completion
        self.logger.info(f"Request processing complete for session '{session_id}'")
