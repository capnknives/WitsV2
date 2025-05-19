import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator, List, Dict, Optional, Union, Any
import time

from agents.base_agent import BaseAgent
from agents.orchestrator_agent import OrchestratorAgent
from core.config import AppConfig
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager
from core.schemas import StreamData, MemorySegment, MemorySegmentContent

class WitsControlCenterAgent(BaseAgent):
    """WITS Control Center Agent - The top-level coordinator of the WITS system.
    
    This agent is responsible for:
    1. Parsing user input and determining the appropriate response type
    2. Delegating complex tasks to the OrchestratorAgent
    3. Managing direct responses and clarifications
    4. Maintaining conversation context and memory

    I'm like the manager trying to make sense of everyone's requests xD
    """

    def __init__(
        self,
        agent_name: str,
        config: AppConfig,
        llm_interface: LLMInterface,
        memory_manager: MemoryManager,
        orchestrator_delegate: OrchestratorAgent,
        specialized_agents: Optional[Dict[str, BaseAgent]] = None
    ) -> None:
        """Initialize our awesome control center! Let's get this party started =D
        
        Args:
            agent_name: What they call me in the logs ^_^
            config: All the settings and stuff we need
            llm_interface: Our connection to the language model (please work!)
            memory_manager: Gotta remember things or we're toast x.x
            orchestrator_delegate: The ReAct planning mastermind we delegate to
            specialized_agents: The cool specialists we can call on (optional)
        """
        super().__init__(agent_name, config, llm_interface, memory_manager)
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
        
        Args:
            raw_user_input: What the user just told us
            conversation_history: The saga so far... 
        
        Returns:
            str: A carefully crafted prompt to help our LLM friend understand what's up
        """
        # Convert history to a nice readable format =D
        history_str = "\\n".join([f"{turn['role']}: {turn['content']}" for turn in conversation_history])
        
        # Build our super helpful prompt! Don't mess this up brain >.>
        return f"""
You are the WITS Control Center, a master AI assistant. Your primary task is to understand 
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

3. For unclear/vague requests:
   {{"type": "clarification_needed", "clarification_question": "<Your question to get more details>"}}

4. For direct, simple questions:
   {{"type": "direct_answer", "response": "<A clear, concise answer>"}}

Guidelines:
- Use "chat_response" for casual conversation
- Format tasks as clear "goal_statement"s
- Keep chat responses friendly and natural
- Structure goals for the Orchestrator to understand

Output only the JSON object, nothing else!
"""

    async def handle_llm_response(
        self, 
        parsed_response: Dict, 
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle whatever our LLM friend decided to do with the user's input.
        
        This is where we figure out if we're:
        - Just chatting (easy mode activated!)
        - Doing a complex task (deploy the Orchestrator! \\o/)
        - Asking for clarification (What do you mean? o.O)
        - Answering directly (I know this one! ...I think)
        
        Args:
            parsed_response: The LLM's decision in JSON form
            session_id: Which conversation we're in
            
        Yields:
            StreamData: Progress updates and results for the UI
        """
        # What kind of response are we dealing with? Let's find out! =D
        response_type = parsed_response.get("type")
        
        if response_type == "chat_response":
            # Just a friendly chat! Don't overthink it XD
            yield StreamData(
                type="response",
                content=parsed_response.get("response", "Oops, I lost my train of thought there! x.x")
            )
        
        elif response_type == "goal_defined":
            # Time for some serious business! Let's get the Orchestrator on this =D
            goal_statement = parsed_response.get("goal_statement")
            if not goal_statement or not isinstance(goal_statement, str):
                self.logger.error(f"Got a goal_defined but no valid goal?! What's happening?! O.o Goal: {goal_statement}")
                yield StreamData(
                    type="error",
                    content="Uh oh, my brain got confused. Can you try rephrasing that?"
                )
                return
                
            self.logger.debug(f"Passing the torch to Orchestrator for session '{session_id}' \\o/")
            
            # HERE WE GO! Time to delegate like a boss ^_^
            async for stream_data in self.orchestrator_delegate.run(
                goal_statement,
                {"session_id": session_id}
            ):
                yield stream_data
        
        elif response_type == "clarification_needed":
            # I am confusion! America explain! XD
            yield StreamData(
                type="clarification_request",
                content=parsed_response.get(
                    "clarification_question", 
                    "I'm a bit lost here... Could you explain that differently? >.>"
                )
            )
        
        elif response_type == "direct_answer":
            # I know this one! ...probably
            yield StreamData(
                type="response",
                content=parsed_response.get(
                    "response", 
                    "I thought I knew the answer but now I'm not so sure x.x"
                )
            )
        
        else:
            # What is this response type?! I've never seen this before! o.O
            yield StreamData(
                type="error",
                content=f"My brain encountered an unexpected response type: '{response_type}'. "
                        "Could you try asking in a different way?"
            )

    async def run(
        self,
        raw_user_input: str,
        conversation_history: List[Dict[str, str]],
        session_id: Optional[str] = None
    ) -> AsyncGenerator[StreamData, None]:
        """Time to process a user request! Let's do this! \\o/
        
        This is our main entry point where we:
        1. Log what the user said (don't forget or it's gone forever! x.x)
        2. Ask our LLM friend what to do (please be smart today)
        3. Handle the response (what could go wrong? ...everything)
        4. Remember what happened (future us will thank us... maybe)
        
        Args:
            raw_user_input: The user's message in its pure, unfiltered glory
            conversation_history: The epic tale so far
            session_id: Which conversation we're in (we'll make one if needed!)
        """
        # Make a session ID if we don't have one (we're so responsible! ^_^)
        if session_id is None:
            session_id = str(uuid.uuid4())
            self.logger.info(f"Created new session ID: {session_id} (I'm helping! =D)")

        self.logger.info(
            f"WCCA received message in session '{session_id}': '{raw_user_input}' "
            f"(history length: {len(conversation_history)})"
        )

        try:
            # First things first - save what the user said! Don't be lazy >.>
            initial_user_input_content = MemorySegmentContent(text=raw_user_input)
            initial_user_input_segment = MemorySegment(
                type="user_input_to_wcca",
                source="USER",
                content=initial_user_input_content,
                metadata={
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "processed_by_agent": self.agent_name,
                    "role": "user"  # For history reconstruction later! We're so smart ^_^
                }
            )
            await self.memory.add_memory_segment(initial_user_input_segment)
            self.logger.debug(f"Saved user input to memory like a good agent =D (ID: {initial_user_input_segment.id})")

            # Build our super smart prompt to figure out what to do
            prompt = self._build_goal_elicitation_prompt(raw_user_input, conversation_history)
            self.logger.debug(f"Built a beautiful prompt for session '{session_id}':\\n{prompt}")

            # Set up options for our LLM friend
            options = {}
            if hasattr(self.config_full, 'default_temperature'):
                options["temperature"] = self.config_full.default_temperature
                self.logger.debug(f"Using temperature {options['temperature']} (feeling creative today! ^_^)")

            # Ask the LLM what we should do (fingers crossed! x.x)
            llm_response = await self.llm.chat_completion_async(
                model_name=self.control_center_model_name,
                messages=[{"role": "user", "content": prompt}],
                options=options
            )
            
            # Get the actual response text and clean it up
            response_text = llm_response.get('response', '').strip()
            self.logger.debug(f"LLM responded! Let's see what we got... O.o\\n{response_text}")

            # Try to parse that JSON (this is where things usually explode XD)
            try:
                json_str_cleaned = self._extract_json_from_response(response_text)
                parsed_response_json = json.loads(json_str_cleaned)
                self.logger.info(f"Successfully parsed LLM response! We're on fire! \\o/")

                # Process what we got through our handler
                async for result in self.handle_llm_response(parsed_response_json, session_id):
                    yield result

                # Log what we decided (for science! And debugging... mostly debugging)
                await self._log_decision_to_memory(
                    session_id=session_id,
                    raw_user_input=raw_user_input,
                    response_text=response_text,
                    parsed_response=parsed_response_json,
                    conversation_history_length=len(conversation_history)
                )

            except json.JSONDecodeError as e:
                # JSON parsing failed... time to panic! x.x
                self.logger.error(
                    f"Failed to parse LLM response! What even is this?! >.>\\n"
                    f"Error: {e}\\nResponse: {response_text}"
                )
                yield StreamData(
                    type="error",
                    content=(
                        "I got a bit confused trying to process that... "
                        "Could you try asking in a different way?"
                    ),
                    error_details=f"JSON parse error: {str(e)}"
                )
                
                # Log our failure (it's important to remember our mistakes... I guess >.>)
                await self._log_error_to_memory(
                    session_id=session_id,
                    error_type="json_parse_error",
                    error=e,
                    raw_user_input=raw_user_input,
                    response_text=response_text
                )

        except Exception as e:
            # Something went really wrong... time to gracefully panic! O.o
            self.logger.exception(f"Something exploded in the WCCA! HELP! x.x\\nError: {e}")
            yield StreamData(
                type="error",
                content="Oops! Something went wrong in my brain... Could you try that again?",
                error_details=str(e)
            )
            
            # Log the catastrophe (for future archaeologists to study XD)
            await self._log_error_to_memory(
                session_id=session_id,
                error_type="unexpected_error",
                error=e,
                raw_user_input=raw_user_input
            )

    def _extract_json_from_response(self, response_text: str) -> str:
        """Try to find the JSON in an LLM response (it's like Where's Waldo! ^_^)"""
        if "```json" in response_text:
            # Fancy markdown format detected! =D
            json_block_start = response_text.find("```json") + len("```json")
            json_block_end = response_text.rfind("```")
            return response_text[json_block_start:json_block_end].strip()
        elif "```" in response_text:
            # Regular code block (you tried! ^_^)
            json_block_start = response_text.find("```") + len("```")
            json_block_end = response_text.rfind("```")
            return response_text[json_block_start:json_block_end].strip()
        else:
            # Look for bare JSON (living dangerously! o.O)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                return response_text[json_start:json_end + 1]
            return response_text  # YOLO! Maybe it's already clean JSON? >.>

    async def _log_decision_to_memory(
        self,
        session_id: str,
        raw_user_input: str,
        response_text: str,
        parsed_response: Dict,
        conversation_history_length: int
    ) -> None:
        """Log our decision process (because documentation is important... I guess x.x)"""
        decision_details = {
            "raw_user_input": raw_user_input,
            "llm_response_snippet": response_text[:500],  # Keep it reasonable XD
            "llm_parsed_response": parsed_response,
            "conversation_history_length": conversation_history_length
        }
        
        memory_segment = MemorySegment(
            type="agent_decision_process",
            source=self.agent_name,
            content=MemorySegmentContent(text=json.dumps(decision_details)),
            metadata={
                "session_id": session_id,
                "timestamp": time.time(),
                "decision_type": parsed_response.get("type", "unknown")
            }
        )
        await self.memory.add_memory_segment(memory_segment)
        self.logger.info(f"Logged decision process! (ID: {memory_segment.id}) We're so organized! =D")

    async def _log_error_to_memory(
        self,
        session_id: str,
        error_type: str,
        error: Exception,
        raw_user_input: str,
        response_text: Optional[str] = None
    ) -> None:
        """Log our failures (everyone messes up sometimes... right? >.>)"""
        error_details = {
            "raw_user_input": raw_user_input,
            "error_type": error_type,
            "error_message": str(error)
        }
        if response_text:
            error_details["llm_response"] = response_text[:500]  # Truncate the evidence XD
            
        error_segment = MemorySegment(
            type="agent_error",
            source=self.agent_name,
            content=MemorySegmentContent(text=json.dumps(error_details)),
            metadata={
                "session_id": session_id,
                "timestamp": time.time(),
                "error_type": error_type
            }
        )
        await self.memory.add_memory_segment(error_segment)
        self.logger.info(f"Logged our oopsie for posterity x.x (ID: {error_segment.id})")
