import asyncio
import json
import logging
from typing import AsyncGenerator, List, Dict, Optional, Union, Any
import time # Add time for timestamps in memory

from agents.base_agent import BaseAgent
from agents.orchestrator_agent import OrchestratorAgent
from core.config import AppConfig
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager
from core.schemas import StreamData, MemorySegment, MemorySegmentContent # StreamData is now imported from core.schemas

class WitsControlCenterAgent(BaseAgent):
    def __init__(self,
                 agent_name: str,
                 config: AppConfig,
                 llm_interface: LLMInterface,
                 memory_manager: MemoryManager,
                 orchestrator_delegate: OrchestratorAgent,
                 specialized_agents: Dict[str, BaseAgent]):
        super().__init__(agent_name, config, llm_interface, memory_manager)
        self.orchestrator_delegate = orchestrator_delegate
        self.specialized_agents = specialized_agents
        self.logger = logging.getLogger(f"WITS.{self.__class__.__name__}")
        self.control_center_model_name = getattr(config.models, 'control_center', config.models.default)
        self.logger.info(f"WCCA initialized, using model: {self.control_center_model_name} for its decisions.")

    def _build_goal_elicitation_prompt(self, raw_user_input: str, conversation_history: List[Dict[str, str]]) -> str:
        history_str = "\\n".join([f"{turn['role']}: {turn['content']}" for turn in conversation_history])
        return f"""
You are the WITS Control Center, a master AI assistant. Your primary task is to understand the user's input within the context of the ongoing conversation and decide the best course of action.

Conversation History (most recent last):
{history_str}

Current User Input:
{raw_user_input}

Based on the 'Current User Input' and 'Conversation History', analyze the user's intent.
Respond with a single, valid JSON object containing a "type" field and other relevant fields based on that type.

Possible "type" values and their corresponding structures:

1.  If a clear, actionable goal for the Orchestrator sub-agent can be determined from the input:
    {{"type": "goal_defined", "goal_statement": "<The clear, actionable goal statement for the Orchestrator>"}}

2.  If the input is vague, ambiguous, or more information is needed to form an actionable goal:
    {{"type": "clarification_needed", "clarification_question": "<Your specific question to the user to get the necessary details>"}}

3.  (Future Expansion - Acknowledge but don't implement fully yet) If the input is a direct question that you (WITS Control Center) can answer without complex task execution:
    {{"type": "direct_answer_possible", "question_summary": "<A summary of the user's question>"}}

4.  (Future Expansion - Acknowledge but don't implement fully yet) If the user is asking for a status update or summary of agent activities:
    {{"type": "status_request_identified", "request_details": "<Details of the status request, e.g., which agent or task>"}}

For now, prioritize "goal_defined" and "clarification_needed".
Do not attempt to solve the goal yourself. Your job is to either define the goal clearly for the Orchestrator or ask for clarification.
Ensure the output is ONLY the JSON object.

JSON Response:
"""

    async def run(self, raw_user_input: str, conversation_history: List[Dict[str, str]], session_id: str) -> AsyncGenerator[StreamData, None]:
        self.logger.info(f"WCCA received for session '{session_id}': '{raw_user_input}' with history length: {len(conversation_history)}")

        # Log the initial user input to memory for this session.
        initial_user_input_content = MemorySegmentContent(text=raw_user_input)
        initial_user_input_segment = MemorySegment(
            type="user_input_to_wcca",
            source="USER",
            content=initial_user_input_content,
            metadata={
                "session_id": session_id,
                "timestamp": time.time(),
                "processed_by_agent": self.agent_name,
                "role": "user" # Add role for history reconstruction
            }
        )
        await self.memory.add_memory_segment(initial_user_input_segment)
        self.logger.debug(f"WCCA logged initial user input for session '{session_id}' to memory (ID: {initial_user_input_segment.id}).")

        prompt = self._build_goal_elicitation_prompt(raw_user_input, conversation_history)
        self.logger.debug(f"WCCA Goal Elicitation Prompt for session '{session_id}':\\n{prompt}")

        try:
            llm_response_str = await self.llm.chat_completion_async(
                model_name=self.control_center_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config_full.default_temperature,
            )
            self.logger.debug(f"WCCA LLM Raw Response for session '{session_id}': {llm_response_str}")

            parsed_response_json = None
            try:
                if "```json" in llm_response_str:
                    json_block_start = llm_response_str.find("```json") + len("```json")
                    json_block_end = llm_response_str.rfind("```")
                    json_str_cleaned = llm_response_str[json_block_start:json_block_end].strip()
                elif "```" in llm_response_str:
                    json_block_start = llm_response_str.find("```") + len("```")
                    json_block_end = llm_response_str.rfind("```")
                    json_str_cleaned = llm_response_str[json_block_start:json_block_end].strip()
                else:
                    json_start_index = llm_response_str.find('{')
                    json_end_index = llm_response_str.rfind('}')
                    if json_start_index != -1 and json_end_index != -1 and json_end_index > json_start_index:
                        json_str_cleaned = llm_response_str[json_start_index : json_end_index + 1]
                    else:
                        json_str_cleaned = llm_response_str
                
                parsed_response_json = json.loads(json_str_cleaned)
                self.logger.info(f"WCCA Parsed LLM JSON for session '{session_id}': {parsed_response_json}")

            except json.JSONDecodeError as e:
                self.logger.error(f"WCCA failed to parse LLM JSON response for session '{session_id}': {e}. Raw response: {llm_response_str}")
                yield StreamData(
                    type="error", 
                    content=f"Error: Could not understand the internal decision process. Raw response snippet: {llm_response_str[:200]}",
                    error_details=f"JSONDecodeError: {str(e)}. Raw response: {llm_response_str}"
                )
                
                error_details_dict = {
                    "raw_user_input": raw_user_input,
                    "llm_prompt_summary": "Goal Elicitation Prompt (see debug logs for full text)",
                    "llm_response_raw": llm_response_str,
                    "error_message": str(e)
                }
                error_segment_content_obj = MemorySegmentContent(text=json.dumps(error_details_dict))
                error_segment = MemorySegment(
                    type="agent_error",
                    source=self.agent_name,
                    content=error_segment_content_obj,
                    metadata={
                        "session_id": session_id,
                        "error_source_method": "WitsControlCenterAgent.run",
                        "error_subtype": "llm_json_parse_error",
                        "timestamp": time.time()
                    }
                )
                await self.memory.add_memory_segment(error_segment)
                return

            response_type = parsed_response_json.get("type")
            
            decision_details_dict = {
                "raw_user_input": raw_user_input,
                "llm_prompt_summary": "Goal Elicitation Prompt (see debug logs for full text)",
                "llm_response_raw_snippet": llm_response_str[:500],
                "llm_response_parsed": parsed_response_json,
                "conversation_history_length_at_call": len(conversation_history)
            }
            decision_segment_content_obj = MemorySegmentContent(text=json.dumps(decision_details_dict))
            memory_segment = MemorySegment(
                type="agent_decision_process",
                source=self.agent_name,
                content=decision_segment_content_obj,
                metadata={
                    "session_id": session_id,
                    "source_method": "WitsControlCenterAgent.run",
                    "llm_model": self.control_center_model_name,
                    "decision_type": response_type,
                    "timestamp": time.time()
                }
            )
            await self.memory.add_memory_segment(memory_segment)
            self.logger.info(f"WCCA decision process memory segment stored for session '{session_id}' (ID: {memory_segment.id}).")

            if response_type == "goal_defined":
                goal_statement = parsed_response_json.get("goal_statement")
                if not goal_statement:
                    self.logger.error(f"WCCA: 'goal_defined' type received for session '{session_id}' but 'goal_statement' is missing.")
                    yield StreamData(
                        type="error", 
                        content="Internal error: Goal statement missing from LLM decision.",
                        error_details="LLM response for goal_defined missing goal_statement field."
                    )
                    return

                yield StreamData(
                    type="info", 
                    content="Goal identified. Delegating to Orchestrator...",
                    goal_statement=goal_statement
                )
                self.logger.info(f"WCCA: Goal defined for session '{session_id}': '{goal_statement}'. Delegating to OrchestratorAgent.")
                
                orchestrator_context = {
                    "conversation_history": conversation_history, # This is prior history
                    "wcca_raw_user_input": raw_user_input, # Current input that led to this goal
                    "wcca_identified_goal": goal_statement,
                    "session_id": session_id
                }

                async for orchestrator_response in self.orchestrator_delegate.run(
                    user_goal=goal_statement,
                    context=orchestrator_context
                ):
                    # If orchestrator yields a final answer, WCCA should log it to memory for this session
                    if orchestrator_response.type == "final_answer":
                        final_answer_content = MemorySegmentContent(text=orchestrator_response.content)
                        final_answer_segment = MemorySegment(
                            type="ai_final_answer",
                            source=self.orchestrator_delegate.agent_name, # Source is orchestrator
                            content=final_answer_content,
                            metadata={
                                "session_id": session_id,
                                "timestamp": time.time(),
                                "role": "assistant" # Add role for history reconstruction
                            }
                        )
                        await self.memory.add_memory_segment(final_answer_segment)
                        self.logger.debug(f"WCCA logged Orchestrator's final answer for session '{session_id}' to memory (ID: {final_answer_segment.id}).")
                    yield orchestrator_response

            elif response_type == "clarification_needed":
                clarification_question = parsed_response_json.get("clarification_question")
                if not clarification_question:
                    self.logger.error(f"WCCA: 'clarification_needed' type received for session '{session_id}' but 'clarification_question' is missing.")
                    yield StreamData(
                        type="error", 
                        content="Internal error: Clarification question missing from LLM decision.",
                        error_details="LLM response for clarification_needed missing clarification_question field."
                    )
                    return
                
                self.logger.info(f"WCCA: Clarification needed for session '{session_id}'. Question: '{clarification_question}'")
                clarification_content = MemorySegmentContent(text=clarification_question)
                clarification_segment = MemorySegment(
                    type="ai_clarification_request",
                    source=self.agent_name,
                    content=clarification_content,
                    metadata={
                        "session_id": session_id,
                        "timestamp": time.time(),
                        "role": "assistant" # Add role for history reconstruction
                    }
                )
                await self.memory.add_memory_segment(clarification_segment)
                yield StreamData(
                    type="clarification_request_to_user", 
                    content=clarification_question,
                    clarification_question=clarification_question
                )

            elif response_type == "direct_answer_possible":
                question_summary = parsed_response_json.get("question_summary", "User asked a direct question.")
                self.logger.info(f"WCCA: Direct answer possible for session '{session_id}': '{question_summary}'. (Feature not fully implemented)")
                direct_answer_text = f"I understand you're asking: '{question_summary}'. I'm not yet equipped to answer this directly. Could you please rephrase your request as a goal I can work on?"
                direct_answer_content = MemorySegmentContent(text=direct_answer_text)
                direct_answer_segment = MemorySegment(
                    type="ai_direct_answer_attempt", # Changed type for clarity
                    source=self.agent_name,
                    content=direct_answer_content,
                    metadata={
                        "session_id": session_id,
                        "timestamp": time.time(),
                        "role": "assistant"
                    }
                )
                await self.memory.add_memory_segment(direct_answer_segment)
                yield StreamData(
                    type="clarification_request_to_user", 
                    content=direct_answer_text,
                    clarification_question=direct_answer_text
                )

            elif response_type == "status_request_identified":
                request_details = parsed_response_json.get("request_details", "User asked for status.")
                self.logger.info(f"WCCA: Status request identified for session '{session_id}': '{request_details}'. (Feature not fully implemented)")
                status_response_text = f"I understand you're asking for status: '{request_details}'. This feature is under development. Could you please state a different goal?"
                status_response_content = MemorySegmentContent(text=status_response_text)
                status_response_segment = MemorySegment(
                    type="ai_status_request_handling", # Distinct type for status request handling
                    source=self.agent_name,
                    content=status_response_content,
                    metadata={
                        "session_id": session_id,
                        "timestamp": time.time(),
                        "role": "assistant"
                    }
                )
                await self.memory.add_memory_segment(status_response_segment)
                yield StreamData(
                    type="clarification_request_to_user", 
                    content=status_response_text,
                    clarification_question=status_response_text
                )
                
            else:
                self.logger.warning(f"WCCA: Received unknown or unhandled response type from LLM for session '{session_id}': '{response_type}'. Full response: {parsed_response_json}")
                yield StreamData(
                    type="error", 
                    content=f"Received an unexpected decision type: '{response_type}'. Please try rephrasing your request.",
                    error_details=f"LLM returned unhandled type: {response_type}. Full response: {json.dumps(parsed_response_json)}"
                )

        except Exception as e:
            self.logger.exception(f"WCCA: An unexpected error occurred in 'run' method for session '{session_id}': {e}")
            yield StreamData(
                type="error", 
                content=f"An unexpected error occurred in the WITS Control Center: {str(e)}",
                error_details=str(e)
            )
