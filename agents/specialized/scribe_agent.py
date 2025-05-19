import json
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field, ValidationError

from agents.base_agent import BaseAgent
from core.llm_interface import LLMInterface
from core.memory_manager import MemoryManager # Kept for __init__ signature

# Define the output schema for the ScribeAgent
class ScribeOutputSchema(BaseModel):
    formatted_text: str = Field(description="The primary text output from the Scribe agent based on the task.")
    summary: Optional[str] = Field(None, description="An optional summary of the generated text or input content.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Any relevant metadata about the scribing task or output.")

class ScribeAgent(BaseAgent):
    def __init__(self, agent_name: str, config: Dict[str, Any], llm_interface: LLMInterface, memory_manager: MemoryManager, tool_registry: Optional[Any] = None):
        super().__init__(agent_name, config, llm_interface, memory_manager, tool_registry)
        self.llm_model_name = self.config_full.models.get('scribe', self.config_full.models.default)
        self.logger.info(f"'{self.agent_name}' initialized with LLM model: {self.llm_model_name}.")

    @staticmethod
    def get_description() -> str:
        return "An agent specialized in documentation, content transformation, and formatting. It takes text and instructions, then returns structured JSON output."

    @staticmethod
    def get_config_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "output_format_guidelines": {"type": "string", "default": "Ensure clarity and conciseness.", "description": "Specific guidelines for the Scribe's output format."}
            }
        }

    def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> str:
        effective_context = context if context is not None else {}
        session_id = effective_context.get("session_id", "unknown_session")
        
        book_writing_state_slice: Dict = effective_context.get("book_writing_state_slice", {})
        project_name = book_writing_state_slice.get("project_name", "Unknown Project")
        
        input_text_to_process = effective_context.get("input_text", "No specific input text provided. Task should be self-contained or use general knowledge.")
        # Access agent-specific config using self.agent_config (inherited from BaseAgent)
        output_format_guidelines = self.agent_config.get("output_format_guidelines", "Ensure clarity and conciseness.")

        self.logger.info(f"'{self.agent_name}' (session: {session_id}) received task: '{task_description}'.")

        plot_summary_excerpt = str(book_writing_state_slice.get("overall_plot_summary", "Not available."))[:300]
        chapter_outlines_count = len(book_writing_state_slice.get("detailed_chapter_outlines", []))
        character_profiles_count = len(book_writing_state_slice.get("character_profiles", []))
        world_notes_keys = list(book_writing_state_slice.get("world_building_notes", {}).keys())[:5]

        prompt = f"""You are the Scribe Agent for the project: "{project_name}".
Your task is to process text, format content, or generate documentation based on the instructions.

Task: {task_description}

Input Text to Process (if applicable):
---
{input_text_to_process}
---

General Output Formatting Guidelines: {output_format_guidelines}

Relevant Project Context (for awareness, not necessarily direct processing unless task specifies):
- Project Name: {project_name}
- Plot Summary Excerpt: {plot_summary_excerpt}...
- Number of Chapter Outlines: {chapter_outlines_count}
- Number of Character Profiles: {character_profiles_count}
- Sample World Building Note Keys: {world_notes_keys}

Your response MUST be a single JSON object conforming to the ScribeOutputSchema.
Schema: {ScribeOutputSchema.schema_json(indent=2)}

Example Response:
{{
  "formatted_text": "This is the primary output based on the task. It could be a summary, a reformatted document, a list, etc.",
  "summary": "Optional: A brief summary of what was done or the input text.",
  "metadata": {{
    "input_char_count": {len(str(input_text_to_process))},
    "task_type": "(e.g., summarization, reformatting, list_generation)"
  }}
}}

Ensure your output is valid JSON.
"""
        self.logger.debug(f"ScribeAgent LLM Prompt (session {session_id}):\\n{prompt[:1000]}...")
        
        try:
            llm_response_str = self.llm.chat_completion(
                model_name=self.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config_full.agent_temperature,
                max_tokens=self.config_full.agent_max_tokens,
                json_mode=True
            )
            self.logger.debug(f"ScribeAgent LLM Raw Response (session {session_id}): {llm_response_str[:300]}...")

            if not llm_response_str or not llm_response_str.strip():
                raise ValueError("LLM returned empty or whitespace-only response.")
            
            response_data = json.loads(llm_response_str)

        except json.JSONDecodeError as jde:
            self.logger.error(f"'{self.agent_name}' (session: {session_id}) JSONDecodeError: {jde}. Response: {llm_response_str}")
            return json.dumps({"error": "LLM output was not valid JSON.", "details": str(jde), "llm_response": llm_response_str})
        except Exception as e:
            self.logger.error(f"'{self.agent_name}' (session: {session_id}) Error during LLM call or parsing: {e}")
            return json.dumps({"error": "Error during Scribe task.", "details": str(e)})

        try:
            validated_output = ScribeOutputSchema.parse_obj(response_data)
            self.logger.info(f"'{self.agent_name}' (session: {session_id}) successfully validated Scribe output.")
            return validated_output.json() # Return as JSON string
        except ValidationError as ve:
            self.logger.error(f"'{self.agent_name}' (session: {session_id}) ValidationError for ScribeOutputSchema: {ve}. Data: {response_data}")
            return json.dumps({"error": "LLM output did not conform to ScribeOutputSchema.", "details": str(ve), "llm_response": response_data})
