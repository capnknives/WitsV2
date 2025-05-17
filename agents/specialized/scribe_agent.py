from typing import Any, Dict, Optional, List
import logging
from agents.base_agent import BaseAgent
from core.schemas import MemorySegment
from pydantic import BaseModel, Field
from core.debug_utils import log_async_execution_time

class DocumentRequest(BaseModel):
    """Request model for documentation tasks."""
    content: str = Field(..., description="The content to be documented or transformed.")
    format_type: str = Field(..., description="Desired documentation format (e.g., 'markdown', 'report', 'summary').")
    target_audience: str = Field(default="general", description="Target audience for the documentation.")
    max_length: Optional[int] = Field(None, description="Maximum length of the output document.")

class DocumentResponse(BaseModel):
    """Response model for documentation tasks."""
    content: str = Field(..., description="The formatted documentation.")
    format_type: str = Field(..., description="The format of the documentation provided.")
    word_count: int = Field(..., description="Word count of the generated document.")
    error: Optional[str] = Field(None, description="Error message if documentation failed.")

class ScribeAgent(BaseAgent):
    """
    Specialized agent for documentation, content creation, and text transformation tasks.
    
    This agent is responsible for:
    1. Converting technical information into clear, concise documentation
    2. Generating reports and summaries from raw data
    3. Transforming content between different formats (technical to non-technical, etc.)
    4. Creating structured documentation following best practices
    """
    
    def __init__(self, agent_name: str, config: Any, llm_interface: Any, memory_manager: Any):
        super().__init__(agent_name, config, llm_interface, memory_manager)
        self.logger = logging.getLogger(f"WITS.Agents.{agent_name}")
        self.logger.info(f"ScribeAgent initialized with model: {self.agent_config['model_name']}")
    
    @log_async_execution_time(logging.getLogger('WITS.Agents.ScribeAgent'))
    async def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute a documentation or content creation task.
        
        Args:
            task_description: Description of the documentation task to perform
            context: Optional additional context
            
        Returns:
            str: The resulting documentation or content
        """
        context = context or {}
        self.logger.info(f"Running ScribeAgent task: {task_description[:100]}{'...' if len(task_description) > 100 else ''}")
        
        # Prepare prompt for document generation
        system_prompt = """You are a specialized Scribe Agent in the WITS-NEXUS system. 
Your purpose is to create high-quality documentation, reports, and other written content based on provided information.
Always focus on clarity, structure, and appropriate level of detail for the target audience.
Follow formatting instructions precisely and maintain a professional, coherent style throughout.
"""
        
        # Enhance context with relevant memory segments if available
        if self.memory:
            memory_segments = await self.memory.search(task_description, limit=5)
            if memory_segments:
                context["relevant_memories"] = [seg.content for seg in memory_segments]
        
        # Construct the full prompt with context
        user_prompt = f"Task: {task_description}\n\n"
        
        if "content" in context:
            user_prompt += f"Content to process: {context['content']}\n\n"
        
        if "format_type" in context:
            user_prompt += f"Desired format: {context['format_type']}\n"
            
        if "target_audience" in context:
            user_prompt += f"Target audience: {context['target_audience']}\n"
            
        if "max_length" in context:
            user_prompt += f"Maximum output length: {context['max_length']} words\n"
            
        if "relevant_memories" in context:
            user_prompt += "\nReference information:\n"
            for i, memory in enumerate(context["relevant_memories"]):
                user_prompt += f"[Reference {i+1}]: {memory}\n"
        
        # Generate documentation using LLM
        try:
            response = await self.llm.generate_text(
                prompt=user_prompt,
                model_name=self.agent_config["model_name"],
                system_prompt=system_prompt
            )
            
            # Store the result in memory
            if self.memory:
                memory_segment = MemorySegment(
                    type="AGENT_RESPONSE",
                    source=self.agent_name,
                    content=response
                )
                await self.memory.add(memory_segment)
            
            self.logger.info("Documentation task completed successfully")
            return response
            
        except Exception as e:
            error_msg = f"Error generating documentation: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    async def create_document(self, request: DocumentRequest) -> DocumentResponse:
        """
        Create a document based on the specified format and content.
        
        Args:
            request: DocumentRequest with content and format specifications
            
        Returns:
            DocumentResponse: The formatted document
        """
        try:
            context = {
                "content": request.content,
                "format_type": request.format_type,
                "target_audience": request.target_audience,
                "max_length": request.max_length
            }
            
            task_description = f"Create a {request.format_type} document from the provided content for {request.target_audience} audience."
            formatted_content = await self.run(task_description, context)
            
            # Calculate word count
            word_count = len(formatted_content.split())
            
            return DocumentResponse(
                content=formatted_content,
                format_type=request.format_type,
                word_count=word_count,
                error=None
            )
            
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            return DocumentResponse(
                content="",
                format_type=request.format_type,
                word_count=0,
                error=error_msg
            )
    
    async def summarize(self, content: str, max_length: int = 500) -> str:
        """
        Create a concise summary of the provided content.
        
        Args:
            content: The text to summarize
            max_length: Maximum length of the summary in words
            
        Returns:
            str: The summarized content
        """
        request = DocumentRequest(
            content=content,
            format_type="summary",
            max_length=max_length
        )
        
        response = await self.create_document(request)
        return response.content
