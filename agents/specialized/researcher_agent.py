from typing import Any, Dict, Optional, List
import logging
import json
import time
from agents.base_agent import BaseAgent
from core.schemas import MemorySegment
from pydantic import BaseModel, Field
from core.debug_utils import log_async_execution_time

class ResearchQuery(BaseModel):
    """Request model for research tasks."""
    query: str = Field(..., description="The research question or topic to investigate.")
    depth: str = Field(default="standard", description="Depth of research: 'quick', 'standard', or 'deep'.")
    focus_areas: List[str] = Field(default_factory=list, description="Specific areas to focus on within the research topic.")
    exclude_areas: List[str] = Field(default_factory=list, description="Areas to exclude from the research.")
    required_sources: int = Field(default=3, description="Minimum number of sources to include.")

class ResearchResult(BaseModel):
    """Response model for research tasks."""
    summary: str = Field(..., description="Executive summary of the research findings.")
    key_findings: List[str] = Field(..., description="List of key findings from the research.")
    sources: List[Dict[str, str]] = Field(..., description="Sources cited in the research.")
    confidence: float = Field(..., description="Confidence rating for the research results (0.0-1.0).")
    limitations: List[str] = Field(default_factory=list, description="Limitations of the research.")
    error: Optional[str] = Field(None, description="Error message if research encountered issues.")

class ResearcherAgent(BaseAgent):
    """
    Specialized agent for conducting research, gathering information, and synthesizing findings.
    
    This agent is responsible for:
    1. Researching topics using available knowledge and tools
    2. Gathering information from various sources
    3. Synthesizing findings into cohesive reports
    4. Identifying knowledge gaps and uncertainties
    """
    
    def __init__(self, agent_name: str, config: Any, llm_interface: Any, memory_manager: Any):
        super().__init__(agent_name, config, llm_interface, memory_manager)
        self.logger = logging.getLogger(f"WITS.Agents.{agent_name}")
        self.logger.info(f"ResearcherAgent initialized with model: {self.agent_config['model_name']}")
        
        # Track recent research topics to avoid redundancy
        self.recent_topics = []
    
    @log_async_execution_time(logging.getLogger('WITS.Agents.ResearcherAgent'))
    async def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute a research task.
        
        Args:
            task_description: Description of the research task to perform
            context: Optional additional context
            
        Returns:
            str: Research findings and synthesis
        """
        context = context or {}
        self.logger.info(f"Running ResearcherAgent task: {task_description[:100]}{'...' if len(task_description) > 100 else ''}")
        
        # Track this research topic
        self.recent_topics.append(task_description)
        if len(self.recent_topics) > 10:
            self.recent_topics.pop(0)
        
        # Prepare prompt for research
        system_prompt = """You are a specialized Researcher Agent in the WITS-NEXUS system.
Your purpose is to investigate topics thoroughly, gather relevant information, and provide comprehensive findings.
Focus on accuracy, objectivity, and proper citation of sources.
Distinguish between facts, expert opinions, and your own assessments.
Highlight areas of uncertainty and knowledge gaps when appropriate.
Structure your research with clear sections and organize information logically.
"""
        
        # Enhance context with relevant memory segments if available
        if self.memory:
            memory_segments = await self.memory.search(task_description, limit=5)
            if memory_segments:
                context["relevant_memories"] = [seg.content for seg in memory_segments]
                
            # Also check if we've done similar research recently
            for topic in self.recent_topics[:-1]:  # Exclude the current topic
                if self._calculate_similarity(topic, task_description) > 0.7:  # Arbitrary similarity threshold
                    context["similar_research"] = topic
                    related_segments = await self.memory.search(topic, limit=2)
                    if related_segments:
                        context["related_research_findings"] = [seg.content for seg in related_segments]
        
        # Construct the full prompt with context
        user_prompt = f"Research Task: {task_description}\n\n"
        
        if "depth" in context:
            user_prompt += f"Research depth: {context['depth']}\n"
            
        if "focus_areas" in context and context["focus_areas"]:
            user_prompt += f"Focus areas: {', '.join(context['focus_areas'])}\n"
            
        if "exclude_areas" in context and context["exclude_areas"]:
            user_prompt += f"Areas to exclude: {', '.join(context['exclude_areas'])}\n"
            
        if "required_sources" in context:
            user_prompt += f"Minimum sources required: {context['required_sources']}\n"
        
        if "similar_research" in context:
            user_prompt += f"\nNote: Similar research was previously conducted on: {context['similar_research']}\n"
        
        if "related_research_findings" in context:
            user_prompt += "\nPrevious related findings:\n"
            for i, finding in enumerate(context["related_research_findings"]):
                user_prompt += f"[Previous Finding {i+1}]: {finding}\n"
            user_prompt += "\nPlease build upon these previous findings and avoid redundancy.\n"
            
        if "relevant_memories" in context:
            user_prompt += "\nRelevant information from memory:\n"
            for i, memory in enumerate(context["relevant_memories"]):
                user_prompt += f"[Memory {i+1}]: {memory}\n"
        
        # Generate research results using LLM
        try:
            response = await self.llm.generate_text(
                prompt=user_prompt,
                model_name=self.agent_config["model_name"],
                system_prompt=system_prompt
            )
            
            # Store the result in memory
            if self.memory:
                memory_segment = MemorySegment(
                    type="RESEARCH_FINDINGS",
                    source=self.agent_name,
                    content=response,
                    metadata={"query": task_description, "timestamp": time.time()}
                )
                await self.memory.add(memory_segment)
            
            self.logger.info("Research task completed successfully")
            return response
            
        except Exception as e:
            error_msg = f"Error conducting research: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    async def research(self, query: ResearchQuery) -> ResearchResult:
        """
        Conduct research on a specified topic.
        
        Args:
            query: ResearchQuery with topic and parameters
            
        Returns:
            ResearchResult: Research findings, sources, and confidence
        """
        try:
            context = {
                "depth": query.depth,
                "focus_areas": query.focus_areas,
                "exclude_areas": query.exclude_areas,
                "required_sources": query.required_sources
            }
            
            research_findings = await self.run(query.query, context)
            
            # Extract sources from the research
            sources_prompt = f"""
Extract all sources mentioned in the following research.
For each source, provide a JSON object with "title", "author" (if available), and "description".
Return the results as a JSON array of sources.

Research:
{research_findings}
"""
            
            sources_response = await self.llm.generate_text(
                prompt=sources_prompt,
                model_name=self.agent_config["model_name"]
            )
            
            # Try to parse sources as JSON
            try:
                # Look for JSON-like content in the response
                start_idx = sources_response.find('[')
                end_idx = sources_response.rfind(']') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = sources_response[start_idx:end_idx]
                    sources = json.loads(json_str)
                else:
                    sources = []
                    # Fallback parsing for non-JSON responses
                    for line in sources_response.split('\n'):
                        if line.strip().startswith('- '):
                            sources.append({"title": line.strip('- '), "description": ""})
            except:
                self.logger.warning("Failed to parse sources as JSON")
                sources = [{"title": "Source parsing failed", "description": sources_response}]
            
            # Extract key findings
            key_findings_prompt = f"""
Extract 5-7 key findings from the following research.
Each finding should be a concise statement representing an important discovery or conclusion.

Research:
{research_findings}
"""
            
            findings_response = await self.llm.generate_text(
                prompt=key_findings_prompt,
                model_name=self.agent_config["model_name"]
            )
            
            # Parse key findings into a list
            key_findings = [line.strip().replace('- ', '') for line in findings_response.split('\n') 
                           if line.strip() and line.strip().startswith('- ')]
            
            # Calculate confidence based on quality of sources and depth
            confidence = min(0.95, 0.5 + (len(sources) / 10) + (0.1 if query.depth == "deep" else 0))
            
            # Extract limitations
            limitations_prompt = f"""
What are the limitations or weaknesses of the following research?
List 2-4 significant limitations that might affect the reliability of the findings.

Research:
{research_findings}
"""
            
            limitations_response = await self.llm.generate_text(
                prompt=limitations_prompt,
                model_name=self.agent_config["model_name"]
            )
            
            # Parse limitations into a list
            limitations = [line.strip().replace('- ', '') for line in limitations_response.split('\n') 
                          if line.strip() and line.strip().startswith('- ')]
            
            # Create executive summary
            summary_prompt = f"""
Create a concise executive summary (maximum 3 paragraphs) of the following research:

{research_findings}
"""
            
            summary = await self.llm.generate_text(
                prompt=summary_prompt,
                model_name=self.agent_config["model_name"]
            )
            
            return ResearchResult(
                summary=summary,
                key_findings=key_findings if key_findings else ["No key findings extracted"],
                sources=sources,
                confidence=confidence,
                limitations=limitations
            )
            
        except Exception as e:
            error_msg = f"Error in research function: {str(e)}"
            self.logger.error(error_msg)
            return ResearchResult(
                summary="Research failed due to an error",
                key_findings=["Research error"],
                sources=[],
                confidence=0.0,
                error=error_msg
            )
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate a simple similarity score between two text strings.
        This is a placeholder for more sophisticated similarity calculation.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        # Very simple word overlap calculation
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
