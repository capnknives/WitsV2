from typing import Any, Dict, Optional, List, Union
import logging
import json
from agents.base_agent import BaseAgent
from core.schemas import MemorySegment
from pydantic import BaseModel, Field
from core.debug_utils import log_async_execution_time

class AnalysisRequest(BaseModel):
    """Request model for data analysis tasks."""
    data: Union[str, Dict[str, Any], List[Any]] = Field(..., description="Data to be analyzed. Can be raw text, JSON object, or list of data points.")
    analysis_type: str = Field(..., description="Type of analysis to perform (e.g., 'statistical', 'sentiment', 'trend', 'pattern').")
    instructions: Optional[str] = Field(None, description="Specific instructions for the analysis.")
    format: str = Field(default="text", description="Desired output format ('text', 'json', 'markdown').")

class AnalysisResponse(BaseModel):
    """Response model for data analysis tasks."""
    result: Any = Field(..., description="The analysis result. Format depends on the request format.")
    insights: List[str] = Field(default_factory=list, description="Key insights derived from the analysis.")
    confidence: Optional[float] = Field(None, description="Confidence score for the analysis, if applicable.")
    error: Optional[str] = Field(None, description="Error message if analysis failed.")

class AnalystAgent(BaseAgent):
    """
    Specialized agent for data analysis, pattern recognition, and insight generation.
    
    This agent is responsible for:
    1. Analyzing structured and unstructured data
    2. Identifying patterns, trends, and anomalies
    3. Providing statistical analysis and visualizations
    4. Drawing meaningful conclusions and actionable insights
    """
    
    def __init__(self, agent_name: str, config: Any, llm_interface: Any, memory_manager: Any):
        super().__init__(agent_name, config, llm_interface, memory_manager)
        self.logger = logging.getLogger(f"WITS.Agents.{agent_name}")
        self.logger.info(f"AnalystAgent initialized with model: {self.agent_config['model_name']}")
    
    @log_async_execution_time(logging.getLogger('WITS.Agents.AnalystAgent'))
    async def run(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute a data analysis task.
        
        Args:
            task_description: Description of the analysis task to perform
            context: Optional additional context including data to analyze
            
        Returns:
            str: Analysis results and insights
        """
        context = context or {}
        self.logger.info(f"Running AnalystAgent task: {task_description[:100]}{'...' if len(task_description) > 100 else ''}")
        
        # Prepare prompt for analysis
        system_prompt = """You are a specialized Analyst Agent in the WITS-NEXUS system.
Your purpose is to analyze data, identify patterns, and provide meaningful insights.
You should focus on accuracy, clarity, and actionable conclusions.
Present your analysis in a structured format with clear sections for methodology, findings, and recommendations.
Support your conclusions with evidence from the data whenever possible.
"""
        
        # Enhance context with relevant memory segments if available
        if self.memory:
            memory_segments = await self.memory.search(task_description, limit=3)
            if memory_segments:
                context["relevant_memories"] = [seg.content for seg in memory_segments]
        
        # Construct the full prompt with context
        user_prompt = f"Task: {task_description}\n\n"
        
        if "data" in context:
            # Format data based on type
            if isinstance(context["data"], (dict, list)):
                data_str = json.dumps(context["data"], indent=2)
                user_prompt += f"Data to analyze (in JSON format):\n```json\n{data_str}\n```\n\n"
            else:
                user_prompt += f"Data to analyze:\n{context['data']}\n\n"
        
        if "analysis_type" in context:
            user_prompt += f"Analysis type: {context['analysis_type']}\n"
            
        if "instructions" in context and context["instructions"]:
            user_prompt += f"Specific instructions: {context['instructions']}\n"
            
        if "format" in context:
            user_prompt += f"Output format: {context['format']}\n"
            
        if "relevant_memories" in context:
            user_prompt += "\nRelevant previous analyses:\n"
            for i, memory in enumerate(context["relevant_memories"]):
                user_prompt += f"[Reference {i+1}]: {memory}\n"
        
        # Generate analysis using LLM
        try:
            response = await self.llm.generate_text(
                prompt=user_prompt,
                model_name=self.agent_config["model_name"],
                system_prompt=system_prompt
            )
            
            # Store the result in memory
            if self.memory:
                memory_segment = MemorySegment(
                    type="AGENT_ANALYSIS",
                    source=self.agent_name,
                    content=response
                )
                await self.memory.add(memory_segment)
            
            self.logger.info("Analysis task completed successfully")
            return response
            
        except Exception as e:
            error_msg = f"Error performing analysis: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    async def analyze_data(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        Analyze data according to the specified analysis type and format.
        
        Args:
            request: AnalysisRequest with data and analysis specifications
            
        Returns:
            AnalysisResponse: The analysis results and insights
        """
        try:
            context = {
                "data": request.data,
                "analysis_type": request.analysis_type,
                "instructions": request.instructions,
                "format": request.format
            }
            
            task_description = f"Perform {request.analysis_type} analysis on the provided data."
            analysis_result = await self.run(task_description, context)
            
            # Extract insights from the analysis
            insights_prompt = f"""
Based on the following analysis, extract 3-5 key insights as a bulleted list:

{analysis_result}
"""
            
            insights_response = await self.llm.generate_text(
                prompt=insights_prompt,
                model_name=self.agent_config["model_name"]
            )
            
            # Parse insights into a list
            insights = [line.strip().replace('- ', '') for line in insights_response.split('\n') 
                       if line.strip() and line.strip().startswith('- ')]
            
            # Format result according to requested format
            if request.format == "json":
                try:
                    # First check if the result is already in JSON format
                    result = json.loads(analysis_result)
                except:
                    # If not, convert to a simple JSON structure
                    result = {"analysis": analysis_result}
            else:
                result = analysis_result
            
            return AnalysisResponse(
                result=result,
                insights=insights,
                confidence=0.85,  # Placeholder confidence score
                error=None
            )
            
        except Exception as e:
            error_msg = f"Error in analyze_data: {str(e)}"
            self.logger.error(error_msg)
            return AnalysisResponse(
                result="Error analyzing data",
                insights=["Analysis failed due to an error"],
                confidence=0.0,
                error=error_msg
            )
    
    async def identify_patterns(self, data: Union[List[Any], str], pattern_type: str = "general") -> Dict[str, Any]:
        """
        Identify patterns in the provided data.
        
        Args:
            data: Data in which to search for patterns (list or text)
            pattern_type: Type of patterns to look for ("general", "temporal", "statistical", etc.)
            
        Returns:
            Dict[str, Any]: Dictionary containing identified patterns
        """
        request = AnalysisRequest(
            data=data,
            analysis_type=f"pattern_{pattern_type}",
            instructions=f"Identify {pattern_type} patterns in the data and describe each one",
            format="json"
        )
        
        response = await self.analyze_data(request)
        
        if isinstance(response.result, dict):
            return response.result
        else:
            return {"patterns": response.result, "insights": response.insights}
