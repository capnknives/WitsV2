# tools/web_search_tool.py
from typing import ClassVar, Type, Dict, Any, List, Optional
from pydantic import BaseModel, Field
import aiohttp
import json
import re
import asyncio
from .base_tool import BaseTool

class WebSearchArgs(BaseModel):
    """Arguments for the WebSearchTool."""
    query: str = Field(..., description="The search query to execute.")
    num_results: int = Field(5, description="Number of search results to return. Defaults to 5.")
    include_snippets: bool = Field(True, description="Whether to include snippet text in results. Defaults to True.")

class WebSearchResult(BaseModel):
    """A single search result."""
    title: str = Field(..., description="Title of the search result.")
    url: str = Field(..., description="URL of the search result.")
    snippet: Optional[str] = Field(None, description="Snippet or description of the search result.")

class WebSearchResponse(BaseModel):
    """Response from the WebSearchTool."""
    results: List[WebSearchResult] = Field([], description="List of search results.")
    total_results_found: int = Field(0, description="Total number of results found.")
    query: str = Field(..., description="The original search query.")
    error: Optional[str] = Field(None, description="Error message if search failed.")

class WebSearchTool(BaseTool):
    """
    Tool for performing web searches using a public search API.
    
    This tool allows the agent to search the web for information that it doesn't have
    or that might have changed since its training data cutoff.
    """
    
    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = "Search the web for current information. Use this when you need to find information about current events, trending topics, specific facts that might have changed since your knowledge cutoff, or anything else you don't have definitive information about."
    args_schema: ClassVar[Type[BaseModel]] = WebSearchArgs
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the web search tool with configuration."""
        super().__init__()
        self.config = config or {}
        self.search_api_url = self.config.get('search_api_url', 'https://ddg-api.herokuapp.com/search')
        self.enabled = self.config.get('internet_access', True)
        
        if not self.enabled:
            print(f"[{self.name}] WARNING: Internet access is disabled in configuration. Web search will return only a notification.")
    
    async def execute(self, args: WebSearchArgs) -> WebSearchResponse:
        """
        Execute a web search with the given arguments.
        
        Args:
            args: WebSearchArgs containing the query and configuration
            
        Returns:
            WebSearchResponse: The search results or error
        """
        if not self.enabled:
            return WebSearchResponse(
                results=[],
                total_results_found=0,
                query=args.query,
                error="Internet access is disabled in system configuration."
            )
        
        query = args.query.strip()
        if not query:
            return WebSearchResponse(
                results=[],
                total_results_found=0,
                query=query,
                error="Search query cannot be empty."
            )
        
        try:
            # Create a timeout context for the request
            timeout = aiohttp.ClientTimeout(total=15)  # 15 seconds timeout
            
            # Execute the search request
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    self.search_api_url, 
                    params={'q': query, 'max_results': args.num_results}
                ) as response:
                    
                    if response.status != 200:
                        return WebSearchResponse(
                            results=[],
                            total_results_found=0,
                            query=query,
                            error=f"Search API returned status code {response.status}."
                        )
                    
                    # Parse the JSON response
                    result_data = await response.json()
                    
                    # Check for error in API response
                    if isinstance(result_data, dict) and result_data.get('error'):
                        return WebSearchResponse(
                            results=[],
                            total_results_found=0,
                            query=query,
                            error=f"Search API error: {result_data.get('error')}"
                        )
                    
                    # Handle different response formats
                    results = []
                    if isinstance(result_data, list):
                        # Standard format: list of results
                        for item in result_data[:args.num_results]:
                            result = WebSearchResult(
                                title=item.get('title', 'No title'),
                                url=item.get('href', ''),
                                snippet=item.get('body', '') if args.include_snippets else None
                            )
                            results.append(result)
                    elif isinstance(result_data, dict) and 'results' in result_data:
                        # Alternative format: { results: [...] }
                        for item in result_data['results'][:args.num_results]:
                            result = WebSearchResult(
                                title=item.get('title', 'No title'),
                                url=item.get('link', item.get('url', '')),
                                snippet=item.get('snippet', '') if args.include_snippets else None
                            )
                            results.append(result)
                    
                    # Build final response
                    return WebSearchResponse(
                        results=results,
                        total_results_found=len(result_data) if isinstance(result_data, list) else 
                                          len(result_data.get('results', [])),
                        query=query
                    )
        
        except asyncio.TimeoutError:
            return WebSearchResponse(
                results=[],
                total_results_found=0,
                query=query,
                error="Search request timed out after 15 seconds."
            )
        except Exception as e:
            return WebSearchResponse(
                results=[],
                total_results_found=0,
                query=query,
                error=f"Error during web search: {str(e)}"
            )
