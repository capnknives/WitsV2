# tools/web_search_tool.py
from typing import ClassVar, Type, Dict, Any, List, Optional
from pydantic import BaseModel, Field
import aiohttp
import json
import re
import asyncio
from .base_tool import BaseTool

# Search the interwebs like it's 2007! \\o/

class WebSearchArgs(BaseModel):
    """The stuff we need to explore the digital wilderness! ^_^"""
    query: str = Field(..., description="What are we searching for today? =D")
    num_results: int = Field(5, description="How many results do you want? (Default: 5, cuz who reads more? >.>)")
    include_snippets: bool = Field(True, description="Want little previews? Of course you do! \\o/")

class WebSearchResult(BaseModel):
    """One shiny result from our web expedition! ^_^"""
    title: str = Field(..., description="The clickbait- I mean, professional title! =P")
    url: str = Field(..., description="Where this treasure can be found! \\o/")
    snippet: Optional[str] = Field(None, description="A teeny tiny preview (if you asked for it ^_^)")

class WebSearchResponse(BaseModel):
    """Everything we found in our internet adventures! O.o"""
    results: List[WebSearchResult] = Field([], description="All the cool stuff we found! =D")
    total_results_found: int = Field(0, description="How many results we stumbled upon x.x")
    query: str = Field(..., description="What you asked us to look for \\o/")
    error: Optional[str] = Field(None, description="Oopsie alerts! (hopefully none >.>)")

class WebSearchTool(BaseTool):
    """
    Your personal internet explorer! (not *that* Internet Explorer lol)
    
    I'm like a search engine with personality! \\o/
    Need current info? I gotchu! =D
    Breaking news? On it! ^_^
    Random facts? That's my jam! \\o/
    
    Just don't ask me about the meaning of life... 
    that query takes too long to process x.x
    """
    
    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = "Search the web for current information. Use this when you need to find information about current events, trending topics, specific facts that might have changed since your knowledge cutoff, or anything else you don't have definitive information about."
    args_schema: ClassVar[Type[BaseModel]] = WebSearchArgs
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the web search tool with configuration. Time to power up! \\o/"""
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
