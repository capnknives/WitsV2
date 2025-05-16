# core/memory_manager.py - PART 1
from typing import List, Dict, Any, Optional, Set, Union, Tuple
from datetime import datetime
import json
import os
import uuid
import time
import re
import logging
from pydantic import BaseModel, Field
import aiofiles

from .debug_utils import log_execution_time, log_async_execution_time, DebugInfo, log_debug_info, PerformanceMonitor

# Import for vector embeddings and search
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss
    VECTOR_SEARCH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Vector search dependencies not available ({str(e)}). Vector search will be disabled.")
    VECTOR_SEARCH_AVAILABLE = False
    np = None
    SentenceTransformer = None
    faiss = None

from .schemas import MemorySegment, MemorySegmentContent

# Debug model for memory operations
class MemoryDebugInfo(BaseModel):
    """Debug information specific to memory operations."""
    operation_type: str = Field(..., description="Type of memory operation (add, search, etc.)")
    segment_count: int = Field(0, description="Number of segments involved")
    vector_search_used: bool = Field(False, description="Whether vector search was used")
    query_length: Optional[int] = Field(None, description="Length of search query if applicable")
    match_count: Optional[int] = Field(None, description="Number of matches found if applicable")
    embedding_time_ms: Optional[float] = Field(None, description="Time taken for embedding generation")
    search_time_ms: Optional[float] = Field(None, description="Time taken for search operation")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional operation details")

class MemoryManager:
    """
    Comprehensive memory system for WITS-NEXUS v2.
    Features:
    - Stores memory segments with their embeddings
    - Supports goal tracking and management
    - Vector-based semantic search
    - Memory pruning and persistence
    - Performance monitoring and debugging
    """
    
    def __init__(self, config: Any, memory_file_path: Optional[str] = None):
        """Initialize the MemoryManager with configuration and optional file path."""
        self.config = config
        self.memory_file = memory_file_path or config.memory_manager.memory_file_path
        self.max_segments = 1000  # Default max segments
        
        # Main storage containers
        self.segments: List[MemorySegment] = []
        self.goals: List[Dict[str, Any]] = []
        self.completed_goals: List[Dict[str, Any]] = []
        self.last_agent_output: Dict[str, str] = {}
        self.last_agent_name: Optional[str] = None
        
        # Set up logging
        self.logger = logging.getLogger('WITS.MemoryManager')
        
        # Debug configuration
        self.debug_enabled = config.debug.enabled if hasattr(config, 'debug') else False
        if self.debug_enabled and hasattr(config.debug, 'components'):
            self.debug_config = config.debug.components.memory_manager
        else:
            self.debug_config = None
            
        self.performance_monitor = PerformanceMonitor("MemoryManager")
        
        # FAISS and embedding setup
        self.vector_model_name = self.config.memory_manager.vector_model
        print(f"[MemoryManager] Loading embedding model: {self.vector_model_name}")
        
        try:
            self.embedding_model = SentenceTransformer(self.vector_model_name)
            self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"[MemoryManager] Vector dimension: {self.vector_dim}")
            
            # Initialize FAISS index
            self.index = faiss.IndexFlatL2(self.vector_dim)
            
            # Mappings between segment IDs and FAISS indices
            self.id_to_faiss_idx: Dict[str, int] = {}
            self.faiss_idx_to_id: List[str] = []
            
        except Exception as e:
            print(f"[MemoryManager_ERROR] Failed to initialize embedding model: {e}")
            self.embedding_model = None
            self.index = None
            self.vector_dim = 384  # Default fallback dimension
        
        # Memory will be loaded in initialize_db_async

    def _generate_embedding(self, text: str) -> Optional[Any]:
        """Generate an embedding for the given text."""
        if not self.embedding_model:
            return None
        
        try:
            # Generate embedding and convert to numpy array
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"[MemoryManager] Error generating embedding: {e}")
            return None

    def add_segment(self, 
                   segment_type: str, 
                   content_text: Optional[str] = None,
                   source: Optional[str] = None, 
                   metadata: Optional[Dict[str, Any]] = None,
                   tool_name: Optional[str] = None, 
                   tool_args: Optional[Dict[str, Any]] = None,
                   tool_output: Optional[str] = None,
                   importance: float = 0.5) -> str:
        """
        Add a new memory segment with appropriate content types.
        
        Args:
            segment_type: Type of memory segment (e.g., "USER_GOAL", "LLM_THOUGHT")
            content_text: Optional text content
            source: Source of the segment (e.g., agent name, "USER")
            metadata: Additional metadata for the segment
            tool_name: Name of tool if this segment is a tool call or result
            tool_args: Arguments passed to the tool
            tool_output: Output from the tool execution
            importance: Importance score (0.0 to 1.0) for memory retention
            
        Returns:
            str: ID of the created segment
        """
        # Create segment content
        segment_content = MemorySegmentContent(
            text=content_text,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_output=tool_output
        )
        
        # Create full segment
        segment = MemorySegment(
            type=segment_type,
            source=source or "unknown",
            content=segment_content,
            metadata=metadata or {},
            importance=importance
        )
        
        # Generate embedding if we have an embedding model and valid text content
        embedding_text = content_text or tool_output or ""
        if embedding_text and self.embedding_model:
            segment.embedding = self._generate_embedding(embedding_text)
        
        # Add to memory
        self.segments.append(segment)
        
        # Add to FAISS index if we have an embedding
        if self.index and segment.embedding is not None:
            current_idx = self.index.ntotal
            self.index.add(np.array([segment.embedding]))
            self.id_to_faiss_idx[segment.id] = current_idx
            self.faiss_idx_to_id.append(segment.id)
        
        # Save after adding 
        self._save_memory()
        
        print(f"[MemoryManager] Added segment: ID={segment.id}, Type='{segment.type}', Source='{segment.source}'")
        return segment.id

    def get_formatted_history(self, limit: int = 10, for_llm: bool = True) -> str:
        """
        Get recent history formatted for LLM prompts.
        
        Args:
            limit: Maximum number of segments to retrieve
            for_llm: Whether formatting is for LLM consumption
            
        Returns:
            str: Formatted history string
        """
        relevant_history = self.segments[-limit:] if len(self.segments) > 0 else []
        formatted_lines = []
        
        for seg in relevant_history:
            # Format timestamp
            timestamp_str = seg.timestamp.strftime('%Y-%m-%d %H:%M')
            
            # Start the line with timestamp, type, and source
            line = f"[{timestamp_str}] ({seg.type} from {seg.source}): "
            
            # Add content based on what's available
            if seg.content.text:
                line += seg.content.text
            elif seg.content.tool_name:
                args_str = json.dumps(seg.content.tool_args) if seg.content.tool_args else "{}"
                line += f"Tool Call: {seg.content.tool_name}, Args: {args_str}"
                if seg.content.tool_output:
                    # For tool output, keep it concise for prompt
                    output_preview = seg.content.tool_output
                    if len(output_preview) > 200:
                        output_preview = output_preview[:197] + "..."
                    line += f" → Output: {output_preview}"
            
            formatted_lines.append(line)
        
        if not formatted_lines:
            return "No relevant history."
        
        return "\n".join(formatted_lines)

    @log_execution_time(logging.getLogger('WITS.MemoryManager'))
    def semantic_search(self, query: str, limit: int = 5, 
                       segment_type_filter: Optional[str] = None) -> List[MemorySegment]:
        """
        Search for semantically similar segments with performance tracking.
        
        Args:
            query: The search query text
            limit: Maximum number of results to return
            segment_type_filter: Optional filter by segment type
            
        Returns:
            List[MemorySegment]: List of relevant memory segments
        """
        start_time = time.time()
        search_debug = MemoryDebugInfo(
            operation_type="semantic_search",
            segment_count=len(self.segments),
            vector_search_used=False,
            query_length=len(query)
        )
        
        if not self.embedding_model or not self.index or self.index.ntotal == 0:
            self.logger.warning("Semantic search unavailable (no model, empty index)")
            
            if self.debug_enabled:
                debug_info = DebugInfo(
                    timestamp=datetime.now().isoformat(),
                    component="MemoryManager",
                    action="semantic_search_failed",
                    details={
                        "reason": "vector search unavailable",
                        "has_model": self.embedding_model is not None,
                        "has_index": self.index is not None,
                        "index_size": self.index.ntotal if self.index else 0
                    },
                    duration_ms=0.0,
                    success=False,
                    error="Vector search components not available"
                )
                log_debug_info(self.logger, debug_info)
            return []
        
        # Track embedding generation time
        embed_start = time.time()
        query_embedding = self._generate_embedding(query)
        embed_time = (time.time() - embed_start) * 1000  # ms
        search_debug.embedding_time_ms = embed_time
        
        if query_embedding is None:
            self.logger.warning("Query embedding generation failed")
            if self.debug_enabled:
                debug_info = DebugInfo(
                    timestamp=datetime.now().isoformat(),
                    component="MemoryManager",
                    action="semantic_search_failed",
                    details={"reason": "embedding generation failed", "query": query[:100]},
                    duration_ms=(time.time() - start_time) * 1000,
                    success=False,
                    error="Failed to generate embedding"
                )
                log_debug_info(self.logger, debug_info)
            return []
        
        # Track search time
        search_start = time.time()
        k_search = min(limit * 3, self.index.ntotal)  # Search for more initially to allow filtering
        if k_search == 0:
            return []
        
        # Update debug info - we're actually using vector search now
        search_debug.vector_search_used = True
        
        # Perform the search
        distances, faiss_indices = self.index.search(query_embedding.reshape(1, -1), k_search)
        search_time = (time.time() - search_start) * 1000  # ms
        search_debug.search_time_ms = search_time
        
        # Process results
        results: List[MemorySegment] = []
        matched_segments = set()
        
        for i, faiss_idx in enumerate(faiss_indices[0]):
            if faiss_idx < 0 or faiss_idx >= len(self.faiss_idx_to_id):
                continue
            
            segment_id = self.faiss_idx_to_id[faiss_idx]
            
            # Find the segment by ID
            for segment in self.segments:
                if segment.id == segment_id:
                    # Apply type filter if specified
                    if segment_type_filter is None or segment.type == segment_type_filter:
                        # Store distance as metadata for debugging/visualization
                        if not hasattr(segment, 'metadata'):
                            segment.metadata = {}
                        segment.metadata["search_distance"] = float(distances[0][i])
                        results.append(segment)
                        matched_segments.add(segment_id)
                    break
            
            if len(results) >= limit:
                break
                
        # Log debug info
        search_debug.match_count = len(results)
        search_debug.details = {
            "filter_applied": segment_type_filter is not None,
            "filter_type": segment_type_filter,
            "requested_limit": limit,
            "search_k": k_search
        }
        
        total_time = (time.time() - start_time) * 1000
        
        # Log complete debug info
        if self.debug_enabled and self.debug_config and getattr(self.debug_config, 'log_searches', False):
            debug_info = DebugInfo(
                timestamp=datetime.now().isoformat(),
                component="MemoryManager",
                action="semantic_search",
                details={
                    "query": query[:100] + "..." if len(query) > 100 else query,
                    "segment_type_filter": segment_type_filter,
                    "limit": limit,
                    "found_count": len(results),
                    "embedding_time_ms": search_debug.embedding_time_ms,
                    "search_time_ms": search_debug.search_time_ms,
                    "total_time_ms": total_time
                },
                duration_ms=total_time,
                success=True
            )
            log_debug_info(self.logger, debug_info)
            
            # Log performance data if enabled
            if getattr(self.debug_config, 'log_performance', False):
                self.logger.debug(
                    f"Performance: semantic_search - "
                    f"Embedding: {search_debug.embedding_time_ms:.2f}ms, "
                    f"Search: {search_debug.search_time_ms:.2f}ms, "
                    f"Total: {total_time:.2f}ms, "
                    f"Results: {len(results)}/{k_search}"
                )
                
        return results

    @log_execution_time(logging.getLogger('WITS.MemoryManager'))
    def remember_agent_output(self, agent_name: str, output: str) -> Optional[str]:
        """Store the most recent output from a specific agent."""
        if not agent_name:
            if self.debug_enabled:
                self.logger.warning("Attempted to remember agent output with no agent name")
            return None
        
        agent_name_lower = agent_name.lower()
        self.last_agent_output[agent_name_lower] = output
        self.last_agent_name = agent_name_lower
        
        # Also save it as a memory segment
        segment_id = self.add_segment(
            segment_type="AGENT_OUTPUT",
            content_text=output,
            source=agent_name_lower,
            importance=0.6  # Medium-high importance
        )
        
        return segment_id

    def recall_agent_output(self, agent_name: str) -> Optional[str]:
        """Retrieve the most recent output from a specific agent."""
        if not agent_name:
            return None
        
        return self.last_agent_output.get(agent_name.lower())

    def get_last_agent(self) -> Optional[str]:
        """Get the name of the last agent that provided output."""
        return self.last_agent_name

    async def initialize_db_async(self):
        """Initialize the database asynchronously."""
        try:
            if os.path.exists(self.memory_file):
                async with aiofiles.open(self.memory_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content)
                    # Load segments
                    for segment_data in data.get('segments', []):
                        segment = MemorySegment(**segment_data)
                        self.segments.append(segment)
                        # Add to FAISS index if we have embeddings
                        if hasattr(segment, 'embedding') and segment.embedding is not None:
                            vector = np.array(segment.embedding).astype(np.float32).reshape(1, -1)
                            self.index.add(vector)
                            self.id_to_faiss_idx[segment.id] = len(self.faiss_idx_to_id)
                            self.faiss_idx_to_id.append(segment.id)
                    print(f"[MemoryManager] Loaded {len(self.segments)} segments from {self.memory_file}")
            else:
                print(f"[MemoryManager] No memory file found at {self.memory_file}. Starting with empty memory.")
                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
                
        except Exception as e:
            print(f"[MemoryManager_ERROR] Error loading memory: {e}")
            # Start with empty memory
            self.segments = []
            self.id_to_faiss_idx = {}