# core/memory_manager.py - PART 1
from typing import List, Dict, Any, Optional, Set, Union, Tuple
from datetime import datetime
import json
import os
import uuid
import time
import re
from pydantic import BaseModel
import aiofiles

# Import for vector embeddings and search
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    print("[MemoryManager] Warning: sentence-transformers or faiss not available. Vector search will be disabled.")
    VECTOR_SEARCH_AVAILABLE = False
    SentenceTransformer = None
    np = None
    faiss = None

from .schemas import MemorySegment, MemorySegmentContent

class MemoryManager:
    """
    Comprehensive memory system for WITS-NEXUS v2.
    Features:
    - Stores memory segments with their embeddings
    - Supports goal tracking and management
    - Vector-based semantic search
    - Memory pruning and persistence
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

    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
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

    def semantic_search(self, query: str, limit: int = 5, 
                       segment_type_filter: Optional[str] = None) -> List[MemorySegment]:
        """
        Search for semantically similar segments.
        
        Args:
            query: The search query text
            limit: Maximum number of results to return
            segment_type_filter: Optional filter by segment type
            
        Returns:
            List[MemorySegment]: List of relevant memory segments
        """
        if not self.embedding_model or not self.index or self.index.ntotal == 0:
            print("[MemoryManager] Semantic search unavailable (no model, empty index)")
            return []
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        if query_embedding is None:
            return []
        
        # Search in FAISS index
        k_search = min(limit * 3, self.index.ntotal)  # Search for more initially to allow filtering
        if k_search == 0:
            return []
        
        distances, faiss_indices = self.index.search(query_embedding.reshape(1, -1), k_search)
        
        results: List[MemorySegment] = []
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
                        segment.metadata["search_distance"] = float(distances[0][i])
                        results.append(segment)
                    break
            
            if len(results) >= limit:
                break
        
        return results

    def remember_agent_output(self, agent_name: str, output: str) -> Optional[str]:
        """Store the most recent output from a specific agent."""
        if not agent_name:
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