# core/memory_manager.py
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
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss  # Keep faiss import for type hints if needed by other parts, or for clarity
import torch

from .debug_utils import log_execution_time, log_async_execution_time, DebugInfo, log_debug_info, PerformanceMonitor
from .faiss_utils import initialize_gpu_resources, create_gpu_index, add_vectors, search_vectors
from .schemas import MemorySegment, MemorySegmentContent # Ensure schemas are imported

# Initialize GPU resources
try:
    RES, CUDA_DEVICE = initialize_gpu_resources()
    VECTOR_SEARCH_AVAILABLE = True
    VECTOR_SEARCH_GPU = True  # Explicitly state GPU is used
    logger = logging.getLogger('WITS.MemoryManager')
    logger.info("Successfully initialized GPU resources for FAISS.")
except Exception as e:
    # Log the error and re-raise to ensure application fails if GPU is not available
    logging.getLogger('WITS.MemoryManager').error(f"CRITICAL: Failed to initialize GPU support: {str(e)}. GPU support is required.")
    raise RuntimeError(f"Failed to initialize GPU support: {str(e)}. GPU support is required.") from e

class MemoryManager:
    """
    Comprehensive memory system for WITS-NEXUS v2 with GPU-accelerated vector operations.
    Features:
    - GPU-accelerated vector storage and search using FAISS
    - Stores memory segments with their embeddings
    - Supports goal tracking and management
    - Memory pruning and persistence
    - Performance monitoring and debugging
    """
    
    def __init__(self, config: Any, memory_file_path: Optional[str] = None):
        """Initialize the MemoryManager with configuration."""
        # If config is AppConfig, access config.memory_manager
        # If config is MemoryManagerConfig, access config directly
        if hasattr(config, 'memory_manager') and isinstance(getattr(config, 'memory_manager'), BaseModel):
            self.resolved_memory_config = config.memory_manager
            self.debug_config_source = config.debug if hasattr(config, 'debug') else None
        else:
            # Assuming config is already MemoryManagerConfig or a compatible structure
            self.resolved_memory_config = config
            # Try to get debug config from a potential parent AppConfig if passed differently or not available
            # This part might need adjustment based on how AppConfig is structured if config is just MemoryManagerConfig
            self.debug_config_source = getattr(config, '_parent_debug_config', None) # Placeholder for actual debug config access

        self.memory_file = memory_file_path or self.resolved_memory_config.memory_file_path
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
        self.debug_enabled = False
        self.debug_component_config = None # Renamed from self.debug_config to avoid confusion

        if self.debug_config_source and hasattr(self.debug_config_source, 'enabled'):
            self.debug_enabled = self.debug_config_source.enabled
            if self.debug_enabled and hasattr(self.debug_config_source, 'components') and hasattr(self.debug_config_source.components, 'memory_manager'):
                self.debug_component_config = self.debug_config_source.components.memory_manager
            
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor("MemoryManager")
        
        # FAISS and embedding setup
        self.vector_model_name = getattr(self.resolved_memory_config, 'vector_model', "all-MiniLM-L6-v2") # Use resolved_memory_config
        self.embedding_model = None
        self.index: Optional[faiss.Index] = None # Generic type hint for FAISS index
        self.vector_dim: Optional[int] = 384  # Default fallback dimension
        self.id_to_faiss_idx: Dict[str, int] = {}
        self.faiss_idx_to_id: List[str] = []
        
        # Initialize vector search (which includes FAISS index creation)
        self._initialize_vector_search()
    
    def _initialize_vector_search(self):
        """Initialize vector search capabilities with GPU support."""
        if not VECTOR_SEARCH_AVAILABLE:
            self.logger.warning("Vector search is not available. Skipping initialization.")
            return

        if not self.vector_model_name:
            self.logger.warning("No vector model name configured. Skipping vector search initialization.")
            return

        self.logger.info(f"Initializing vector search. Loading embedding model: {self.vector_model_name}")
        try:
            self.embedding_model = SentenceTransformer(self.vector_model_name)
            self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.logger.info(f"Vector dimension set to: {self.vector_dim}")
            
            if not isinstance(self.vector_dim, int):
                 raise ValueError(f"Vector dimension must be an integer, got {self.vector_dim}")

            # Initialize FAISS index with GPU support
            # RES and CUDA_DEVICE are globally available from the initial GPU check
            self.index = create_gpu_index(self.vector_dim, RES, CUDA_DEVICE)
            self.logger.info(f"Successfully created GPU-enabled FAISS index on device {CUDA_DEVICE}.")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model or FAISS index: {e}", exc_info=True)
            self.embedding_model = None
            self.index = None
            # Re-raise as this is critical for GPU-dependent operation
            raise RuntimeError(f"Failed to initialize vector search components: {e}") from e

    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate an embedding for the given text."""
        if not self.embedding_model:
            self.logger.warning("Embedding model not available, cannot generate embedding.")
            return None
        
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32) if embedding is not None else None
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}", exc_info=True)
            return None

    def _add_to_index(self, vector: np.ndarray, segment_id: str) -> bool:
        """Add a vector to the FAISS index."""
        if self.index is None:
            self.logger.error("FAISS index is not initialized. Cannot add vector.")
            return False
        try:
            # add_vectors expects a 2D array
            vector_2d = vector.reshape(1, -1) if vector.ndim == 1 else vector
            faiss_idx = add_vectors(self.index, vector_2d) # faiss_utils.add_vectors
            
            # Assuming add_vectors returns the index of the *first* added vector if multiple were added,
            # or handles single vector addition appropriately.
            # For a single vector, self.index.ntotal-1 would be its index.
            # Let's assume faiss_utils.add_vectors returns the correct internal FAISS index.
            # If add_vectors returns the new ntotal, then the index is ntotal -1.
            # The current faiss_utils.add_vectors returns self.index.ntotal -1, which is the last added index.
            
            actual_faiss_id = self.index.ntotal - 1 # Get the actual index in FAISS
            self.id_to_faiss_idx[segment_id] = actual_faiss_id
            # Ensure faiss_idx_to_id has placeholders up to actual_faiss_id if needed,
            # or simply append if it's always sequential.
            # For simplicity, assuming sequential appends match FAISS internal IDs.
            # This needs careful handling if FAISS IDs are not simple appends.
            # If faiss_idx_to_id is a direct map, its length should be ntotal.
            while len(self.faiss_idx_to_id) <= actual_faiss_id:
                self.faiss_idx_to_id.append("") # Placeholder or handle more robustly
            self.faiss_idx_to_id[actual_faiss_id] = segment_id
            
            self.logger.debug(f"Added vector for segment {segment_id} to FAISS index at internal id {actual_faiss_id}.")
            return True
        except Exception as e:
            self.logger.error(f"Error adding vector for segment {segment_id} to index: {e}", exc_info=True)
        return False

    def _search_similar(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[float, str]]:
        """Search for similar vectors in the index."""
        if self.index is None:
            self.logger.error("FAISS index is not initialized. Cannot search.")
            return []
        try:
            # search_vectors expects a 2D array for queries
            query_vector_2d = query_vector.reshape(1, -1) if query_vector.ndim == 1 else query_vector
            distances, indices = search_vectors(self.index, query_vector_2d, k) # faiss_utils.search_vectors
            
            results = []
            # search_vectors from faiss_utils returns distances[0], indices[0] for a single query
            for dist, idx in zip(distances, indices):
                if idx != -1 and idx < len(self.faiss_idx_to_id): # FAISS returns -1 for no/invalid neighbors
                    segment_id = self.faiss_idx_to_id[idx]
                    results.append((float(dist), segment_id))
                else:
                    self.logger.warning(f"Invalid index {idx} returned from FAISS search.")
            return results
        except Exception as e:
            self.logger.error(f"Error searching vectors: {e}", exc_info=True)
        return []

    async def add_segment(self, 
                   segment_type: str, 
                   content_text: Optional[str] = None,
                   source: Optional[str] = None, 
                   tool_name: Optional[str] = None, 
                   tool_args: Optional[Dict[str, Any]] = None,
                   tool_output: Optional[str] = None,
                   importance: float = 0.5,
                   meta: Optional[Dict[str, Any]] = None) -> str:
        """Add a new memory segment with appropriate content types."""
        segment_content = MemorySegmentContent(
            text=content_text,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_output=tool_output
        )
        
        segment = MemorySegment(
            type=segment_type,
            source=source or "unknown",
            content=segment_content,
            metadata=meta or {},
            importance=importance
        )
        
        embedding_text = content_text or tool_output or ""
        if embedding_text and self.embedding_model and self.index is not None:
            embedding = self._generate_embedding(embedding_text)
            if embedding is not None:
                segment.embedding = embedding.tolist() # Store as list in MemorySegment
                self._add_to_index(embedding, segment.id)
        
        self.segments.append(segment)
        
        try:
            await self._save_to_disk()
        except Exception as e:
            self.logger.error(f"Error saving memory to disk: {e}", exc_info=True)
        
        self.logger.info(f"Added segment: ID={segment.id}, Type='{segment.type}', Source='{segment.source}'")
        return segment.id

    async def _save_to_disk(self):
        """Save memory segments to disk asynchronously."""
        if not self.memory_file:
            self.logger.warning("Memory file path not set. Cannot save to disk.")
            return
            
        try:
            # Prepare data for JSON serialization, converting datetime to ISO format string
            save_data = []
            for seg in self.segments:
                seg_dict = seg.model_dump(exclude_none=True)
                if 'timestamp' in seg_dict and isinstance(seg_dict['timestamp'], datetime):
                    seg_dict['timestamp'] = seg_dict['timestamp'].isoformat()
                save_data.append(seg_dict)
            
            async with aiofiles.open(self.memory_file, 'w') as f:
                await f.write(json.dumps(save_data, indent=2))
            self.logger.debug(f"Memory saved to {self.memory_file}")
        except Exception as e:
            self.logger.error(f"Error saving memory to disk: {e}", exc_info=True)
    
    def _check_vector_search_status(self) -> Dict[str, Any]:
        """Check the status of vector search capabilities."""
        status = {
            "vector_search_available": VECTOR_SEARCH_AVAILABLE,
            "gpu_enabled": VECTOR_SEARCH_GPU, # This is now a top-level constant
            "embedding_model": self.vector_model_name if self.embedding_model else None,
            "vector_dimension": self.vector_dim if self.embedding_model else None,
            "index_type": type(self.index).__name__ if self.index else None,
            "total_vectors": self.index.ntotal if self.index and hasattr(self.index, 'ntotal') else 0,
        }
        
        if self.debug_enabled:
            debug_info = DebugInfo(
                timestamp=datetime.now().isoformat(),
                component="MemoryManager",
                action="check_vector_search_status",
                details=status,
                duration_ms=0, # This is a status check, not a timed operation
                success=True
            )
            log_debug_info(self.logger, debug_info)
        
        return status

    async def initialize_db(self):
        """Initialize the memory database from disk and rebuild FAISS index."""
        self.logger.info("Initializing memory database...")
        try:
            if os.path.exists(self.memory_file):
                async with aiofiles.open(self.memory_file, 'r') as f:
                    content = await f.read()
                    if not content:
                        self.logger.info(f"Memory file {self.memory_file} is empty. Starting with no segments.")
                        self.segments = []
                    else:
                        loaded_segments_data = json.loads(content)
                        self.segments = [MemorySegment(**seg_data) for seg_data in loaded_segments_data]
                
                self.logger.info(f"Loaded {len(self.segments)} segments from {self.memory_file}")
                
                # Rebuild FAISS index if it's initialized and segments were loaded
                if self.index is not None and self.segments:
                    self.logger.info("Rebuilding FAISS index from loaded segments...")
                    # Clear existing index mappings before rebuilding
                    self.id_to_faiss_idx.clear()
                    self.faiss_idx_to_id.clear()
                    # It's safer to create a new index instance or reset the existing one
                    if isinstance(self.vector_dim, int):
                         self.index = create_gpu_index(self.vector_dim, RES, CUDA_DEVICE) # Recreate/reset
                    else:
                        self.logger.error("Cannot rebuild FAISS index: vector_dim is not an integer.")
                        # Potentially raise an error or handle as a critical failure
                        return


                    for segment in self.segments:
                        if segment.embedding: # If embedding was saved
                            embedding_np = np.array(segment.embedding, dtype=np.float32)
                            self._add_to_index(embedding_np, segment.id)
                        else: # If embedding needs to be regenerated
                            embedding_text = segment.content.text or segment.content.tool_output or ""
                            if embedding_text and self.embedding_model:
                                embedding_np = self._generate_embedding(embedding_text)
                                if embedding_np is not None:
                                    segment.embedding = embedding_np.tolist()
                                    self._add_to_index(embedding_np, segment.id)
                    self.logger.info(f"FAISS index rebuilt with {self.index.ntotal if self.index else 0} vectors.")
            else:
                self.logger.info(f"No existing memory file found at {self.memory_file}. Creating directory if needed.")
                os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            status = self._check_vector_search_status()
            self.logger.info(f"Memory database initialization complete. Vector search status: {json.dumps(status, indent=2)}")
        
        except Exception as e:
            self.logger.critical(f"Failed to initialize memory database: {e}", exc_info=True)
            # This is a critical failure, re-raise to stop application if memory can't init
            raise RuntimeError(f"Failed to initialize memory database: {e}") from e

    async def search_memory(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search memory for segments similar to the query text."""
        self._search_start_time = time.time() # Capture start time for duration calculation

        if not self.embedding_model or not self.index:
            self.logger.warning("Embedding model or FAISS index not available for search.")
            return []

        query_embedding = self._generate_embedding(query_text)
        if query_embedding is None:
            self.logger.warning(f"Could not generate embedding for query: {query_text}")
            return []

        search_results_tuples = self._search_similar(query_embedding, k)
        
        # Retrieve full segment data for search results
        final_results = []
        for dist, segment_id in search_results_tuples:
            segment = next((s for s in self.segments if s.id == segment_id), None)
            if segment:
                final_results.append({
                    "segment_id": segment.id,
                    "score": 1 - (dist / (self.vector_dim if self.vector_dim and self.vector_dim > 0 else 1)), # Normalize L2 to similarity
                    "content": segment.content.model_dump(),
                    "source": segment.source,
                    "type": segment.type,
                    "timestamp": segment.timestamp.isoformat(),
                    "importance": segment.importance
                })
            else:
                self.logger.warning(f"Segment ID {segment_id} from search result not found in memory segments.")
        
        # Sort by score descending
        final_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Calculate duration for debug log
        end_time = time.time() # Assuming start_time was captured at the beginning of the method
        # Need to add start_time capture if not already present

        if self.debug_enabled:
            # Ensure start_time is captured at the beginning of the search_memory method
            # For now, let's assume it was, or set a placeholder if it wasn't.
            # Placeholder for duration if start_time is not available here:
            duration_ms = (end_time - getattr(self, '_search_start_time', end_time)) * 1000

            log_debug_info(self.logger, DebugInfo(
                timestamp=datetime.now().isoformat(),
                component="MemoryManager",
                action="search_memory",
                details={"query": query_text, "k": k, "num_results": len(final_results)},
                duration_ms=duration_ms, # Added duration_ms
                success=True 
            ))
        return final_results

    async def add_memory_segment(self, segment: MemorySegment):
        """Adds a new memory segment to the list and persists to file."""
        # Ensure metadata is handled correctly if it wasn't explicitly before
        # The MemorySegment model itself handles the metadata field, so direct assignment is fine.
        
        self.segments.append(segment)
        self.logger.debug(f"Added memory segment (ID: {segment.id}, Type: {segment.type}, Source: {segment.source}, Session: {segment.metadata.get('session_id')})")
        await self._save_to_disk()
        # ... (existing logic for FAISS if applicable) ...

    async def get_history_for_session(self, session_id: str, limit: int = 20) -> List[Dict[str, str]]:
        """
        Retrieves conversation history for a given session_id, ordered by timestamp.
        Filters for segments that have a 'role' in their metadata.
        """
        self.logger.debug(f"Retrieving history for session_id: {session_id} with limit: {limit}")
        session_segments = []
        for segment in self.segments:
            if segment.metadata and segment.metadata.get("session_id") == session_id:
                # We are looking for segments that represent conversational turns
                # These should have a 'role' (user/assistant) and 'text' content
                role = segment.metadata.get("role")
                if role and segment.content and segment.content.text:
                    session_segments.append(
                        {
                            "role": role,
                            "content": segment.content.text,
                            "timestamp": segment.timestamp # Keep timestamp for sorting
                        }
                    )
        
        # Sort by timestamp (datetime objects)
        session_segments.sort(key=lambda x: x["timestamp"])
        
        # Remove timestamp and take the most recent 'limit' turns
        formatted_history = [{"role": s["role"], "content": s["content"]} for s in session_segments]
        
        if len(formatted_history) > limit:
            self.logger.debug(f"History for session '{session_id}' exceeds limit {limit}, trimming to most recent {limit} turns.")
            return formatted_history[-limit:]
        
        self.logger.info(f"Retrieved {len(formatted_history)} history turns for session_id: {session_id}")
        return formatted_history

    # ... (other methods like goal management, pruning, etc. would go here)