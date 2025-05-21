# Welcome to the Memory Palace! Where all our AI memories live ^_^
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime  # Time is an illusion... but we track it anyway =P
import json  # JSON is life! \o/
import os  # Need this for file stuff
import uuid  # Every memory needs a unique ID (please don't collide x.x)
import time  # More time tracking because why not?
import re  # Regex: The language of the gods... and debugging nightmares lol
import logging  # For when things go wrong (and they will!)
from pydantic import BaseModel, Field  # Keeping our data clean and validated =D
import aiofiles  # Async file ops ftw!
import numpy as np  # Math magic incoming!
import faiss
from sentence_transformers import SentenceTransformer
from .schemas import MemorySegment, MemorySegmentContent, MemoryConfig

# GPU resources for FAISS
RES: Optional[faiss.StandardGpuResources] = None
CUDA_DEVICE: Optional[int] = None

try:
    # Try to initialize GPU resources =D
    RES = faiss.StandardGpuResources()
    CUDA_DEVICE = 0  # Use first GPU by default \o/
    logging.info("FAISS GPU resources initialized! SPEEEEED! \\o/")
except Exception as e:
    logging.warning(f"Could not initialize FAISS GPU resources! x.x: {str(e)}")
    RES = None
    CUDA_DEVICE = None

def create_gpu_index(dimension: int, res: faiss.StandardGpuResources, device: int) -> faiss.GpuIndexFlatL2:
    """Create a GPU-powered FAISS index! Time to go FAST! \\o/"""
    config = faiss.GpuIndexFlatConfig()
    config.device = device
    return faiss.GpuIndexFlatL2(res, dimension, config)

# Import all our helper functions (don't forget these or everything breaks! x.x)
from .debug_utils import log_execution_time, log_async_execution_time, DebugInfo, log_debug_info, PerformanceMonitor
from .faiss_utils import initialize_gpu_resources, create_gpu_index, add_vectors, search_vectors

# Setup our logger first
logger = logging.getLogger('WITS.MemoryManager')

# Time to wake up the GPU! Let's hope it had its coffee today >.>
RES: Optional[faiss.StandardGpuResources] = None
CUDA_DEVICE: Optional[int] = None
VECTOR_SEARCH_AVAILABLE = False
VECTOR_SEARCH_GPU = False

try:
    res, device = initialize_gpu_resources()  # Poking the GPU with a stick
    VECTOR_SEARCH_AVAILABLE = True  # Yay, vectors! \o/
    VECTOR_SEARCH_GPU = True  # GPU go brrrrr! =D
    RES, CUDA_DEVICE = res, device
    logger.info(f"GPU is awake and ready to crunch some vectors on device {device}! ^_^")
except Exception as e:  # Uh oh... something's not right x.x
    # Just log the error and continue, we'll raise errors later if needed
    logger.error(f"CRITICAL: GPU seems sleepy: {str(e)}. Need GPU superpowers! x.x")

class MemoryManager:
    """
    Memory Manager - Manages our vector database of memories! \\o/
    Uses FAISS for super fast vector search! =D
    
    Features (aka our superpowers! ^_^):
    - GPU-powered vector search with FAISS (zoom zoom! =D)
    - Memory segments with embeddings (math magic! O.o)
    - Goal tracking (because we need direction! >.>)
    - Memory pruning and saving (spring cleaning for data! x.x)
    - Performance monitoring (gotta go fast! \\o/)
    """
    
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        debug_component_config: Optional[Dict[str, Any]] = None
    ):
        """
        Time to build our memory palace! Let's get this party started! \\o/
        
        Args:
            config: Configuration object for memory manager settings
            debug_component_config: Debug configuration for enabling/disabling debug features
        """
        # Load config or use defaults \o/
        self.config = config or MemoryConfig()
        self.debug_component_config = debug_component_config or self.config.debug_components
        self.debug_enabled = self.config.debug_enabled  # Initialize our components ^.^
        self.embedding_model: Optional[SentenceTransformer] = None
        self.vector_dim: int = 0
        self.index: Optional[Union[faiss.IndexFlatL2, faiss.GpuIndexFlatL2]] = None
        self.memory_segments: List[MemorySegment] = []
        # Alias for compatibility with test code
        self.segments = self.memory_segments
        self.is_initialized: bool = False
        self.faiss_idx_to_id: List[str] = []  # Gotta keep track of our FAISS IDs! ^_^
        self.id_to_faiss_idx: Dict[str, int] = {}  # And the reverse mapping, just in case! =P
        self.memory_file: Optional[str] = self.config.memory_file # Where we save our precious memories! <3
        
        # Set up logging =D
        self.logger = logging.getLogger(__name__)
        self.logger.info("Memory Manager powering up! Time to make some memories! \\o/")

    async def initialize_db(self) -> None:
        """Initialize our memory database! Time to get this party started! \\o/"""
        try:
            # Set up our embedding model =D
            self.logger.info(f"Loading embedding model {self.config.vector_model}... ^.^")
            self.embedding_model = SentenceTransformer(self.config.vector_model)
            dim = self.embedding_model.get_sentence_embedding_dimension()
            if not isinstance(dim, int):
                # This should ideally not happen with standard sentence transformers, but good to be safe!
                self.logger.error("Embedding dimension is not an int! This is weird! x.x")
                raise ValueError("Invalid embedding dimension! x.x")
            self.vector_dim = dim
            
            # Create our FAISS index with GPU support if available! SPEEEEED! \\o/
            if RES is not None and CUDA_DEVICE is not None and VECTOR_SEARCH_GPU: # Check VECTOR_SEARCH_GPU too!
                self.index = create_gpu_index(self.vector_dim, RES, CUDA_DEVICE)
                self.logger.info("GPU-powered FAISS index created! Time to go FAST! =D")
            else:
                # Fallback to CPU if no GPU available x.x
                self.index = faiss.IndexFlatL2(self.vector_dim)
                self.logger.warning("No GPU available or GPU search disabled - using CPU index instead >.>")
            
            self.is_initialized = True
            self.logger.info("Memory system initialized! Ready to remember ALL THE THINGS! \\o/")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory system! Everything is broken! x.x: {str(e)}", exc_info=True)
            self.is_initialized = False
            raise

    def _initialize_vector_search(self):
        """
        Time to set up our vector search superpowers! \\o/
        GPU-accelerated memory searching, because we're living in the future! ^_^
        """
        if not VECTOR_SEARCH_AVAILABLE:
            self.logger.warning("Vector search is sleeping! Taking a nap... x.x")
            return

        if not self.config.vector_model: # Use self.config.vector_model here!
            self.logger.warning("No vector model name in config! We're flying blind! >.>")
            return

        self.logger.info(f"Powering up vector search! Loading {self.config.vector_model}... =D")
        try:
            # Time to load our embedding model! *sparkles* \\o/
            self.embedding_model = SentenceTransformer(self.config.vector_model) # Use self.config.vector_model
            embedding_dim_result = self.embedding_model.get_sentence_embedding_dimension()
            if not isinstance(embedding_dim_result, int):
                self.logger.error(f"Embedding dimension is not an int! Got {embedding_dim_result} x.x")
                raise ValueError(f"Vector dimension should be a number, not {embedding_dim_result}! Math is hard x.x")
            self.vector_dim = embedding_dim_result
            self.logger.info(f"Vector dimension set to {self.vector_dim}! That's a lot of numbers! ^_^")
            
            # FAISS index time! GPU power activate! =P
            if RES is not None and CUDA_DEVICE is not None and VECTOR_SEARCH_GPU: # Check VECTOR_SEARCH_GPU
                self.index = create_gpu_index(self.vector_dim, RES, CUDA_DEVICE)
                self.logger.info(f"GPU-powered FAISS index is ready on device {CUDA_DEVICE}! Let's goooo! \\o/")
            elif self.vector_dim > 0: # Ensure vector_dim is valid before creating CPU index
                self.index = faiss.IndexFlatL2(self.vector_dim)
                self.logger.info("CPU FAISS index created. GPU not available or disabled. Still pretty fast though! =P")
            else:
                self.logger.error("Vector dimension is not valid for FAISS index creation! x.x")
                raise ValueError("Cannot create FAISS index with invalid vector dimension.")
            
        except Exception as e:
            self.logger.error(f"Oopsie! Vector initialization went boom: {e}", exc_info=True)
            self.embedding_model = None
            self.index = None
            raise RuntimeError(f"Vector search setup failed... Need more GPU power! >.>") from e

    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Turn words into vectors! Math magic time! ^_^
        
        Args:
            text: The words we want to vectorize! =D
            
        Returns:
            np.ndarray: A fancy math vector! (or None if something goes wrong x.x)
        """
        if not self.embedding_model:
            self.logger.warning("No embedding model found! We're vector-less! O.o")
            return None

        try:
            start_time = time.time()
            vector = self.embedding_model.encode([text])[0]  # Vectorize! \\o/
            generation_time = (time.time() - start_time) * 1000  # ms
            
            if self.debug_enabled and isinstance(self.debug_component_config, dict):
                if self.debug_component_config.get('log_embeddings'):
                    self.logger.debug(
                        f"Vector generated! =D\n"
                        f"Text length: {len(text)} chars\n"
                        f"Vector dimension: {len(vector)} (that's a lot of numbers! ^_^)\n"
                        f"Generation time: {generation_time:.2f}ms (zoom zoom! \\o/)"
                    )

            return vector
        except Exception as e:
            self.logger.error(f"Vector generation went sideways: {e} x.x", exc_info=True)
            return None

    def _add_to_index(self, vector: np.ndarray, segment_id: str) -> bool:
        """
        Add a vector to our FAISS index! More math magic! \\o/
        
        Args:
            vector: The math representation we want to store ^_^
            segment_id: How we'll find it later =D
            
        Returns:
            bool: True if it worked, False if something went boom! x.x
        """
        if self.index is None:
            self.logger.warning("No FAISS index! Where did it go? O.o")
            return False

        try:
            # Prepare the vector (gotta make it look nice for FAISS >.>)
            vector_np = np.array([vector], dtype=np.float32)
            
            # Add it to the index! *magic wand wave* \\o/
            add_vectors(self.index, vector_np)
            
            # Keep track of where we put it =P
            idx = len(self.faiss_idx_to_id)
            self.id_to_faiss_idx[segment_id] = idx
            self.faiss_idx_to_id.append(segment_id)
            if self.debug_enabled and isinstance(self.debug_component_config, dict):
                if self.debug_component_config.get('log_embeddings'):
                    self.logger.debug(f"Vector added to index! ID: {segment_id}, Position: {idx} ^_^")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add vector to index: {e} x.x", exc_info=True)
            return False

    def _search_similar(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[float, str]]:
        """
        Find similar vectors in our memory palace! Searching... searching... ^_^
        
        Args:
            query_vector: What we're looking for \\o/
            k: How many matches we want =D
            
        Returns:
            List of (score, segment_id) tuples! The lower the score, the better! O.o
        """
        if self.index is None or len(self.faiss_idx_to_id) == 0:
            self.logger.warning("No index or it's empty! Nothing to search! x.x")
            return []

        try:            # Search time! *puts on detective hat* \\o/
            # Correctly use the imported search_vectors utility
            distances, indices = search_vectors(
                self.index, # Pass the index object
                np.array([query_vector], dtype=np.float32),
                min(k, len(self.faiss_idx_to_id))  # Can't find more than we have! =P
            )

            # Process results (math → meaning) ^_^
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.faiss_idx_to_id):  # Safety first! >.>
                    segment_id = self.faiss_idx_to_id[idx]
                    results.append((float(dist), segment_id))
            
            if self.debug_enabled and isinstance(self.debug_component_config, dict):
                if self.debug_component_config.get('log_searches'):
                    self.logger.debug(
                        f"Vector search complete! \\o/\n"
                        f"Found {len(results)} matches!\n"
                        f"Best match distance: {results[0][0] if results else 'N/A'} =D"
                    )

            return results

        except Exception as e:
            self.logger.error(f"Search failed! Detective work is hard: {e} x.x", exc_info=True)
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
        """
        Time to make new memories! Adding a fresh segment to our collection! \\o/
        
        Args:
            segment_type: What kind of memory is this? ^_^
            content_text: The actual memory content =D
            source: Who/what created this memory? \\o/
            tool_name: Did a tool help make this memory? O.o
            tool_args: Tool settings used (if any) >.>
            tool_output: What did the tool say? =P
            importance: How important is this? (0-1) x.x
            meta: Extra info for the curious! =D
            
        Returns:
            str: The memory's special ID! Don't lose it! \\o/
        """
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
            metadata=meta or {}
        )
        
        # Set importance if supported by model
        if hasattr(segment, 'importance'):
            segment.importance = importance
        
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
        """
        Time to write our memories to disk! Don't forget anything! ^_^
        This is like our digital diary, but async because we're fancy! \\o/
        """
        if not self.memory_file: # Check if memory_file is set
            self.logger.warning("Memory file path not set. Cannot save to disk. Oops >.>")
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
        """
        Let's see how our vector search is doing! Status check time! =D
        Returns a report card of our vector search capabilities! O.o
        """
        status = {
            "vector_search_available": VECTOR_SEARCH_AVAILABLE,
            "gpu_enabled": VECTOR_SEARCH_GPU, # This is now a top-level constant
            "embedding_model": self.config.vector_model if self.embedding_model else None, # Use self.config.vector_model
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

    def debug_dump_memory(self) -> None:
        """
        Dump the current memory segments to the debug log! For when you want to see ALL the things! O.o
        """
        try:
            if not self.segments or len(self.segments) == 0:
                self.logger.info("No memory segments to dump. It's all quiet... too quiet... >.>")
                return
            
            self.logger.info(f"Dumping {len(self.segments)} memory segments to debug log:")
            for segment in self.segments:
                self.logger.info(
                    f"Segment ID: {segment.id}\n"
                    f"Type: {segment.type}\n" # Corrected from segment_type to type
                    f"Content: {segment.content}\n"
                    f"Importance: {segment.importance}\n"
                    f"Timestamp: {segment.timestamp}\n"
                    f"Metadata: {segment.metadata}\n"
                    f"Embedding (first 5 dims): {segment.embedding[:5] if segment.embedding else 'None'}\n"
                    f"-----------------------------"
                )
        except Exception as e:
            self.logger.error(f"Failed to dump memory segments: {e}", exc_info=True)

    async def load_memory(self) -> None:
        """
        Load memory segments from disk! Time to see what we've got stored! ^_^
        """
        if not self.memory_file or not os.path.exists(self.memory_file): # Added self.memory_file check
            self.logger.warning(f"Memory file not found or path not set: {self.memory_file}. Starting with empty memory.")
            return
        
        try:
            async with aiofiles.open(self.memory_file, 'r') as f:
                contents = await f.read()
                if not contents.strip():
                    self.logger.warning("Memory file is empty! Nothing to load.")
                    return
                
                # Deserialize the JSON data
                loaded_segments_data = json.loads(contents)
                loaded_segments_count = 0
                for seg_dict in loaded_segments_data:
                    try:
                        # Convert timestamp back to datetime object
                        if 'timestamp' in seg_dict and isinstance(seg_dict['timestamp'], str):
                            seg_dict['timestamp'] = datetime.fromisoformat(seg_dict['timestamp'])
                        
                        # Recreate the MemorySegment object
                        segment = MemorySegment.model_validate(seg_dict)
                        self.segments.append(segment) # self.segments is an alias for self.memory_segments
                        
                        # Add to FAISS index if initialized and segment has embedding
                        if self.is_initialized and self.index is not None and segment.embedding:
                            try:
                                embedding_np = np.array(segment.embedding, dtype=np.float32)
                                if embedding_np.ndim == 1:
                                    # Check for dimension mismatch against the initialized model's dimension
                                    if self.vector_dim > 0 and len(embedding_np) != self.vector_dim:
                                        self.logger.warning(
                                            f"Skipping indexing segment {segment.id} during load: embedding dimension mismatch. "
                                            f"Expected {self.vector_dim}, got {len(embedding_np)}."
                                        )
                                    else:
                                        self._add_to_index(embedding_np, segment.id) # Correctly updates mappings
                                else:
                                    self.logger.warning(f"Segment {segment.id} has an invalid embedding shape during load, not indexing.")
                            except Exception as e:
                                self.logger.error(f"Error indexing segment {segment.id} during load: {e}", exc_info=True)
                        
                        self.logger.info(f"Loaded segment: ID={segment.id}, Type='{segment.type}', Source='{segment.source}'")
                        loaded_segments_count +=1
                    except Exception as e:
                        self.logger.error(f"Error loading individual segment from disk: {e}", exc_info=True)
            
            self.logger.info(f"Loaded {loaded_segments_count} memory segments from disk.")
        except Exception as e:
            self.logger.error(f"Failed to load memory from disk: {e}", exc_info=True)

    async def close(self) -> None:
        """
        Close the memory manager! Like saying goodbye to a friend... until next time! T_T
        """
        try:
            # Optionally, save memory to disk on close
            await self._save_to_disk()
            
            # Clear all resources
            self.segments.clear()
            self.id_to_faiss_idx.clear()
            self.faiss_idx_to_id.clear()
            
            if self.index is not None:
                self.index.reset()
                del self.index
                self.index = None
            
            if self.embedding_model is not None:
                del self.embedding_model
                self.embedding_model = None
            
            self.logger.info("Memory manager closed. Until next time, stay curious! ^_^")
        except Exception as e:
            self.logger.error(f"Error closing memory manager: {e}", exc_info=True)

    async def search_memory(
        self, 
        query: str, 
        limit: int = 5,
        min_relevance: float = 0.0
    ) -> List[MemorySegment]:
        """
        Search through our memories like a pro gamer! B-)
        
        Args:
            query: What we're looking for o.O
            limit: How many memories to return (default: 5)
            min_relevance: Minimum relevance score (0 to 1) ^.^
            
        Returns:
            List[MemorySegment]: The most relevant memories we found! \\\\o/
        """
        try:
            if not self.is_initialized:
                self.logger.warning("Memory system not initialized! Hold up...attempting to fix that! >.<")
                await self.initialize_db() 
                if not self.is_initialized: 
                    self.logger.error("Failed to initialize memory system after attempt! x.x")
                    raise RuntimeError("Failed to initialize memory system! x.x")

            if not self.index or not hasattr(self.index, 'ntotal'):
                self.logger.warning("FAISS index is not available or not initialized properly. Cannot search. >.<")
                return []
            
            num_vectors_in_index = self.index.ntotal
            if num_vectors_in_index == 0:
                self.logger.warning("No memories to search through yet (index is empty)! >.<")
                return []

            query_vector = await self.get_embedding(query)
            if query_vector is None:
                self.logger.error("Failed to create query vector! The math spirits are angry today >.>")
                return []

            k_search = min(limit * 2, num_vectors_in_index) 
            if k_search <= 0:
                 self.logger.info("k_search is 0 or less, no vectors to search for effectively.")
                 return []

            # Use search_vectors utility from faiss_utils.py
            distances_faiss, indices_faiss = search_vectors(
                self.index,
                np.array([query_vector], dtype='float32'), 
                k_search
            )

            results = []
            if distances_faiss is not None and indices_faiss is not None and \
               len(distances_faiss) > 0 and len(indices_faiss) > 0 and \
               len(distances_faiss[0]) > 0 and len(indices_faiss[0]) > 0:
                
                for dist, faiss_idx in zip(distances_faiss[0], indices_faiss[0]):
                    if faiss_idx < 0: # FAISS can return -1 for invalid indices
                        self.logger.debug(f"FAISS returned invalid index {faiss_idx}, skipping.")
                        continue
                    
                    if faiss_idx >= len(self.faiss_idx_to_id):
                        self.logger.warning(f"Invalid faiss_idx {faiss_idx} (out of range for faiss_idx_to_id list of len {len(self.faiss_idx_to_id)}) encountered during search.")
                        continue

                    segment_id_from_faiss = self.faiss_idx_to_id[faiss_idx]
                    memory_segment = await self.get_memory(segment_id_from_faiss)

                    if memory_segment:
                        relevance = 1.0 - (float(dist)**2 / 2.0) # This assumes embeddings are normalized (length 1)
                        relevance = max(0.0, min(1.0, relevance))

                        if relevance >= min_relevance:
                            setattr(memory_segment, 'relevance_score', relevance) # Add relevance score to segment
                            results.append(memory_segment)

                            if len(results) >= limit:
                                break
                    else:
                        self.logger.warning(f"Memory segment with ID {segment_id_from_faiss} (faiss_idx {faiss_idx}) not found via get_memory.")
            else:
                self.logger.info(f"Search returned no valid distances/indices from search_vectors for query: '{query[:50]}...'")
            
            # Sort by relevance score descending before returning
            results.sort(key=lambda seg: getattr(seg, 'relevance_score', 0.0), reverse=True)


            if self.debug_enabled and isinstance(self.debug_component_config, dict) and self.debug_component_config.get('log_searches'):
                top_score_str = 'N/A'
                if results and hasattr(results[0], 'relevance_score'):
                    top_score_str = f"{results[0].relevance_score:.4f}"
                self.logger.debug(
                    f"Search complete! Found {len(results)} memories! =D\\n"
                    f"Query: {query}\\n"
                    f"Top relevance score: {top_score_str}"
                )

            return results
        except Exception as e:
            self.logger.error(f"Memory search failed! Something's broken: {str(e)} >.<", exc_info=True)
            return []

    async def get_memory(self, memory_id: str) -> Optional[MemorySegment]:
        """
        Find a specific memory by ID! Like searching for your keys... but easier! ^.^
        
        Args:
            memory_id: The ID of the memory we want =D
            segment_type: Optional filter for the type of memory segment
            metadata_filter: Optional dictionary to filter segments by metadata key-value pairs
        """
        try:
            for segment in self.memory_segments:
                if segment.id == memory_id:
                    return segment
            return None
        except Exception as e:
            self.logger.error(f"Failed to get memory! ID={memory_id}: {str(e)} x.x")
            return None

    async def remove_memory(self, memory_id: str) -> bool: # Corrected return type annotation
        """
        Remove a memory from our collection! Sometimes we gotta let go... T.T
        
        Args:
            memory_id: The ID of the memory to remove >.>
            
        Returns:
            bool: True if removed successfully, False if failed x.x
        """
        try:
            segment_to_remove = None
            segment_idx_in_list = -1

            for idx, segment in enumerate(self.memory_segments):
                if segment.id == memory_id:
                    segment_to_remove = segment
                    segment_idx_in_list = idx
                    break
            
            if segment_to_remove is None: 
                self.logger.warning(f"Memory not found for removal! ID={memory_id} x.x")
                return False

            # Remove from memory segments list
            self.memory_segments.pop(segment_idx_in_list)
            
            # If the segment had an embedding and an index exists, rebuild the index
            # to ensure consistency of FAISS index and mappings.
            if self.index is not None and segment_to_remove.embedding:
                self.logger.info(f"Segment {memory_id} removed, rebuilding FAISS index.")
                self._rebuild_index() 
            
            await self._save_to_disk() 
            self.logger.info(f"Memory removed successfully! ID={memory_id} \\o/")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove memory! ID={memory_id}: {str(e)} >.<", exc_info=True)
            return False

    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Turn text into a vector! It's like magic, but with math! ^_^
        
        Args:
            text: The text to vectorize \\o/
            
        Returns:
            np.ndarray: The vector representation (or None if something went wrong x.x)
        """
        if not text or not isinstance(text, str):
            self.logger.warning("Invalid text input for embedding - must be non-empty string! >.>")
            return None
            
        try:
            if self.embedding_model is None:
                await self.initialize_db()
                if self.embedding_model is None:
                    self.logger.error("Embedding model still not available after attempting initialization in get_embedding.")
                    return None

            # Add debug logging if enabled =D
            if self.debug_enabled and self.debug_component_config.get('log_embeddings'):
                self.logger.debug(f"Generating embedding for text (length: {len(text)}) ^_^")

            # Time to do some vector math! \\o/
            start_time = time.time()
            vector = self.embedding_model.encode([text])[0]
            generation_time = (time.time() - start_time) * 1000  # ms
            
            if self.debug_enabled and self.debug_component_config.get('log_embeddings'):
                self.logger.debug(
                    f"Vector generated in {generation_time:.2f}ms! =D\n"
                    f"Shape: {vector.shape} \\o/"
                )
            
            return vector.astype('float32')  # Make sure it's float32 for FAISS
            
        except Exception as e:
            self.logger.error(f"Vector creation failed! Math is hard: {str(e)} x.x", exc_info=True)
            return None

    def _search_vectors(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors using FAISS! Magic time! \\\\o/
        
        Args:
            query_vector: The vector to search for ^.^ (should be 2D, e.g., (1, dim))
            k: How many results we want =D
            
        Returns:
            Tuple of (distances, indices) arrays \\\\o/
        """
        if not self.index:
            self.logger.error("No index available for search! x.x") 
            raise RuntimeError("No index available for search! x.x")
            
        try:
            # Make sure vector is the right shape and type =D
            # FAISS expects query_vector to be (nqueries, dimension)
            if query_vector.ndim == 1:
                 query_vector_reshaped = query_vector.reshape(1, -1).astype(np.float32)
            elif query_vector.ndim == 2:
                 query_vector_reshaped = query_vector.astype(np.float32)
            else:
                self.logger.error(f"Invalid query_vector dimension: {query_vector.ndim}. Expected 1 or 2.")
                return np.array([]).reshape(0,0), np.array([]).reshape(0,0) # Adjusted shape for empty result

            # Get number of results we can actually return \\o/
            current_ntotal = self.index.ntotal if hasattr(self.index, 'ntotal') else 0
            
            # k_actual is the number of neighbors to request from FAISS
            k_actual = min(k, current_ntotal)

            if k_actual == 0: 
                # Return empty arrays with shape (nqueries, 0) for distances and indices
                return np.array([]).reshape(query_vector_reshaped.shape[0],0), np.array([]).reshape(query_vector_reshaped.shape[0],0) 
            
            # Perform the search using the search_vectors utility
            # This returns distances D (nqueries, k_actual) and indices I (nqueries, k_actual)
            distances, indices = search_vectors(self.index, query_vector_reshaped, k_actual)
            return distances, indices
            
        except Exception as e:
            self.logger.error(f"Error in _search_vectors: {e}", exc_info=True)
            # Adjusted shape for empty result based on potential nqueries
            num_queries = query_vector.shape[0] if query_vector.ndim == 2 else (1 if query_vector.ndim == 1 else 0)
            return np.array([]).reshape(num_queries,0), np.array([]).reshape(num_queries,0)

    def _add_vectors(self, vectors: np.ndarray) -> None:
        # This method is currently not implemented or used.
        # If batch vector addition is needed, it should be implemented here,
        # potentially using the add_vectors utility from faiss_utils.
        # For now, _add_to_index handles single vector additions.
        pass

    def _rebuild_index(self) -> None:
        """
        Rebuilds the FAISS index and associated mappings from self.memory_segments.
        This is useful after bulk changes like removing segments.
        """
        if not self.is_initialized or self.index is None:
            self.logger.warning("Cannot rebuild index: MemoryManager not initialized or no FAISS index.")
            return

        self.logger.info("Rebuilding FAISS index and mappings...")
        try:
            self.index.reset() # Clears the FAISS index
        except Exception as e:
            self.logger.error(f"Error resetting FAISS index during rebuild: {e}", exc_info=True)
            # Depending on the error, might not be safe to proceed.
            return 
            
        self.faiss_idx_to_id.clear()
        self.id_to_faiss_idx.clear()

        for segment in self.memory_segments:
            if segment.embedding: # segment.embedding is List[float]
                try:
                    embedding_np = np.array(segment.embedding, dtype=np.float32)
                    if embedding_np.ndim == 1:
                        # Check for dimension mismatch
                        if self.vector_dim > 0 and len(embedding_np) != self.vector_dim:
                            self.logger.warning(
                                f"Skipping segment {segment.id} during index rebuild: embedding dimension mismatch. "
                                f"Expected {self.vector_dim}, got {len(embedding_np)}."
                            )
                            continue
                        self._add_to_index(embedding_np, segment.id)
                    else:
                        self.logger.warning(
                            f"Segment {segment.id} has an invalid embedding shape (ndim != 1) during rebuild, skipping."
                        )
                except Exception as e:
                    self.logger.error(f"Error processing segment {segment.id} during index rebuild: {e}", exc_info=True)
        
        current_ntotal = self.index.ntotal if hasattr(self.index, 'ntotal') else 'N/A'
        self.logger.info(f"FAISS index rebuild complete. Index size: {current_ntotal}")

async def load_memory_from_disk(memory_file_path: str) -> List[MemorySegment]:
    """Load memories from a file! Let's see what we've forgotten! O.o"""
    # Ensure memory_file_path is a valid string and the file exists before trying to open it!
    if not memory_file_path or not isinstance(memory_file_path, str) or not os.path.exists(memory_file_path):
        logger.warning(f"Memory file '{memory_file_path}' not found or path is invalid. Starting with a blank slate! =P")
        return []
    try:
        async with aiofiles.open(memory_file_path, 'r') as f: # Now we know memory_file_path is a valid string path
            data = await f.read()
            if not data.strip(): # Check if file is empty
                logger.warning(f"Memory file '{memory_file_path}' is empty. Starting fresh! ^_^")
                return []
            segments_data = json.loads(data)
            # Convert timestamp strings back to datetime objects before creating MemorySegment
            loaded_segments = []
            for seg_data in segments_data:
                if "timestamp" in seg_data and isinstance(seg_data["timestamp"], str):
                    try:
                        seg_data["timestamp"] = datetime.fromisoformat(seg_data["timestamp"])
                    except ValueError as ve:
                        logger.error(f"Error converting timestamp for segment: {seg_data.get('id', 'N/A')}. {ve} x.x")
                        # Decide how to handle this - skip segment, use default time, etc.
                        # For now, let's skip if timestamp is crucial and invalid
                        continue 
                loaded_segments.append(MemorySegment(**seg_data))
            return loaded_segments
    except Exception as e:
        logger.error(f"Error loading memory from {memory_file_path}: {e} x.x", exc_info=True)
        return []

class EnhancedMemoryManager(MemoryManager):
    """
    Our MemoryManager, but with EXTRA features! Like a deluxe edition! \\o/
    This one can load and save memories, and even re-index them! So fancy! =D
    """
    def __init__(self, config: Optional[MemoryConfig] = None, debug_component_config: Optional[Dict[str, Any]] = None):
        super().__init__(config, debug_component_config)
        self.memory_file: Optional[str] = (config.memory_file if config else None) # Ensure memory_file is set from config
        self.logger = logging.getLogger(f"WITS.MemoryManager.{self.__class__.__name__}") # More specific logger =P
        self.logger.info(f"{self.__class__.__name__} is here to remember ALL THE THINGS! \\o/")

    async def initialize_and_load(self) -> None:
        """Initialize the DB and load memories from disk! Double whammy! =D"""
        await self.initialize_db() # First, get the DB ready! ^.^
        if self.memory_file and self.is_initialized: # Only load if we have a file and DB is ready!
            self.logger.info(f"Loading memories from {self.memory_file}... Hope I remember where I put them! >.>")
            # load_memory_from_disk expects a non-Optional string, self.memory_file is Optional[str]
            # We've already checked self.memory_file is not None here.
            loaded_segments = await load_memory_from_disk(self.memory_file)
            if loaded_segments:
                self.segments.extend(loaded_segments)
                self.logger.info(f"Loaded {len(loaded_segments)} memories! My brain is full! O.o")
                await self.reindex_all_segments() # Re-index after loading! Important step! x.x
            else:
                self.logger.info(f"No memories found in '{self.memory_file}' or file is empty. Starting fresh! =P")
        elif not self.memory_file:
            self.logger.warning("No memory file configured. Starting with a blank slate. So empty... O.o")
        elif not self.is_initialized:
            self.logger.error("Cannot load memories, DB initialization failed! This is not good! x.x")

    async def reindex_all_segments(self) -> None:
        """Re-index all existing memory segments. Spring cleaning for our vectors! \\\\o/"""
        if not self.index or not self.embedding_model:
            self.logger.warning("Cannot re-index. Index or embedding model not ready. Maybe later? >.>")
            return

        self.logger.info(f"Re-indexing {len(self.segments)} segments... This might take a moment! =P")
        
        # Reset FAISS index and mappings (careful, this erases the old index! x.x)
        if RES is not None and CUDA_DEVICE is not None and VECTOR_SEARCH_GPU:
            self.index = create_gpu_index(self.vector_dim, RES, CUDA_DEVICE)
        elif self.vector_dim > 0:
            self.index = faiss.IndexFlatL2(self.vector_dim)
        else:
            self.logger.error("Cannot re-create FAISS index, vector dimension is invalid! Oh noes! x.x")
            return
            
        self.faiss_idx_to_id = []
        self.id_to_faiss_idx = {}
        
        vectors_to_add = []
        segment_ids_for_vectors = []

        for segment in self.segments:
            embedding_text = segment.content.text or segment.content.tool_output or ""
            if embedding_text:
                embedding = self._generate_embedding(embedding_text)
                if embedding is not None:
                    segment.embedding = embedding.tolist() # Update segment's embedding
                    vectors_to_add.append(embedding)
                    segment_ids_for_vectors.append(segment.id)
        
        if vectors_to_add:
            vectors_np = np.array(vectors_to_add, dtype=np.float32)
            try:
                add_vectors(self.index, vectors_np) # Use the utility here
                for i, seg_id in enumerate(segment_ids_for_vectors):
                    self.id_to_faiss_idx[seg_id] = i
                    self.faiss_idx_to_id.append(seg_id)
                self.logger.info(f"Successfully re-indexed {len(vectors_to_add)} segments! All shiny and new! \\o/")
            except Exception as e:
                self.logger.error(f"Failed to add batch of vectors during re-index: {e} x.x", exc_info=True)
        else:
            self.logger.info("No segments with valid text found to re-index. All quiet on the western front! >.>")

    async def add_memory_segment(self, segment_to_add: MemorySegment) -> Optional[MemorySegment]: # MODIFIED: Changed segment_data: Dict[str, Any] to segment_to_add: MemorySegment
        """Adds a new memory segment (expects a MemorySegment object), saves, and re-indexes.""" # MODIFIED: Updated docstring
        try:
            # The input is now a MemorySegment object.
            # Pydantic validation would have occurred when segment_to_add was created.
            # Ensure 'id' and 'timestamp' are set (Pydantic models with default_factory should handle this)
            if not segment_to_add.id: # Should be set by default_factory
                segment_to_add.id = str(uuid.uuid4())
            if not segment_to_add.timestamp: # Should be set by default_factory
                segment_to_add.timestamp = datetime.now()
            elif isinstance(segment_to_add.timestamp, str): # Robustness for string timestamps
                try:
                    segment_to_add.timestamp = datetime.fromisoformat(segment_to_add.timestamp)
                except ValueError:
                    self.logger.warning(f"Invalid timestamp string format for segment {segment_to_add.id}. Using current time.")
                    segment_to_add.timestamp = datetime.now()


            # Generate embedding if not provided and content is available
            if segment_to_add.embedding is None:
                embedding_text = None
                if segment_to_add.content: # content is MemorySegmentContent
                    embedding_text = segment_to_add.content.text or segment_to_add.content.tool_output
                
                if embedding_text and self.embedding_model:
                    embedding_vector = self._generate_embedding(embedding_text)
                    if embedding_vector is not None:
                        segment_to_add.embedding = embedding_vector.tolist()
            
            self.memory_segments.append(segment_to_add)

            # Add to FAISS index if embedding exists
            if segment_to_add.embedding and self.index is not None:
                embedding_np = np.array(segment_to_add.embedding, dtype=np.float32)
                if embedding_np.ndim == 1: # Ensure it's a 1D array before adding
                    # Check for dimension mismatch before adding to index
                    if self.vector_dim > 0 and len(embedding_np) != self.vector_dim:
                        self.logger.warning(
                            f"Segment {segment_to_add.id} embedding dimension mismatch. "
                            f"Expected {self.vector_dim}, got {len(embedding_np)}. Not adding to FAISS index."
                        )
                    else:
                        self._add_to_index(embedding_np, segment_to_add.id)
                else:
                    self.logger.warning(f"Segment {segment_to_add.id} has invalid embedding shape (ndim: {embedding_np.ndim}), not adding to FAISS index.")

            await self._save_to_disk() # Save all segments
            self.logger.info(f"Added and saved new memory segment: {segment_to_add.id}")
            return segment_to_add

        except Exception as e:
            self.logger.error(f"Error in EnhancedMemoryManager add_memory_segment: {e}", exc_info=True)
            return None

    async def search_memory(
        self, 
        query: str, 
        limit: int = 5,
        min_relevance: float = 0.0,
        segment_type: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[MemorySegment]:
        """
        Search for memories! Let's find what we're looking for! \\o/
        Can filter by type and metadata too! So powerful! =D
        """
        try:
            if not self.is_initialized:
                self.logger.warning("EnhancedMemoryManager not initialized. Attempting to initialize.")
                await self.initialize_and_load() # Use the enhanced init
                if not self.is_initialized:
                    self.logger.error("Failed to initialize EnhancedMemoryManager.")
                    return []

            if not self.index or self.index.ntotal == 0:
                self.logger.info("FAISS index is empty or not available.")
                return []

            query_vector = await self.get_embedding(query)
            if query_vector is None:
                self.logger.error("Failed to get query embedding.")
                return []

            k_search = min(limit * 2 if limit > 0 else 10, self.index.ntotal)
            if k_search <= 0:
                return []

            # Use the search_vectors utility function
            distances, indices = search_vectors(
                self.index,
                np.array([query_vector], dtype='float32'),
                k_search
            )

            results = []
            if distances is not None and indices is not None and len(distances[0]) > 0:
                for dist, faiss_idx in zip(distances[0], indices[0]):
                    if faiss_idx < 0 or faiss_idx >= len(self.faiss_idx_to_id):
                        continue
                    
                    segment_id = self.faiss_idx_to_id[faiss_idx]
                    segment = await self.get_memory(segment_id)

                    if segment:
                        if segment_type and segment.type != segment_type:
                            continue
                        if metadata_filter:
                            match = all(segment.metadata.get(k) == v for k, v in metadata_filter.items())
                            if not match:
                                continue
                        
                        relevance = 1.0 - (float(dist)**2 / 2.0) # Assuming normalized embeddings
                        relevance = max(0.0, min(1.0, relevance))

                        if relevance >= min_relevance:
                            setattr(segment, 'relevance_score', relevance)
                            results.append(segment)
            
            results.sort(key=lambda s: getattr(s, 'relevance_score', 0.0), reverse=True)
            return results[:limit]

        except Exception as e:
            self.logger.error(f"Error in EnhancedMemoryManager search_memory: {e}", exc_info=True)
            return []

    async def remove_memory_segment(self, segment_id_to_remove: str) -> bool:
        """Removes a memory segment and re-indexes if necessary."""
        try:
            segment_exists = any(s.id == segment_id_to_remove for s in self.memory_segments)
            if not segment_exists:
                self.logger.warning(f"Segment with ID '{segment_id_to_remove}' not found for removal.")
                return False

            self.memory_segments = [s for s in self.memory_segments if s.id != segment_id_to_remove]

            # Check if the segment was in the FAISS index
            if segment_id_to_remove in self.id_to_faiss_idx:
                # Rebuild the index and mappings
                self.logger.info(f"Segment {segment_id_to_remove} removed. Rebuilding FAISS index.")
                self._rebuild_index() # This method handles clearing and re-adding all segments
            else:
                self.logger.info(f"Segment {segment_id_to_remove} was not in FAISS index or had no embedding. No re-index needed.")

            await self._save_to_disk() 
            self.logger.info(f"Successfully removed segment {segment_id_to_remove} and saved memory.")
            return True
        except Exception as e:
            self.logger.error(f"Error removing segment {segment_id_to_remove}: {e}", exc_info=True)
            return False

    def _rebuild_index(self) -> None:
        """Rebuilds the FAISS index from current memory segments."""
        if not self.is_initialized or self.embedding_model is None or self.vector_dim == 0:
            self.logger.warning("MemoryManager not fully initialized. Cannot rebuild index.")
            return

        self.logger.info("Rebuilding FAISS index...")
        
        # Reset existing index and mappings
        if self.index:
            self.index.reset()
        elif self.vector_dim > 0: # If index was None but should exist
            if RES and CUDA_DEVICE is not None and VECTOR_SEARCH_GPU:
                self.index = create_gpu_index(self.vector_dim, RES, CUDA_DEVICE)
            else:
                self.index = faiss.IndexFlatL2(self.vector_dim)
        else:
            self.logger.error("Cannot rebuild index: vector_dim is 0.")
            return
            
        self.faiss_idx_to_id.clear()
        self.id_to_faiss_idx.clear()

        # Add all segments with embeddings to the new index
        for segment in self.memory_segments:
            if segment.embedding:
                try:
                    embedding_np = np.array(segment.embedding, dtype=np.float32)
                    if embedding_np.ndim == 1 and len(embedding_np) == self.vector_dim:
                        # The _add_to_index method handles adding to self.index 
                        # and updating faiss_idx_to_id and id_to_faiss_idx
                        self._add_to_index(embedding_np, segment.id)
                    else:
                        self.logger.warning(f"Skipping segment {segment.id} during rebuild: invalid embedding shape or dimension.")
                except Exception as e:
                    self.logger.error(f"Error re-indexing segment {segment.id}: {e}", exc_info=True)
        
        if self.index:
            self.logger.info(f"FAISS index rebuilt. Total vectors: {self.index.ntotal}")
        else:
            self.logger.error("FAISS index is None after attempting rebuild.")