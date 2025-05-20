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
                raise ValueError("Invalid embedding dimension! x.x")
            self.vector_dim = dim
            
            # Create our FAISS index with GPU support if available! SPEEEEED! \\o/
            if RES is not None and CUDA_DEVICE is not None:
                self.index = create_gpu_index(self.vector_dim, RES, CUDA_DEVICE)
                self.logger.info("GPU-powered FAISS index created! Time to go FAST! =D")
            else:
                # Fallback to CPU if no GPU available x.x
                self.index = faiss.IndexFlatL2(self.vector_dim)
                self.logger.warning("No GPU available - using CPU index instead >.>")
            
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

        if not self.vector_model_name:
            self.logger.warning("No vector model name! We're flying blind! >.>")
            return

        self.logger.info(f"Powering up vector search! Loading {self.vector_model_name}... =D")
        try:
            # Time to load our embedding model! *sparkles* \\o/
            self.embedding_model = SentenceTransformer(self.vector_model_name)
            self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.logger.info(f"Vector dimension set to {self.vector_dim}! That's a lot of numbers! ^_^")
            
            if not isinstance(self.vector_dim, int):
                raise ValueError(f"Vector dimension should be a number, not {self.vector_dim}! Math is hard x.x")

            # FAISS index time! GPU power activate! =P
            self.index = create_gpu_index(self.vector_dim, RES, CUDA_DEVICE)
            self.logger.info(f"GPU-powered FAISS index is ready on device {CUDA_DEVICE}! Let's goooo! \\o/")
            
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
            distances, indices = self._search_vectors(
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
        """
        Let's see how our vector search is doing! Status check time! =D
        Returns a report card of our vector search capabilities! O.o
        """
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
                    f"Type: {segment.segment_type}\n"
                    f"Type: {segment.type}\n"
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
        if not os.path.exists(self.memory_file):
            self.logger.warning(f"Memory file not found: {self.memory_file}. Starting with empty memory.")
            return
        
        try:
            async with aiofiles.open(self.memory_file, 'r') as f:
                contents = await f.read()
                if not contents.strip():
                    self.logger.warning("Memory file is empty! Nothing to load.")
                    return
                
                # Deserialize the JSON data
                loaded_segments = json.loads(contents)
                for seg_dict in loaded_segments:
                    try:
                        # Convert timestamp back to datetime object
                        if 'timestamp' in seg_dict and isinstance(seg_dict['timestamp'], str):
                            seg_dict['timestamp'] = datetime.fromisoformat(seg_dict['timestamp'])
                        
                        # Recreate the MemorySegment object
                        segment = MemorySegment.model_validate(seg_dict)
                        self.segments.append(segment)
                        
                        # Add to FAISS index
                        if self.index is not None and segment.embedding is not None:
                            self.index.add(np.array([segment.embedding]).astype('float32'))
                        
                        self.logger.info(f"Loaded segment: ID={segment.id}, Type='{segment.type}', Source='{segment.source}'")
                    except Exception as e:
                        self.logger.error(f"Error loading segment from disk: {e}", exc_info=True)
            
            self.logger.info(f"Loaded {len(loaded_segments)} memory segments from disk.")
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
            List[MemorySegment]: The most relevant memories we found! \\o/
        """
        try:
            if not self.is_initialized:
                self.logger.warning("Memory system not initialized! Hold up...attempting to fix that! >.<")
                await self.initialize_db()
                if not self.is_initialized:
                    raise RuntimeError("Failed to initialize memory system! x.x")

            if not self.index or len(self.memory_segments) == 0:
                self.logger.warning("No memories to search through yet! >.<")
                return []

            # Get that query vector! =D
            query_vector = await self.get_embedding(query)
            if query_vector is None:
                self.logger.error("Failed to create query vector! The math spirits are angry today >.>")
                return []

            # Time for some FAISS magic! \\o/
            distances, indices = self.index.search(
                np.array([query_vector]).astype('float32'), 
                k=min(limit * 2, len(self.memory_segments))
            )

            # Process results and apply relevance filtering =^.^=
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0 or idx >= len(self.memory_segments):
                    continue
                    
                relevance = 1.0 - (dist / 2.0)  # Convert distance to similarity score \o/
                if relevance < min_relevance:
                    continue

                memory = self.memory_segments[idx]
                memory.relevance_score = relevance
                results.append(memory)

                if len(results) >= limit:
                    break

            if self.debug_enabled and self.debug_component_config.get('log_searches'):
                self.logger.debug(
                    f"Search complete! Found {len(results)} memories! =D\n"
                    f"Query: {query}\n"
                    f"Top relevance score: {results[0].relevance_score if results else 'N/A'}"
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
            
        Returns:
            MemorySegment if found, None if not found x.x
        """
        try:
            for segment in self.memory_segments:
                if segment.id == memory_id:
                    return segment
            return None
        except Exception as e:
            self.logger.error(f"Failed to get memory! ID={memory_id}: {str(e)} x.x")
            return None

    async def remove_memory(self, memory_id: str) -> bool:
        """
        Remove a memory from our collection! Sometimes we gotta let go... T.T
        
        Args:
            memory_id: The ID of the memory to remove >.>
            
        Returns:
            bool: True if removed successfully, False if failed x.x
        """
        try:
            for idx, segment in enumerate(self.memory_segments):
                if segment.id == memory_id:
                    # Remove from memory segments list
                    self.memory_segments.pop(idx)
                    
                    # Remove from FAISS index if we have one
                    if self.index:
                        self.index = faiss.IndexFlatL2(self.vector_dim)  # Create new index
                        # Rebuild index with remaining vectors
                        if self.memory_segments:
                            vectors = np.array([s.embedding for s in self.memory_segments if s.embedding]).astype('float32')
                            if len(vectors) > 0:
                                self.index.add(vectors)
                    
                    self.logger.info(f"Memory removed successfully! ID={memory_id} \\o/")
                    return True
                    
            self.logger.warning(f"Memory not found! ID={memory_id} x.x")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to remove memory! ID={memory_id}: {str(e)} >.<")
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
                    raise RuntimeError("Failed to initialize embedding model! x.x")

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
        Search for similar vectors using FAISS! Magic time! \\o/
        
        Args:
            query_vector: The vector to search for ^.^
            k: How many results we want =D
            
        Returns:
            Tuple of (distances, indices) arrays \\o/
        """
        if not self.index:
            raise RuntimeError("No index available for search! x.x")
            
        try:
            # Make sure vector is the right shape and type =D
            query_vector = query_vector.reshape(1, -1).astype(np.float32)
            
            # Get number of results we can actually return \o/
            n = min(k, self.index.ntotal if hasattr(self.index, 'ntotal') else k)
            
            # Set up our output arrays =^.^=
            distances = np.empty((1, n), dtype=np.float32)
            indices = np.empty((1, n), dtype=np.int64)
            
            # Do the search! =D
            if isinstance(self.index, faiss.GpuIndexFlatL2):
                # GPU index \o/
                self.index.assign(query_vector, n, indices)
                # Calculate distances
                for i in range(n):
                    if indices[0, i] >= 0:
                        recons = np.empty(self.vector_dim, dtype=np.float32)
                        self.index.reconstruct(int(indices[0, i]), recons)
                        diff = query_vector[0] - recons
                        distances[0, i] = float(np.dot(diff, diff))
                    else:
                        distances[0, i] = float('inf')
            else:
                # CPU index ^.^
                self.index.assign(query_vector, n, indices)
                # Calculate distances
                for i in range(n):
                    if indices[0, i] >= 0:
                        recons = np.empty(self.vector_dim, dtype=np.float32)
                        self.index.reconstruct(int(indices[0, i]), recons)
                        diff = query_vector[0] - recons
                        distances[0, i] = float(np.dot(diff, diff))
                    else:
                        distances[0, i] = float('inf')
            
            return distances, indices
            
        except Exception as e:
            self.logger.error(f"Vector search failed! FAISS is being weird x.x: {str(e)}", exc_info=True)
            raise

    def _add_vectors(self, vectors: np.ndarray) -> None:
        """
        Add vectors to our FAISS index! Adding more brain power! =D
        
        Args:
            vectors: The vectors to add \\o/
        """
        if not self.index:
            raise RuntimeError("No index available for adding vectors! x.x")
            
        try:
            # Make sure vectors are the right type
            vectors_array = vectors.astype(np.float32)
            n = vectors_array.shape[0]
            
            # Add the vectors! =D
            if isinstance(self.index, faiss.GpuIndexFlatL2):
                faiss.downcast_index(self.index).add(vectors_array)
            else:
                faiss.downcast_index(self.index).add(vectors_array)
                
        except Exception as e:
            self.logger.error(f"Failed to add vectors! Math overflow? x.x: {str(e)}", exc_info=True)
            raise

    def _rebuild_index(self) -> None:
        """
        Rebuild the FAISS index from scratch! Spring cleaning time! =D
        """
        try:
            # Get all our vectors ready \\o/
            vectors = [
                np.array(segment.embedding)
                for segment in self.memory_segments
                if segment.embedding is not None
            ]
            
            if vectors:
                vectors_array = np.stack(vectors).astype(np.float32)
                
                # Create and populate new index ^.^
                if RES is not None and CUDA_DEVICE is not None:
                    new_index = create_gpu_index(self.vector_dim, RES, CUDA_DEVICE)
                else:
                    new_index = faiss.IndexFlatL2(self.vector_dim)
                
                # Add vectors to new index \o/
                if isinstance(new_index, faiss.GpuIndexFlatL2):
                    faiss.downcast_index(new_index).add(vectors_array)
                else:
                    faiss.downcast_index(new_index).add(vectors_array)
                
                # Switch to the new index! =D
                self.index = new_index
                
                if self.debug_enabled and self.debug_component_config.get('log_initialization'):
                    self.logger.debug(
                        f"Index rebuilt! Stats:\n"
                        f"- Total vectors: {len(vectors)}\n"
                        f"- Vector dimension: {self.vector_dim}\n"
                        f"- Using GPU: {isinstance(new_index, faiss.GpuIndexFlatL2)} \\o/"
                    )
            else:
                # No vectors? Create empty index x.x
                if RES is not None and CUDA_DEVICE is not None:
                    self.index = create_gpu_index(self.vector_dim, RES, CUDA_DEVICE)
                else:
                    self.index = faiss.IndexFlatL2(self.vector_dim)
            
        except Exception as e:
            self.logger.error(f"Failed to rebuild index! Everything is broken! x.x: {str(e)}", exc_info=True)
            raise