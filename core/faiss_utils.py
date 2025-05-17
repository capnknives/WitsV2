"""FAISS GPU utilities for vector search operations."""
import numpy as np
import torch
import faiss
import logging

logger = logging.getLogger('WITS.FAISS')

def initialize_gpu_resources():
    """Initialize FAISS GPU resources."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU support is required.")
        
    if not hasattr(faiss, 'StandardGpuResources'):
        raise RuntimeError("FAISS GPU support is not available. Please install faiss-gpu.")
        
    # Get CUDA device info
    cuda_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(cuda_device)
    total_memory = torch.cuda.get_device_properties(cuda_device).total_memory / (1024**3)  # Convert to GB
    
    logger.info(f"Initializing FAISS GPU support on device: {device_name}")
    logger.info(f"GPU memory available: {total_memory:.2f} GB")
    
    # Initialize GPU resources
    resources = faiss.StandardGpuResources()
    resources.setTempMemory(1024 * 1024 * 1024)  # Use 1GB GPU memory
    
    return resources, cuda_device

def create_gpu_index(dimension: int, res: faiss.StandardGpuResources, device: int):
    """Create a FAISS GPU index."""
    # Create CPU index first
    cpu_index = faiss.IndexFlatL2(dimension)
    
    # Convert to GPU
    gpu_index = faiss.index_cpu_to_gpu(res, device, cpu_index)
    
    return gpu_index

def add_vectors(index, vectors):
    """Add vectors to the FAISS index."""
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    vectors = vectors.astype(np.float32)
    index.add(vectors)
    return index.ntotal - 1  # Return the last index

def search_vectors(index, query_vector, k=5):
    """Search for similar vectors."""
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    query_vector = query_vector.astype(np.float32)
    distances, indices = index.search(query_vector, k)
    return distances[0], indices[0]  # Return first result only since we have 1 query
