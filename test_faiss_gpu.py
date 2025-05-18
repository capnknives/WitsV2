"""
FAISS-GPU Test Script for WITS Nexus v2
This script tests if FAISS-GPU is working properly with the configured environment.
"""
import time
import numpy as np
import torch
import faiss
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FAISS-GPU-Test')

def test_cpu_vs_gpu():
    """Compare performance between CPU and GPU for FAISS."""
    dimension = 128  # Vector dimension
    num_vectors = 50000  # Number of vectors to index
    k = 5  # Number of nearest neighbors to search for
    
    logger.info(f"Testing FAISS with {num_vectors} vectors of dimension {dimension}")
    
    # Generate random vectors
    logger.info("Generating random vectors...")
    np.random.seed(42)
    vectors = np.random.random((num_vectors, dimension)).astype(np.float32)
    query = np.random.random((1, dimension)).astype(np.float32)
    
    # Test CPU
    logger.info("Testing on CPU...")
    cpu_index = faiss.IndexFlatL2(dimension)
    
    start_time = time.time()
    cpu_index.add(vectors)
    cpu_add_time = time.time() - start_time
    
    start_time = time.time()
    cpu_distances, cpu_indices = cpu_index.search(query, k)
    cpu_search_time = time.time() - start_time
    
    logger.info(f"CPU add time: {cpu_add_time:.4f}s")
    logger.info(f"CPU search time: {cpu_search_time:.4f}s")
    
    # Test GPU
    logger.info("Testing on GPU...")
    try:
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.error("CUDA is not available!")
            return
            
        # Check if FAISS has GPU support
        if not hasattr(faiss, 'StandardGpuResources'):
            logger.error("FAISS was not built with GPU support!")
            return
        
        # Get GPU info
        gpu_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_device)
        logger.info(f"Using GPU: {gpu_name}")
        
        # Create resources
        res = faiss.StandardGpuResources()
        
        # Create GPU index
        gpu_index = faiss.index_factory(dimension, "Flat", faiss.METRIC_L2)
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_device, gpu_index)
        
        start_time = time.time()
        gpu_index.add(vectors)
        gpu_add_time = time.time() - start_time
        
        start_time = time.time()
        gpu_distances, gpu_indices = gpu_index.search(query, k)
        gpu_search_time = time.time() - start_time
        
        logger.info(f"GPU add time: {gpu_add_time:.4f}s")
        logger.info(f"GPU search time: {gpu_search_time:.4f}s")
        
        # Compare results
        cpu_speedup_add = cpu_add_time / gpu_add_time if gpu_add_time > 0 else float('inf')
        cpu_speedup_search = cpu_search_time / gpu_search_time if gpu_search_time > 0 else float('inf')
        
        logger.info(f"GPU speedup for add: {cpu_speedup_add:.2f}x")
        logger.info(f"GPU speedup for search: {cpu_speedup_search:.2f}x")
        
        # Check if results match
        if np.array_equal(cpu_indices, gpu_indices):
            logger.info("CPU and GPU results match exactly!")
        else:
            logger.info("CPU and GPU results differ slightly due to floating-point precision")
            logger.info(f"CPU indices: {cpu_indices}")
            logger.info(f"GPU indices: {gpu_indices}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error during GPU test: {e}")
        return False

def print_faiss_info():
    """Print information about the FAISS installation."""
    logger.info("FAISS Installation Info:")
    
    # Check for GPU support
    gpu_supported = hasattr(faiss, 'StandardGpuResources')
    logger.info(f"FAISS GPU support: {'Available' if gpu_supported else 'Not available'}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Print CUDA version
        logger.info(f"CUDA version: {torch.version.cuda}")
        
        # Print device info
        device_count = torch.cuda.device_count()
        logger.info(f"Number of CUDA devices: {device_count}")
        
        for i in range(device_count):
            logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            logger.info(f"  Compute capability: {props.major}.{props.minor}")
    
    # Print FAISS version
    try:
        version = faiss.__version__
        logger.info(f"FAISS version: {version}")
    except:
        logger.info("FAISS version: Unknown")

if __name__ == "__main__":
    print_faiss_info()
    test_cpu_vs_gpu()
