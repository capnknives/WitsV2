import sys
import os
import torch

def check_cuda_setup():
    print("Python Version:", sys.version)
    print("\nCUDA Environment Variables:")
    cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'PATH']
    for var in cuda_vars:
        print(f"{var}:", os.environ.get(var, 'Not set'))
    
    print("\nPyTorch CUDA Info:")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("Current CUDA device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))
        print("Device capability:", torch.cuda.get_device_capability())
        print("Device properties:", torch.cuda.get_device_properties(0))
    else:
        print("CUDA is not available!")
        print("PyTorch build info:", torch.__config__.show())

if __name__ == "__main__":
    check_cuda_setup()
