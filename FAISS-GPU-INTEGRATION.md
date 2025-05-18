# FAISS-GPU Integration Guide for WITS Nexus v2

## Overview
This guide helps you set up GPU-accelerated vector search using FAISS-GPU in WITS Nexus v2.
GPU acceleration can provide 10-100x performance improvements for large vector databases.

## Prerequisites
- NVIDIA GPU with CUDA support
- Anaconda or Miniconda installed (get it from https://docs.conda.io/en/latest/miniconda.html)
- PowerShell (Windows) or Bash (Linux/macOS)

## Quick Start

### Step 1: Preparation
1. Open PowerShell as Administrator
2. Navigate to the project:
   ```powershell
   Set-Location C:\WITS\wits_nexus_v2
   ```
3. If you haven't used conda in PowerShell before, run:
   ```powershell
   conda init powershell
   ```
   Then close and reopen PowerShell.

### Step 2: Setup FAISS-GPU
1. Create a new conda environment with Python 3.10 (specifically required):
   ```powershell
   conda create -n faiss_gpu_env2 python=3.10 -y
   conda activate faiss_gpu_env2
   ```

2. Install NumPy 1.24.3 (exact version required for compatibility):
   ```powershell
   pip install numpy==1.24.3
   ```

3. Install CUDA Toolkit 11.8 (specific version required for compatibility):
   ```powershell
   conda install -c conda-forge cudatoolkit=11.8 -y
   ```

4. Install PyTorch 2.0.1 with CUDA 11.8 support:
   ```powershell
   pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```

5. Install FAISS-GPU 1.7.4 with CUDA 11.8 support:
   ```powershell
   conda install -c conda-forge faiss-gpu=1.7.4 cudatoolkit=11.8 -y
   ```

6. Install the remaining project dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Step 2: Test the FAISS-GPU installation
1. Make sure the conda environment is activated:
   ```
   conda activate faiss_gpu_env2
   ```
2. Run the test script:
   ```
   python test_faiss_gpu.py
   ```
   If successful, you should see information about your GPU and a benchmark comparison
   between CPU and GPU performance.

## Step 3: Integrate with WITS Nexus v2
1. Run the integration script:
   ```
   python integrate_faiss_gpu.py
   ```
   This will create launcher scripts that will use the conda environment.

2. Test the integration:
   ```
   python test_integration.py
   ```
   This will verify that the WITS Nexus v2 project can use FAISS-GPU.

## Step 4: Run WITS Nexus v2 with FAISS-GPU
Simply run the start.bat script which will:
1. Set up and activate the faiss_gpu_env2 conda environment
2. Install FAISS-GPU and its dependencies
3. Activate the project's virtual environment (.venv)
4. Set up the necessary Python path
5. Start the web UI

```batch
.\start.bat
```

## Troubleshooting
- If you encounter CUDA out-of-memory errors, adjust the memory limit in `core/faiss_utils.py`:
  ```python
  resources.setTempMemory(512 * 1024 * 1024)  # Reduce to 512MB
  ```

- If FAISS-GPU is not properly detected:
  1. Make sure your NVIDIA drivers are up to date
  2. Verify that PyTorch can access the GPU:
     ```python
     import torch
     print(torch.cuda.is_available())
     ```

## Performance Considerations
- For small vector databases (<100K vectors), the CPU version might be fast enough
- For larger databases, GPU acceleration can provide 10-100x speedup
- Consider using multiple GPUs for very large databases (>10M vectors)

## Additional Resources
- FAISS documentation: https://github.com/facebookresearch/faiss/wiki
- PyTorch CUDA documentation: https://pytorch.org/docs/stable/cuda.html
