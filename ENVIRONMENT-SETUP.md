# WITS Nexus v2 Environment Setup Guide

## System Requirements

- NVIDIA GPU with CUDA support
- Windows/Linux with Python 3.10
- Anaconda or Miniconda installed

## Complete Environment Setup Instructions

The WITS Nexus v2 system requires specific versions of several components to work correctly. Follow these steps in the exact order specified:

### 1. Create a Conda Environment

```powershell
# Create a new environment with Python 3.10 (required)
conda create -n faiss_gpu_env2 python=3.10 -y
conda activate faiss_gpu_env2
```

### 2. Install NumPy

Install the specific version of NumPy required:

```powershell
pip install numpy==1.24.3
```

### 3. Install CUDA Toolkit

Install CUDA Toolkit 11.8 through conda:

```powershell
conda install -c conda-forge cudatoolkit=11.8 -y
```

### 4. Install PyTorch with CUDA Support

Install PyTorch 2.0.1 with CUDA 11.8 support:

```powershell
pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### 5. Install FAISS-GPU

Install FAISS-GPU 1.7.4 with CUDA 11.8 support:

```powershell
conda install -c conda-forge faiss-gpu=1.7.4 cudatoolkit=11.8 -y
```

### 6. Install Remaining Dependencies

Install all other project dependencies:

```powershell
pip install -r requirements.txt
```

### 7. Verify FAISS-GPU Installation

Run the test script to verify that FAISS-GPU is working correctly with CUDA:

```powershell
python test_faiss_gpu.py
```

You should see output confirming that FAISS is using your GPU.

### 8. Running the Application

Start the WITS Nexus v2 application:

```powershell
python run.py
```

## Troubleshooting

### FAISS Not Found

If you get a "ModuleNotFoundError: No module named 'faiss'" error:

1. Ensure you have activated the correct conda environment:
   ```
   conda activate faiss_gpu_env2
   ```

2. Verify FAISS-GPU is installed:
   ```
   conda list | grep faiss
   ```

3. If not found, reinstall FAISS-GPU:
   ```
   conda install -c conda-forge faiss-gpu=1.7.4 cudatoolkit=11.8 -y
   ```

### CUDA Not Available

If CUDA is not available or not detected:

1. Check CUDA installation:
   ```
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

2. Verify NVIDIA drivers are up-to-date.

3. Run the CUDA verification script:
   ```
   python setup_cuda.py
   ```

### System Design Note

The WITS Nexus v2 system is designed for goal-directed interactions rather than casual conversation. When interacting with the system, phrase your inputs as specific tasks or goals (e.g., "Analyze this code" or "Help me understand FAISS-GPU") rather than casual conversation starters like "Hello".
