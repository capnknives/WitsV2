# WITS Nexus v2 Usage Guide

## Running the Application

### Important: Python Version Requirement

This project **requires Python 3.10** specifically for compatibility with FAISS-GPU and other dependencies. Using other Python versions will result in errors or reduced functionality.

### Recommended Method

The simplest way to run WITS Nexus v2 is to use the provided startup scripts:

1. **PowerShell** (recommended for Windows):
   ```powershell
   .\start_wits.ps1
   ```
   
2. **Command Prompt** (alternative for Windows):
   ```
   start.bat
   ```

These scripts will:
- Check if the conda environment exists and create it if needed
- Activate the correct conda environment with Python 3.10
- Install necessary dependencies
- Run the application with the correct Python version

### Manual Execution

If you need to run commands manually:

1. **Activate the conda environment**:
   ```
   conda activate faiss_gpu_env2
   ```

2. **Run the application**:
   ```
   python run.py
   ```

3. **Verify the environment** (if you're having issues):
   ```
   python verify_environment.py
   ```
   This will check if all required dependencies are properly installed.

4. **Test FAISS-GPU integration** (for GPU-related issues):
   ```
   python test_faiss_gpu.py
   ```
   This will verify if FAISS-GPU is working properly with your GPU.

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'faiss'"**
   - Solution: Make sure you're using the conda environment
   ```
   conda activate faiss_gpu_env2
   ```

2. **Using wrong Python version**
   - Check your Python version:
   ```
   python --version
   ```
   - If not showing 3.10.x, make sure you're using the conda environment

3. **CUDA/GPU not detected**
   - Verify CUDA installation:
   ```
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

4. **Environment setup issues**
   - Recreate the environment using the start script, or follow the steps in FAISS-GPU-INTEGRATION.md

Remember: Always use the startup scripts whenever possible to ensure the correct environment and Python version are being used.
