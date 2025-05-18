# IMPORTANT: PYTHON VERSION REQUIREMENTS

This project requires Python 3.10 specifically for compatibility with FAISS-GPU and other dependencies.

## RECOMMENDED USAGE

1. Use the start_wits.ps1 or start.bat script to launch the application
   These scripts will ensure the correct Python version is used.

2. If you need to run Python directly, activate the conda environment first:
   ```powershell
   conda activate faiss_gpu_env2
   python run.py
   ```

3. To verify the environment is correctly set up:
   ```powershell
   conda activate faiss_gpu_env2
   python verify_environment.py
   ```

## TROUBLESHOOTING PYTHON VERSION CONFLICTS

If you're experiencing issues with Python versions getting mixed up:

### Issue: Mixed Python Environments

This project may have both:
- A conda environment (faiss_gpu_env2) with Python 3.10
- A local venv with Python 3.13 in the project directory

Even when the conda environment is activated, the `python` command may still use the wrong version if venv is earlier in your PATH.

### Solutions

#### Option 1: Use our helper script to fix PATH

Run this script before any Python commands:
```powershell
.\fix_python_env.ps1
```

This will fix your PATH to prioritize the conda Python.

#### Option 2: Use the full path to the conda Python executable

Instead of just `python`, use:
```powershell
& $env:CONDA_PREFIX\python.exe run.py
```

#### Option 3: Use python_correct.bat

Run:
```
python_correct.bat run.py
```

### How to verify correct Python version

```powershell
python -c "import sys; print(sys.version)"
```

You should see Python 3.10.x, not 3.13.x.
