#!/usr/bin/env python
"""
WITS Nexus v2 Python Version Check Script
This script verifies that the correct version of Python (3.10.x) is being used.
"""

import sys
import platform
import os
import subprocess

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def check_conda_environment():
    """Check if running in a conda environment and which one."""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    conda_prefix = os.environ.get('CONDA_PREFIX')
    
    if conda_env:
        print(f"Current conda environment: {conda_env}")
        print(f"Conda environment path: {conda_prefix}")
        return True
    else:
        print("Not running in a conda environment")
        return False

def check_python_version():
    """Check if Python version is 3.10.x."""
    version = platform.python_version()
    print(f"Current Python version: {version}")
    
    major, minor, _ = map(int, version.split('.', 2))
    
    if major == 3 and minor == 10:
        print("✓ CORRECT: Using Python 3.10.x")
        return True
    else:
        print("✗ INCORRECT: This project requires Python 3.10.x")
        return False

def find_python310():
    """Try to locate Python 3.10 on the system."""
    print("\nSearching for Python 3.10 installations:")
    
    # Check for conda environments with Python 3.10
    try:
        result = subprocess.run(
            ["conda", "info", "--envs"], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            print("Found the following conda environments:")
            for line in result.stdout.splitlines():
                if "python=3.10" in line or "py310" in line.lower() or "python310" in line.lower():
                    print(f"  - {line} (likely contains Python 3.10)")
    except:
        print("Could not retrieve conda environments.")
    
    # Check common Python installation paths
    print("\nPotential Python 3.10 installation paths:")
    paths_to_check = [
        r"C:\Python310",
        r"C:\Program Files\Python310",
        r"C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310",
        # For conda environments
        r"C:\Users\%USERNAME%\anaconda3\envs\*\python.exe",
        r"C:\Users\%USERNAME%\miniconda3\envs\*\python.exe",
    ]
    
    for path in paths_to_check:
        print(f"  - {path}")

def print_recommendation():
    """Print a recommendation message."""
    print("\nRECOMMENDATION:")
    print("1. Use the start_wits.ps1 or start.bat script to launch the application")
    print("   These scripts will ensure the correct Python version is used.")
    print("\n2. If you need to run Python directly, activate the conda environment first:")
    print("   conda activate faiss_gpu_env2")
    print("   python run.py")
    print("\n3. To verify the environment is correctly set up:")
    print("   conda activate faiss_gpu_env2")
    print("   python verify_environment.py")

if __name__ == "__main__":
    print_header("WITS Nexus v2 Python Version Check")
    
    in_conda = check_conda_environment()
    correct_version = check_python_version()
    
    print_header("System Details")
    print(f"Executable: {sys.executable}")
    print(f"Version string: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    if not correct_version:
        find_python310()
    
    print_header("Recommendation")
    print_recommendation()
