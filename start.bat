@echo off
echo Setting up WITS-NEXUS with FAISS-GPU integration...

:: Check if running in a virtual environment
if defined VIRTUAL_ENV (
    echo Error: Please deactivate any active virtual environment first
    echo Run 'deactivate' and try again
    exit /b 1
)

:: Activate conda base environment first (needed for conda commands)
call conda activate base || (
    echo Error: Could not activate conda base environment
    exit /b 1
)

:: Create and activate faiss-gpu conda environment if it doesn't exist
conda env list | findstr /C:"faiss_gpu_env2" > nul
if errorlevel 1 (
    echo Creating faiss_gpu_env2 conda environment...
    call conda create -n faiss_gpu_env2 python=3.10 -y || (
        echo Error: Could not create conda environment
        exit /b 1
    )
)

:: Activate faiss_gpu_env2 and install FAISS-GPU
call conda activate faiss_gpu_env2 || (
    echo Error: Could not activate faiss_gpu_env2 environment
    exit /b 1
)

:: Install faiss-gpu using conda to ensure compatibility
call conda install -y faiss-gpu cudatoolkit=11.3 || (
    echo Error: Could not install faiss-gpu
    exit /b 1
)

:: Create virtual environment within the conda environment
python -m venv .venv || (
    echo Error: Could not create virtual environment
    exit /b 1
)

:: Activate virtual environment
call .venv\Scripts\activate || (
    echo Error: Could not activate virtual environment
    exit /b 1
)

:: Install FAISS-GPU and core ML dependencies in conda environment
call conda install -c pytorch -c nvidia faiss-gpu=1.7.2 pytorch=1.13.0 cudatoolkit=11.6 numpy=1.24.3 -y || (
    echo Error: Could not install FAISS-GPU dependencies
    exit /b 1
)

:: Create new virtual environment using conda's Python
echo Creating new virtual environment with Python 3.10...
python -m venv .venv || (
    echo Error: Could not create virtual environment
    exit /b 1
)

:: Activate virtual environment and verify activation
call .venv\Scripts\activate || (
    echo Error: Could not activate virtual environment
    exit /b 1
)

echo Verifying virtual environment activation...
python -c "import sys; print('Virtual environment:', sys.prefix)"

:: Upgrade pip and install core packages
python -m pip install --upgrade pip wheel setuptools || (
    echo Error: Could not upgrade pip and core packages
    exit /b 1
)

:: Install core dependencies in the correct order
python -m pip install typing-extensions==4.13.2 || (
    echo Error: Could not install typing-extensions
    exit /b 1
)

python -m pip install pydantic==2.11.4 || (
    echo Error: Could not install pydantic
    exit /b 1
)

:: Install remaining requirements
python -m pip install -r requirements.txt || (
    echo Error: Could not install requirements
    exit /b 1
)

:: Install core dependencies first with specific versions
pip uninstall typing-extensions -y
pip install --no-cache-dir typing-extensions==4.7.1
pip install numpy==1.24.3
pip install torch==1.13.1
pip install transformers==4.31.0
pip install sentence-transformers==2.2.2

:: Install non-critical requirements without dependencies first
pip install --no-deps -r requirements.txt

:: Now install remaining dependencies (excluding already installed packages)
echo Installing remaining dependencies...
for /f "tokens=*" %%i in ('pip freeze') do (
    if not "%%i"=="numpy==1.24.3" if not "%%i"=="torch==1.13.1" if not "%%i"=="typing-extensions==4.7.1" if not "%%i"=="transformers==4.31.0" if not "%%i"=="sentence-transformers==2.2.2" (
        pip install "%%i"
    )
)

:: Add PYTHONPATH to include conda environment site-packages
set PYTHONPATH=%CONDA_PREFIX%\Lib\site-packages;%PYTHONPATH%

:: Start the web UI
echo Starting Web UI...
python run.py

:: Verify the environment is set up correctly
echo Verifying environment setup...
python -c "import torch; import faiss; import typing_extensions; print('Environment verified successfully!')" || (
    echo Error: Environment verification failed
    exit /b 1
)

echo WITS-NEXUS environment setup complete!
echo To start the application, run: python run.py

:: Keep the window open if there's an error
if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to exit.
    pause > nul
)
