@echo off
SETLOCAL

echo Setting up WITS-NEXUS with FAISS-GPU integration...

REM Check if conda is available by trying to run 'conda --version'
echo Checking for conda...
conda --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo.
    echo ERROR: 'conda' command not recognized.
    echo Please ensure Anaconda or Miniconda is installed and configured correctly.
    echo You may need to:
    echo   1. Add Conda's script directory (e.g., C:\Users\YourUser\anaconda3\Scripts or Miniconda3\Scripts) to your system PATH.
    echo   2. Run this script from an Anaconda Prompt or Anaconda Powershell.
    echo   3. Initialize your shell for conda (e.g., run 'conda init cmd.exe' or 'conda init powershell' in an Anaconda Prompt, then restart your shell).
    echo.
    pause
    exit /b 1
)
echo Conda command found. Proceeding with environment setup.

REM Define environment name and project root with exact dependency versions
SET "ENV_NAME=faiss_gpu_env2"
SET "PROJECT_ROOT=c:\WITS\wits_nexus_v2"
SET "PYTHON_VERSION=3.10"
SET "NUMPY_VERSION=1.24.3"
SET "CUDA_VERSION=11.8"
SET "TORCH_VERSION=2.0.1+cu118"
SET "TORCHVISION_VERSION=0.15.2+cu118"
SET "FAISS_VERSION=1.7.4"

REM Attempt to activate the conda environment
echo Activating %ENV_NAME% conda environment...
call conda activate %ENV_NAME%

REM Check if activation was successful by checking CONDA_PREFIX
SET "ACTIVATION_SUCCESSFUL=0"
IF NOT "%CONDA_PREFIX%"=="" (
    echo "%CONDA_PREFIX%" | findstr /I /C:"%ENV_NAME%" >nul
    IF NOT ERRORLEVEL 1 (
        SET "ACTIVATION_SUCCESSFUL=1"
    )
)

IF "%ACTIVATION_SUCCESSFUL%"=="1" (
    echo Environment '%ENV_NAME%' is active: %CONDA_PREFIX%
    echo Ensuring dependencies are installed/updated in '%ENV_NAME%'...
    call conda activate %ENV_NAME% && pip install -r "%PROJECT_ROOT%\requirements.txt"
    IF ERRORLEVEL 1 (
        echo ERROR: Failed to install/update dependencies from requirements.txt in '%ENV_NAME%'.
        pause
        exit /b 1
    )
    echo Dependencies installed/updated successfully.
) ELSE (
    echo Environment '%ENV_NAME%' not active or not found. Attempting to create and activate...

    echo Creating %ENV_NAME% conda environment with Python %PYTHON_VERSION%...
    call conda create -n %ENV_NAME% python=%PYTHON_VERSION% -c conda-forge -y
    IF ERRORLEVEL 1 (
        echo ERROR: Conda environment '%ENV_NAME%' creation failed.
        pause
        exit /b 1
    )
    echo Environment '%ENV_NAME%' created.

    echo Activating newly created '%ENV_NAME%' environment...
    call conda activate %ENV_NAME%
    
    SET "ACTIVATION_SUCCESSFUL_AFTER_CREATE=0"
    IF NOT "%CONDA_PREFIX%"=="" (
        echo "%CONDA_PREFIX%" | findstr /I /C:"%ENV_NAME%" >nul
        IF NOT ERRORLEVEL 1 (
            SET "ACTIVATION_SUCCESSFUL_AFTER_CREATE=1"
        )
    )

    IF "%ACTIVATION_SUCCESSFUL_AFTER_CREATE%"=="1" (
        echo Environment '%ENV_NAME%' activated successfully after creation: %CONDA_PREFIX%
    ) ELSE (
        echo ERROR: Failed to activate '%ENV_NAME%' even after creation.
        echo Current CONDA_PREFIX: %CONDA_PREFIX%
        echo Please check your conda installation and try activating manually: conda activate %ENV_NAME%
        pause
        exit /b 1
    )

    echo Installing CUDA 11.8 and FAISS-GPU 1.7.4 into '%ENV_NAME%'...
    call conda activate %ENV_NAME% && conda install -c conda-forge cudatoolkit=11.8 faiss-gpu=1.7.4 -y
    IF ERRORLEVEL 1 (
        echo ERROR: CUDA or FAISS-GPU installation failed.
        pause
        exit /b 1
    )
    echo CUDA and FAISS-GPU installed.

    echo Installing dependencies from requirements.txt into '%ENV_NAME%'...
    call conda activate %ENV_NAME% && pip install -r "%PROJECT_ROOT%\requirements.txt"
    IF ERRORLEVEL 1 (
        echo ERROR: Failed to install dependencies from requirements.txt.
        pause
        exit /b 1
    )
    echo Dependencies installed successfully.
)

echo.
echo Launching WITS-NEXUS application in a new window...

REM Find specific Python 3.10 executable in conda environment
echo Verifying environment uses Python 3.10...
FOR /F "tokens=* USEBACKQ" %%F IN (`conda run -n %ENV_NAME% --no-capture-output python -c "import sys; import os; print(os.path.join(os.environ['CONDA_PREFIX'], 'python.exe'))"`) DO (
    SET CONDA_PYTHON_PATH=%%F
)

REM Ensure the new window also activates the environment and changes to the project directory
REM and explicitly uses the conda environment's Python 3.10 executable
echo Using Python executable: %CONDA_PYTHON_PATH%
start "WITS-NEXUS Server" cmd /k "conda activate %ENV_NAME% && cd /D "%PROJECT_ROOT%" && "%CONDA_PYTHON_PATH%" run.py && pause"

IF ERRORLEVEL 1 (
    echo.
    echo ERROR: Failed to launch WITS-NEXUS application (run.py) in a new window.
    echo Please check if a new window appeared and if there were errors there.
    pause
    exit /b 1
)

echo.
echo WITS-NEXUS application is launching in a new window.
echo This script (start.bat) will now pause. Press any key to exit this script.
echo The server will continue running in its own window.

ENDLOCAL
pause
