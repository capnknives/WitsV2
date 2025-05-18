@echo off
echo Running with correct Python environment...

:: Activate conda environment
call conda activate faiss_gpu_env2

:: Get the conda Python path
for /f "tokens=*" %%a in ('where python') do (
    set PYTHON_PATH=%%a
    goto :found_python
)

:found_python
echo Using Python: %PYTHON_PATH%

:: Run the command with the correct Python
%PYTHON_PATH% %*

exit /b %errorlevel%
