# Start WITS Nexus v2 application with FAISS-GPU support
Write-Host "Starting WITS Nexus v2..." -ForegroundColor Green
cd c:\WITS\wits_nexus_v2

# Check if environment exists
$envExists = $false
conda env list | ForEach-Object { if ($_ -match "faiss_gpu_env2") { $envExists = $true } }

if (-not $envExists) {
    Write-Host "Environment 'faiss_gpu_env2' not found. Setting it up now..." -ForegroundColor Yellow
    Write-Host "Creating conda environment with Python 3.10..." -ForegroundColor Cyan
    conda create -n faiss_gpu_env2 python=3.10 -y
    
    Write-Host "Installing NumPy 1.24.3..." -ForegroundColor Cyan
    conda activate faiss_gpu_env2
    pip install numpy==1.24.3
    
    Write-Host "Installing CUDA Toolkit 11.8..." -ForegroundColor Cyan
    conda install -c conda-forge cudatoolkit=11.8 -y
    
    Write-Host "Installing PyTorch 2.0.1 with CUDA 11.8..." -ForegroundColor Cyan
    pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
    
    Write-Host "Installing FAISS-GPU 1.7.4..." -ForegroundColor Cyan
    conda install -c conda-forge faiss-gpu=1.7.4 cudatoolkit=11.8 -y
    
    Write-Host "Installing remaining dependencies..." -ForegroundColor Cyan
    pip install -r requirements.txt
    
    Write-Host "Environment setup complete!" -ForegroundColor Green
}

# Activate conda environment
Write-Host "Activating conda environment..." -ForegroundColor Cyan
conda activate faiss_gpu_env2

# Remove any existing venv from PATH to prevent conflicts
$venvPath = Join-Path $PSScriptRoot "venv\Scripts"
if ($env:PATH -like "*$venvPath*") {
    Write-Host "Removing local venv from PATH to prevent conflicts..." -ForegroundColor Yellow
    $env:PATH = ($env:PATH -split ';' | Where-Object { $_ -ne $venvPath }) -join ';'
}

# Find and use Python 3.10 specifically
Write-Host "Finding Python 3.10 executable..." -ForegroundColor Cyan
$pythonPath = "$env:CONDA_PREFIX\python.exe"
if (Test-Path $pythonPath) {
    Write-Host "Using Python from conda environment: $pythonPath" -ForegroundColor Green
    
    # Verify Python version
    $pythonVersion = & $pythonPath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($pythonVersion.Trim() -eq "3.10") {
        Write-Host "Confirmed Python version: $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "WARNING: Expected Python 3.10 but found $pythonVersion" -ForegroundColor Yellow
    }
    
    # Run the application
    Write-Host "Running the application..." -ForegroundColor Green
    & $pythonPath run.py
} else {
    Write-Host "Error: Could not find Python executable in conda environment!" -ForegroundColor Red
    exit 1
}
