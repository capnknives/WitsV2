# Fix Python environment for WITS Nexus v2
Write-Host "Fixing Python environment for WITS Nexus v2..." -ForegroundColor Green

# Activate conda environment
conda activate faiss_gpu_env2

# Determine the conda Python path
$condaPythonPath = Join-Path $env:CONDA_PREFIX "python.exe"

# Check if the path exists
if (Test-Path $condaPythonPath) {
    Write-Host "Found conda Python: $condaPythonPath" -ForegroundColor Green
    
    # Check Python version
    $pythonVersion = & $condaPythonPath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    Write-Host "Python version: $pythonVersion" -ForegroundColor Cyan
    
    # Remove any existing venv from PATH
    $venvPath = Join-Path $PSScriptRoot "venv\Scripts"
    $env:PATH = ($env:PATH -split ';' | Where-Object { $_ -ne $venvPath }) -join ';'
    
    # Add conda Python path to the beginning of PATH
    $condaBinPath = Split-Path -Parent $condaPythonPath
    $env:PATH = "$condaBinPath;$env:PATH"
    
    Write-Host "Environment PATH updated. Now using conda Python as default." -ForegroundColor Green
    Write-Host "Current Python path: $(Get-Command python | Select-Object -ExpandProperty Source)" -ForegroundColor Cyan
} else {
    Write-Host "Error: Could not find Python in conda environment!" -ForegroundColor Red
}
