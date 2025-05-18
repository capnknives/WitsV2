"""
WITS Nexus v2 Environment Verification Script
This script checks that all required dependencies are correctly installed and configured.
"""

import sys
import platform
import importlib.util
import subprocess
from typing import Dict, Any, Tuple, Optional

def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def check_dependency(module_name: str, min_version: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Check if a Python module is installed and get its version."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False, None
        
        # Try to get version
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "Unknown")
            return True, version
        except:
            return True, "Version unknown"
    except:
        return False, None

def run_command(command: str) -> Tuple[bool, str]:
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()
    except Exception as e:
        return False, str(e)

def main() -> None:
    """Run all environment checks."""
    print_header("WITS Nexus v2 Environment Check")
    
    # System information
    print(f"Python version: {platform.python_version()}")
    print(f"Operating system: {platform.system()} {platform.version()}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # Check conda environment
    success, output = run_command("conda info --env")
    if success:
        print("\nConda environments:")
        print(output)
    else:
        print("\nConda environments: Failed to retrieve")
    
    # Required libraries
    libraries = [
        ("numpy", "1.24.3"),
        ("torch", "2.0.1"),
        ("faiss", "1.7.4"),
        ("pyyaml", None),
        ("pydantic", None),
        ("fastapi", None),
        ("uvicorn", None),
        ("jinja2", None),
        ("aiofiles", None),
        ("transformers", "4.31.0"),
        ("sentence-transformers", "2.2.2"),
        ("huggingface-hub", None),
    ]
    
    print_header("Library Dependencies")
    
    all_ok = True
    for lib, expected_version in libraries:
        installed, version = check_dependency(lib)
        status = "✓ Installed" if installed else "✗ Not found"
        version_info = f"(v{version})" if version else ""
        
        if not installed:
            all_ok = False
            print(f"{status} {lib} {version_info}")
        elif expected_version and version != expected_version:
            all_ok = False
            print(f"{status} {lib} {version_info} - WARNING: Expected v{expected_version}")
        else:
            print(f"{status} {lib} {version_info}")
    
    # Check CUDA availability
    print_header("CUDA and GPU Information")
    
    cuda_ok, torch_cuda = False, False
    
    installed, _ = check_dependency("torch")
    if installed:
        try:
            import torch
            torch_cuda = torch.cuda.is_available()
            if torch_cuda:
                print(f"✓ PyTorch CUDA support available")
                print(f"  - CUDA version: {torch.version.cuda}")
                print(f"  - Device count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
                    print(f"    Capability: {torch.cuda.get_device_capability(i)}")
                    props = torch.cuda.get_device_properties(i)
                    print(f"    Memory: {props.total_memory / (1024**3):.2f} GB")
            else:
                all_ok = False
                print(f"✗ PyTorch CUDA support NOT available")
        except Exception as e:
            all_ok = False
            print(f"✗ Error checking PyTorch CUDA: {e}")
    
    installed, _ = check_dependency("faiss")
    if installed:
        try:
            import faiss
            if hasattr(faiss, "StandardGpuResources"):
                print(f"✓ FAISS GPU support available")
                cuda_ok = True
                
                # Try to initialize GPU resources
                try:
                    res = faiss.StandardGpuResources()
                    print(f"✓ FAISS GPU resources initialized successfully")
                except Exception as e:
                    all_ok = False
                    print(f"✗ Failed to initialize FAISS GPU resources: {e}")
            else:
                all_ok = False
                print(f"✗ FAISS installed but without GPU support")
        except Exception as e:
            all_ok = False
            print(f"✗ Error checking FAISS: {e}")
    
    # Final assessment
    print_header("Assessment")
    if all_ok and cuda_ok and torch_cuda:
        print("✅ All dependencies are correctly installed!")
        print("✅ GPU acceleration is properly configured!")
        print("✅ WITS Nexus v2 should function correctly with FAISS-GPU.")
    else:
        print("⚠️ Some issues were detected:")
        if not all_ok:
            print("  - Not all dependencies are correctly installed or versioned")
        if not cuda_ok:
            print("  - FAISS GPU support is not properly configured")
        if not torch_cuda:
            print("  - PyTorch CUDA support is not available")
        print("\nPlease check the detailed output above and refer to ENVIRONMENT-SETUP.md")

if __name__ == "__main__":
    main()
