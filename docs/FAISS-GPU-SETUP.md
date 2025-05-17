# FAISS GPU Setup Guide

This document outlines the setup process for FAISS GPU support in WITS-NEXUS v2.

## Prerequisites

1. CUDA Toolkit (from NVIDIA website)
2. Conda package manager

## Environment Setup

1. Create and activate a dedicated conda environment:
```bash
conda create -n wits python=3.10
conda activate wits
```

2. Install FAISS-GPU:
```bash
conda install -c conda-forge faiss-gpu
```

3. Install project dependencies:
```bash
pip install -r requirements.txt
```

## Important Notes

- Do not have both faiss-cpu and faiss-gpu installed simultaneously
- NumPy must be version 1.x (FAISS is not yet compatible with NumPy 2.x)
- Always activate the conda environment before running the system:
```bash
conda activate wits
python run.py
```

## Verification

You can verify GPU support is working by running:
```python
python -c "import faiss; print('GPU support available:', faiss.get_num_gpus() > 0)"
```

## Troubleshooting

1. If you see NumPy errors, ensure you're using NumPy 1.x:
```bash
pip install "numpy<2" --force-reinstall
```

2. If FAISS doesn't detect GPU:
   - Verify CUDA Toolkit is installed
   - Ensure you're in the correct conda environment
   - Check that only faiss-gpu (not faiss-cpu) is installed
