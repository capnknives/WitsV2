from setuptools import setup, find_packages

setup(
    name="wits-nexus",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "fastapi>=0.115.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
    ],
    python_requires=">=3.10",
)
