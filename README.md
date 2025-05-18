# WITSAI (WITS-NEXUS v2)

WITSAI is an advanced AI system built on the WITS-NEXUS framework, designed to provide powerful, extensible, and ethical AI capabilities through a modular architecture.

## Requirements

- **Python 3.10.x** - This project requires Python 3.10 specifically for compatibility with FAISS-GPU and other dependencies
- **CUDA-compatible GPU** - For GPU-accelerated vector search and embeddings
- **Conda Environment** - Used to manage Python version and GPU-related dependencies

## Quick Start

### Running the Application

1. **Use the provided startup scripts** (recommended):
   ```
   .\start_wits.ps1   # PowerShell
   ```
   or
   ```
   start.bat          # Command Prompt
   ```
   These scripts automatically ensure the correct Python version and environment are used.

2. **Manual activation** (if needed):
   ```
   conda activate faiss_gpu_env2
   python run.py
   ```

3. **Verify your environment**:
   ```
   conda activate faiss_gpu_env2
   python verify_environment.py
   ```

## Key Features

- 🧠 **Advanced Memory Management** - Vector-based semantic search with FAISS-GPU
- 🛠️ **Extensible Tool System** - Easily add new capabilities through modular tools
- 🤖 **Multiple Specialized Agents** - Dedicated agents for different tasks
- 🌐 **Web Interface** - Modern FastAPI-based web interface
- 🔒 **Ethics Framework** - Built-in ethical considerations
- 📊 **Vector Memory** - Efficient storage and retrieval of conversations
- 🔄 **Async Operation** - High-performance asynchronous execution
- ⚡ **MCP Architecture** - Model-Chosen Parameters for intelligent tool usage
- 🔍 **Semantic Search** - Advanced context retrieval using embeddings
- 📝 **Structured Data** - Pydantic-based validation throughout
- 🔎 **Powerful Debugging** - Comprehensive logging, performance monitoring, and debug visualization

## Components

- **Orchestrator Agent**: Implements the ReAct loop (Reason-Act-Observe) for solving goals
- **Memory Manager**: Stores conversation history, segments, and goals with vector search
- **LLM Interface**: Handles communication with Ollama language models
- **Tool Registry**: Manages available tools that agents can use
- **Tools**:
  - Calculator Tool: Performs math operations safely
  - DateTime Tool: Provides current time with timezone conversion
  - File Tools: Read, write, and list files
  - Web Search Tool: Searches the web for information (when enabled)

## Architecture

WITS-NEXUS v2 uses a structured approach where:

1. The user provides a goal to the system
2. The Orchestrator Agent:
   - Builds a prompt with relevant context
   - Asks the LLM to produce a structured JSON response with:
     - A thought process (reasoning)
     - An action to take (tool to call or final answer)
   - Executes the chosen action
   - Observes the result and loops back to the LLM
3. This continues until a final answer is reached or max iterations are hit

## Setup and Usage

1.  **Prerequisites**:
    * Ensure you have Conda installed if you plan to use `faiss-gpu`.
    * Ensure Ollama is installed and running with the models specified in `config.yaml`.

2.  **Install Python Dependencies**:
    Install the core Python dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Faiss for Vector Search**:
    Faiss is crucial for the vector search capabilities. We strongly recommend using GPU support for better performance.

    * For **GPU support** (recommended):
        1. Install CUDA Toolkit from NVIDIA's website if not already installed
        2. Create and activate a dedicated conda environment with Python 3.10:
            ```bash
            conda create -n wits python=3.10
            conda activate wits
            ```
        3. Install faiss-gpu using conda-forge:
            ```bash
            conda install -c conda-forge faiss-gpu
            ```
        4. Install other project dependencies:
            ```bash
            pip install -r requirements.txt
            ```
        5. Verify GPU support:
            ```python
            python -c "import faiss; print('GPU support available:', faiss.get_num_gpus() > 0)"
            ```
        
        Always activate the conda environment before running the system:
        ```bash
        conda activate wits
        python run.py
        ```

    * For **CPU-only support**:
        If you don't have a compatible GPU or prefer CPU version:
        ```bash
        pip install faiss-cpu
        ```

    **Important Notes**:
    - Do not have both faiss-cpu and faiss-gpu installed simultaneously
    - If switching from CPU to GPU version:
        1. Uninstall faiss-cpu: `pip uninstall faiss-cpu`
        2. Follow the GPU support installation steps above
    - If switching from GPU to CPU version:
        1. Deactivate conda environment: `conda deactivate`
        2. Install faiss-cpu: `pip install faiss-cpu`

4.  **Run the system**:
    ```bash
    python run.py
    ```

5.  Enter your goal at the WITS v2 prompt.

## Project Structure

```
WITSAI/
├── agents/              # AI agent implementations
│   └── specialized/    # Specialized agent types
├── app/                # Web application
│   ├── routes/        # API routes
│   ├── static/        # Static files
│   └── templates/     # HTML templates
├── capabilities/       # System capabilities
├── core/              # Core system components
├── data/              # Data storage
│   ├── memory/       # Vector memory storage
│   └── user_files/   # User file storage
├── tools/             # Tool implementations
└── tests/             # Test suite
```

## Configuration

The system is configured through `config.yaml`. A template is provided in `config.yaml.example`. Key configuration areas include:

- Model settings (default and specialized agent models)
- Web interface configuration
- Memory management settings
- Tool configurations
- Security settings

## Debugging System

WITS-NEXUS v2 comes with a comprehensive debugging system that provides:

- **Structured Logging**: Hierarchical logging with configurable levels
- **Performance Monitoring**: Time tracking for critical operations
- **Debug Visualization**: Metrics display in web interface
- **Component-Specific Debug Options**: Fine-grained control via config
- **Debug Information Models**: Structured data for consistent debugging

The debug system can be configured in `config.yaml`:
```yaml
debug:
  enabled: true
  log_level: "DEBUG"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  console_logging_enabled: true
  file_logging_enabled: true
  log_directory: "logs"
  performance_monitoring: true
  components:
    llm_interface:
      log_prompts: true
      log_responses: true
    memory_manager:
      log_embeddings: true
      log_searches: true
    # Other component-specific settings...
```

## Security Considerations

- `data/memory/` contains conversation history and should be properly secured
- `config.yaml` may contain sensitive configuration - keep it secure
- User uploads in `data/user_files/` should be monitored
- Review `allow_code_execution` setting carefully before enabling

## Development

1. Create a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\Activate   # Windows
# or
source venv/bin/activate  # Linux/Mac
```

2. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

3. Run tests:
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

[MIT License](LICENSE)

## Acknowledgments

- Built on the WITS-NEXUS framework
- Uses Sentence Transformers for vector embeddings
- FAISS for efficient vector search
- FastAPI for modern web interface
- Pydantic for data validation