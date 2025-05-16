# WITSAI (WITS-NEXUS v2)

WITSAI is an advanced AI system built on the WITS-NEXUS framework, designed to provide powerful, extensible, and ethical AI capabilities through a modular architecture.

## Key Features

- 🧠 **Advanced Memory Management** - Vector-based semantic search with FAISS
- 🛠️ **Extensible Tool System** - Easily add new capabilities through modular tools
- 🤖 **Multiple Specialized Agents** - Dedicated agents for different tasks
- 🌐 **Web Interface** - Modern FastAPI-based web interface
- 🔒 **Ethics Framework** - Built-in ethical considerations
- 📊 **Vector Memory** - Efficient storage and retrieval of conversations
- 🔄 **Async Operation** - High-performance asynchronous execution
- ⚡ **MCP Architecture** - Model-Chosen Parameters for intelligent tool usage
- 🔍 **Semantic Search** - Advanced context retrieval using embeddings
- 📝 **Structured Data** - Pydantic-based validation throughout

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
    Faiss is crucial for the vector search capabilities.
    * For **GPU support** (recommended for performance if you have an NVIDIA GPU and CUDA installed):
        It was successfully installed on a Windows system using `conda` from the `conda-forge` channel. Use the following command:
        ```bash
        conda install -c conda-forge faiss-gpu
        ```
        If this command doesn't work for your specific OS/environment, consult the official Faiss installation documentation and the notes in `requirements.txt`.
    * For **CPU-only support**:
        If you don't have a compatible GPU or prefer the CPU version, you can install `faiss-cpu`. You can try:
        ```bash
        pip install faiss-cpu
        ```
        (Ensure `faiss-cpu` is uncommented or added to your `requirements.txt` if you prefer managing it there for pip-based environments).

4.  **Run the system**:
    ```bash
    python run.py
    ```

5.  Enter your goal at the WITS v2 prompt.

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/WITSAI.git
cd WITSAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up configuration:
```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your settings
```

4. Start the application:
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

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