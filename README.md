# WITS-NEXUS v2 (WITS AI System)

**WITS-NEXUS v2** is an advanced, modular AI system designed for complex task orchestration and intelligent decision-making. It leverages a hierarchical agent architecture, sophisticated memory management with GPU-accelerated vector search, and an extensible toolset to tackle diverse goals.

**Current Version:** 4.0.0 (as of May 18, 2025)

## Core Philosophy

WITS-NEXUS v2 aims to provide a robust framework for:
-   **Intelligent Task Decomposition:** Breaking down complex user goals into manageable sub-tasks.
-   **Strategic Delegation:** Assigning tasks to specialized AI agents best suited for the job.
-   **Contextual Awareness:** Maintaining and utilizing rich conversational and operational history.
-   **Extensibility:** Allowing easy integration of new tools and specialized agents.
-   **Transparency:** Providing detailed logging and debugging capabilities for insight into agent operations.

## Key Features

-   🧠 **Hierarchical Agent System:**
    -   **WitsControlCenterAgent (WCCA):** The primary interface for user interaction. It interprets user input, clarifies intent, and delegates well-defined goals to the Orchestrator.
    -   **OrchestratorAgent:** Implements a ReAct-style (Reason-Act-Observe) loop. It plans and executes tasks by either calling tools directly or delegating to specialized agents.
    -   **Specialized Agents:** Focused agents (e.g., `EngineerAgent`, `ScribeAgent`, `AnalystAgent`, `ResearcherAgent`) that perform specific types of tasks using a curated set of tools.
-   💾 **Advanced Memory Management:**
    -   Persistent memory storage for conversation history, agent thoughts, actions, and observations.
    -   GPU-accelerated vector search using **FAISS** for efficient semantic retrieval of relevant memories.
    -   Structured memory segments using Pydantic models (`MemorySegment`, `MemorySegmentContent`).
-   🛠️ **Extensible Tool System:**
    -   A `ToolRegistry` allows dynamic registration and discovery of tools.
    -   Tools are Pydantic-based for clear argument schemas and validation.
    -   Examples: `CalculatorTool`, `DateTimeTool`, `FileTools` (`ReadFileTool`, `WriteFileTool`, `ListFilesTool`), `WebSearchTool`, `GitTool`, `ProjectFileReaderTool`.
-   🌐 **Async Operations:** Built with `asyncio` for high-performance, non-blocking I/O.
-   📝 **Structured Data & Validation:** Extensive use of Pydantic models throughout the system for data integrity and clear schemas (`StreamData`, `OrchestratorLLMResponse`, etc.).
-   🗣️ **LLM Agnostic (via LLMInterface):** Primarily designed for Ollama-hosted models, but `LLMInterface` can be adapted.
-   ⚙️ **Configuration Driven:** System behavior, model selection, and features are managed via `config.yaml`.
-   🔍 **Comprehensive Debugging:**
    -   Detailed logging (console and file) with configurable levels.
    -   Performance monitoring for critical operations.
    -   Component-specific debug flags (e.g., logging prompts/responses).
-   🖥️ **Dual Interface:**
    -   **CLI Mode:** For direct interaction and development.
    -   **Web App Mode (FastAPI):** Provides a web-based UI (details in `app/` directory).

## Project Structure

```
WITS-NEXUS_v2/
├── agents/              # Core agent implementations (WCCA, Orchestrator)
│   └── specialized/     # Specialized agents (Engineer, Scribe, etc.)
├── app/                 # FastAPI Web Application
│   ├── routes/
│   ├── services/
│   ├── static/
│   └── templates/
├── core/                # Core system components (config, llm, memory, schemas, tools)
├── data/                # Data storage (memory files, user files)
│   └── memory/
├── docs/                # Detailed documentation (e.g., FAISS setup)
├── logs/                # Application logs
├── tools/               # Implementations of various tools
├── config.yaml          # Main configuration file
├── run.py               # Main entry point for CLI and Web App
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── ... (other utility scripts and documentation)
```

## Installation and Setup

**1. Prerequisites:**

*   **Python 3.10.x:** This specific version is crucial for compatibility, especially with `faiss-gpu`.
*   **NVIDIA GPU with CUDA Toolkit:** Required for GPU-accelerated FAISS. Ensure your drivers and CUDA version are compatible. (Refer to `docs/FAISS-GPU-SETUP.md` for detailed guidance).
*   **Conda:** Recommended for managing the Python environment and dependencies.
*   **Ollama:** Install and run Ollama with the models specified in `config.yaml` (e.g., Llama3, CodeLlama). Ensure Ollama is accessible (check `ollama_url` in `config.yaml`).

**2. Environment Setup (Recommended using Conda):**

   a.  **Create and Activate Conda Environment:**
       The provided startup scripts (`start_wits.ps1` or `start.bat`) attempt to manage this. For manual setup:
       ```bash
       conda create -n wits_nexus_env python=3.10
       conda activate wits_nexus_env
       ```
       *(Note: Older documentation might refer to `faiss_gpu_env2`. `wits_nexus_env` is the current standard.)*

   b.  **Install FAISS-GPU:**
       ```bash
       conda install -c pytorch -c nvidia faiss-gpu=1.7.4 # Or a compatible version
       # Verify PyTorch version if needed, FAISS conda packages often bundle it.
       ```
       *Refer to `FAISS-GPU-INTEGRATION.md` or `docs/FAISS-GPU-SETUP.md` for troubleshooting.*

   c.  **Install Python Dependencies:**
       ```bash
       pip install -r requirements.txt
       ```

**3. Configuration:**

   a.  Copy `config.yaml.example` to `config.yaml`.
   b.  Review and update `config.yaml`:
       *   **Model Names:** Ensure `models.default`, `models.orchestrator`, `models.control_center`, etc., match the models you have pulled in Ollama.
       *   **Ollama URL:** Set `ollama_url` (e.g., `http://localhost:11434`).
       *   **Embedding Model:** Configure `memory_manager.vector_model` (e.g., `nomic-embed-text`). This model also needs to be available via Ollama or SentenceTransformers.
       *   **Paths:** Verify `memory_manager.memory_file_path` and other paths.
       *   **Internet Access & Git Integration:** Enable/disable as needed.

**4. Verify Environment:**
   Run the verification script (if available and updated for v2) or manually check:
   ```bash
   conda activate wits_nexus_env # Or your chosen environment name
   python
   >>> import faiss
   >>> print(faiss.get_num_gpus()) # Should be > 0
   >>> import torch
   >>> print(torch.cuda.is_available()) # Should be True
   >>> exit()
   ```

## Running WITS-NEXUS v2

**1. Using Startup Scripts (Recommended):**
   These scripts handle environment activation.
   *   PowerShell: `.\start.ps1`
   *   Command Prompt: `start.bat`

**2. Manual Execution:**
   ```bash
   conda activate wits_nexus_env # Or your environment name
   python run.py --mode cli  # For Command Line Interface
   # or
   python run.py --mode web  # For Web Application (starts Uvicorn server)
   ```
   If `--mode` is not specified, it defaults based on `config.yaml` (`web_interface.enabled`).

## How It Works (High-Level Flow)

1.  **User Input:** The user provides a goal or query (via CLI or Web UI).
2.  **WitsControlCenterAgent (WCCA):**
    *   Receives the raw input and conversation history.
    *   Uses an LLM to analyze the input and decide:
        *   If the goal is clear: Formulates a `goal_statement` for the Orchestrator.
        *   If ambiguous: Generates a `clarification_question` back to the user.
3.  **OrchestratorAgent:**
    *   Receives the `goal_statement` from WCCA.
    *   Enters a ReAct loop:
        *   **Reason:** Analyzes the goal, history, and previous steps to form a `thought_process` (thought, reasoning, plan).
        *   **Act:** Decides on an `chosen_action`:
            *   `tool_call`: Selects a tool (e.g., `WebSearchTool`, `EngineerAgent`) and its arguments.
            *   `final_answer`: If the goal is achieved.
        *   Executes the action (calls the tool/agent or provides the answer).
        *   **Observe:** Gets the result/observation from the action.
    *   Feeds the observation back into the ReAct loop.
4.  **Tool/Specialized Agent Execution:**
    *   If a tool is called, it executes its function.
    *   If a specialized agent is called, it runs its own logic (potentially its own ReAct loop or simpler LLM calls) using its dedicated tools.
5.  **Streaming Output:** Progress, thoughts, actions, and final answers are streamed back to the user.
6.  **Memory Persistence:** All significant events, thoughts, and data are stored as `MemorySegment` objects by the `MemoryManager`.

## Development

*   **Virtual Environment:** Use `venv` or Conda for managing dependencies.
*   **Dependencies:** `pip install -r requirements-dev.txt` (if a separate dev requirements file exists).
*   **Testing:** `pytest` (ensure tests are written/updated for v2 components).

## Contributing

Please refer to `CONTRIBUTING.md` for guidelines on contributing to the project.

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/YourFeature`).
3.  Commit your changes (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/YourFeature`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

-   Built upon the foundational concepts of agent-based systems and ReAct methodology.
-   Utilizes [Sentence Transformers](https://www.sbert.net/) for generating high-quality embeddings.
-   Leverages [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search.
-   Powered by [FastAPI](https://fastapi.tiangolo.com/) for the web interface.
-   Relies on [Pydantic](https://docs.pydantic.dev/) for robust data validation and settings management.
-   Interacts with Large Language Models via [Ollama](https://ollama.ai/).