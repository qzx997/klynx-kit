# Klynx


[![PyPI version](https://img.shields.io/pypi/v/klynx.svg)](https://pypi.org/project/klynx/)
[![Python versions](https://img.shields.io/pypi/pyversions/klynx.svg)](https://pypi.org/project/klynx/)
[![License](https://img.shields.io/github/license/qzx997/klynx-kit.svg)](https://github.com/qzx997/klynx-kit/blob/main/LICENSE)

> [!WARNING] 
> **当前为测试版本 (Current version is in Beta testing)**
> Klynx 仍在活跃开发中，部分高级功能及 API 不够完善或可能在未来版本发生变动，请谨慎用于生产环境。

**Klynx** is an advanced, highly customizable autonomous agent framework built on top of [LangGraph](https://langchain-ai.github.io/langgraph/) and [LiteLLM](https://github.com/BerriAI/litellm). It is designed to seamless integrate with top-tier LLM providers globally (OpenAI, Anthropic, Google, DeepSeek, Zhipu, Moonshot, etc.) while offering a robust tool-calling infrastructure and built-in interactive terminal interfaces.

## 🚀 Features
- **Universal LLM Routing:** Write code once, use any model. Powered by `litellm`.
- **Advanced Agent Architectures:** Built with `langgraph` core state machines for recursive reasoning, memory, error reflection, and persistent context loops.
- **Robust Tools Environment:** Ships with native tools (e.g. Browser automation with Playwright, OS terminal integration, local filesystem querying).
- **Interactive TUI:** Comes with a built-in rich Text User Interface (`klynx/tui_app.py`) for streaming console interactions.
- **Proxy-Resistant:** `LiteLLM` adapters are configured internally to gracefully map local network environments without throwing blocking SSL errors.

## 📦 Installation

Install Klynx directly from PyPI:

```bash
pip install klynx
```

To update to the latest version, run:
```bash
pip install --upgrade klynx
```

*(Note: If using browser features, remember to run `playwright install chromium` once).*

## ⚡ Quick Start

With Klynx, creating a multi-modal agent and interacting with it requires just a few lines of code.

### 1. Initialize your Model

Klynx expects explicit passing of your API keys or auto-loading from `.env` files. We fully support major models.

```python
import os
from klynx import setup_model, set_tavily_api

# (Optional) Enable web search capabilities globally
set_tavily_api(os.getenv("TAVILY_API_KEY", ""))

# Setup your agent's brain
api_key = os.getenv("DEEPSEEK_API_KEY")
model = setup_model("deepseek", "deepseek-reasoner", api_key)
```

### 2. Create the Agent

Attach your model to a `KlynxAgent` state machine, optionally granting it all system tools.

```python
from klynx import create_agent

# Initialize agent in current directory
agent = create_agent(
    working_dir=os.getcwd(),
    model=model,
    max_iterations=15 
)

# Grant agent access to all default tools (Browser, Files, Terminal, Web Search)
agent.add_tools("all")
```

### 3. Run a Task

There are two primary ways to run instructions: a direct API call or an interactive terminal stream.

#### Option A: Single Invocation (API Level)
Useful when integrating Klynx into your own backend or processing pipeline. It returns the final agent state and summary.

```python
task = "Search the web for the latest Python version changes and write a summary to a file named 'python_updates.md'."
# `invoke` processes the execution loop and streaming yields events
for event in agent.invoke(task):
    event_type = event.get("type", "")
    content = event.get("content", "")
    
    if event_type == "done":
        # Final result state
        if event.get("task_completed"):
            print("Success! Agent Summary:", event.get("summary_content"))
    else:
        print(f"{content}") 
```

#### Option B: Real-time Terminal Streaming
If you want to watch the agent think, execute tools, and log out its process in a beautiful format directly in the console, use `run_terminal_agent_stream`.

```python
from klynx import run_terminal_agent_stream

# Interactive terminal chat loop
while True:
    task = input("\n[User]> ")
    if task.strip().lower() in ["exit", "q", "quit"]:
        break
    
    # This will stream reasoning tags, tool calls, and results to stdout
    result = run_terminal_agent_stream(agent, task, thread_id="my_cli_session")
```

### 4. Built-in TUI App

Klynx includes a powerful Text User Interface for managing and interacting with your agent directly in your terminal without writing CLI scripts.

Launch the interactive app directly from your terminal:
```bash
python -m klynx.tui_app
```

**Programmable TUI (API Config Injection)**

You can also launch the TUI directly within your python scripts. This allows you to dynamically configure your LLM providers and API keys without relying on environment variables.

```python
from klynx import KlynxTUIApp

# Inject your configuration dynamically
app = KlynxTUIApp(
    model_provider="deepseek",          # e.g., openai, anthropic
    model_name="deepseek-reasoner",     # e.g., o3-mini, claude-3-5-sonnet
    api_key="your_api_key_here",      
    tavily_api_key="your_tavily_key" 
)

app.run()
```

## 🤝 Contributing

We welcome contributions to Klynx! If you'd like to contribute, please follow these steps:

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally: `git clone https://github.com/your-username/klynx-kit.git`
3. **Install dependencies** for development: `pip install -e .` (It's recommended to use a virtual environment).
4. **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name`
5. **Make your changes** and test them thoroughly.
6. **Commit your changes** with descriptive commit messages.
7. **Push your branch** to your fork: `git push origin feature/your-feature-name`
8. **Submit a Pull Request** against the `main` branch of the original Klynx repository.

If you are proposing a significant change or architecture adjustment, please open an Issue first to discuss it with the maintainers.

## 📜 License
This software is licensed under the [MIT License](LICENSE).


