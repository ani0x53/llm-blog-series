# LangGraph AI Agent with Ollama

A simple AI agent built with LangGraph and Ollama that can perform calculations, file operations, and natural conversation.

## Installation

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows - download from https://ollama.com/download
```

### 2. Pull the model and start Ollama

```bash
ollama pull llama3.2:1b
ollama serve
```

### 3. Setup Python environment

```bash
git clone <your-repo-url>
cd langgraph-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python langgraph_agent.py
```

## Usage Examples

```
You: What's 15 times 23?
You: List files in current directory  
You: Read config.txt
You: Hello, how are you?
```

Type 'quit' to exit.