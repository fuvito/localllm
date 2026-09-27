# localllm

Run local LLM inference on your machine using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) and [LangChain Core](https://python.langchain.com/docs/concepts/).

No cloud API keys. No external calls. Everything runs locally.

---

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- A GGUF model file (see below)

## Setup

```bash
# Clone / enter the project directory
cd localllm

# Create virtual environment and install dependencies
uv sync
```

That's it. No `requirements.txt` needed — `uv` reads `pyproject.toml` and locks versions in `uv.lock`.

## Models

Download GGUF models and place them in the `models/` folder (gitignored — not committed):

```
models/
├── google_gemma-4-E4B-it-Q4_K_M.gguf
└── qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Register models in `config.yaml` and switch between them with `--model <name>`.

## Run

```bash
uv run python main.py
```

Use CLI flags to override defaults at runtime:

```bash
# Use a named model from config.yaml
uv run python main.py --model qwen

# Use any GGUF file directly
uv run python main.py --model ./some-model.gguf

# Tune parameters
uv run python main.py --temperature 0.7 --max-tokens 300 --threads 8

# Custom knowledge or config files
uv run python main.py --knowledge ./my_menu.yaml --config ./my_config.yaml
```

## Configuration

Models are registered in `config.yaml`:

```yaml
default_model: gemma4

models:
  gemma4:
    path: ./google_gemma-4-E4B-it-Q4_K_M.gguf
    n_ctx: 8192
    n_threads: 4
    temperature: 0.5
    max_tokens: 250
```

Add a new entry to switch models without touching code. CLI args override config values.

## Project structure

```
localllm/
├── main.py          # entry point
├── config.yaml      # model registry and defaults
├── knowledge.yaml   # menu, hours, dietary info
├── models/          # GGUF model files (gitignored)
├── pyproject.toml   # dependencies (uv)
├── uv.lock          # locked dependency versions
├── CLAUDE.md        # notes for AI assistants
├── BACKLOG.md       # planned features
└── README.md        # this file
```
