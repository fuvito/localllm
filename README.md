# localllm

Run local LLM inference on your machine using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python).

No cloud API keys. No external calls. Everything runs locally.

---

## Requirements

- Python 3.10+
- Node.js 18+ (for the web client)
- [uv](https://docs.astral.sh/uv/) package manager
- A GGUF model file (see below)

## Setup

```bash
# Install Python dependencies
uv sync

# Install client dependencies
cd client && npm install && cd ..
```

## Models

Download GGUF models and place them in the `models/` folder (gitignored — not committed):

```
models/
├── google_gemma-4-E4B-it-Q4_K_M.gguf
└── qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Register models in `config.yaml` and switch between them with `--model <name>`.

## Run

### CLI (terminal chat)

```bash
uv run python main.py

# Options
uv run python main.py --model qwen
uv run python main.py --temperature 0.7 --max-tokens 300 --threads 8
uv run python main.py --model ./path/to/model.gguf
```

### Web (FastAPI server + React client)

In one terminal:
```bash
uv run uvicorn server.main:app --reload
```

In another terminal:
```bash
cd client && npm run dev
```

Then open [http://localhost:5173](http://localhost:5173).

## Configuration

Models are registered in `config.yaml`:

```yaml
default_model: gemma4

models:
  gemma4:
    path: ./models/google_gemma-4-E4B-it-Q4_K_M.gguf
    n_ctx: 8192
    n_threads: 4
    temperature: 0.5
    max_tokens: 250
```

Add a new entry to switch models without touching code. CLI args override config values.

## Project structure

```
localllm/
├── core.py          # shared inference logic
├── main.py          # CLI entry point
├── server/
│   └── main.py      # FastAPI server
├── client/          # React + Vite web client
├── config.yaml      # model registry and defaults
├── knowledge.yaml   # menu, hours, dietary info
├── models/          # GGUF model files (gitignored)
├── pyproject.toml   # Python dependencies (uv)
├── uv.lock          # locked dependency versions
├── CLAUDE.md        # notes for AI assistants
├── BACKLOG.md       # planned features
└── README.md        # this file
```
