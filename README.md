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

## Model

Download a GGUF model and place it in the project root. The default expected filename is:

```
qwen2.5-1.5b-instruct-q4_k_m.gguf
```

To use a different model, update `MODEL_PATH` at the top of `main.py`.

## Run

```bash
uv run python main.py
```

Or, if you've activated the virtual environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

python main.py
```

## Configuration

All tunable parameters are constants at the top of `main.py`:

| Constant      | Default                                    | Description                        |
|---------------|--------------------------------------------|------------------------------------|
| `MODEL_PATH`  | `./qwen2.5-1.5b-instruct-q4_k_m.gguf`     | Path to GGUF model file            |
| `N_CTX`       | `2048`                                     | Context window size (tokens)       |
| `N_THREADS`   | `4`                                        | CPU threads for inference          |
| `MAX_TOKENS`  | `150`                                      | Max tokens per response            |
| `TEMPERATURE` | `0.2`                                      | Randomness (lower = more focused)  |

## Project structure

```
localllm/
├── main.py          # entry point
├── pyproject.toml   # dependencies (uv)
├── uv.lock          # locked dependency versions
├── CLAUDE.md        # notes for AI assistants
├── BACKLOG.md       # planned features
└── README.md        # this file
```
