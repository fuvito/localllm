# localllm

Local LLM inference project using llama-cpp-python and LangChain.

## Stack
- **Runtime**: Python 3.x
- **Inference**: `llama-cpp-python` (llama.cpp bindings)
- **Prompt tooling**: `langchain-core` (prompt templates, message types)
- **Model format**: GGUF (quantized, loaded locally)

## Project layout
```
localllm/
├── main.py        # entry point / demo script
├── CLAUDE.md      # this file
└── BACKLOG.md     # task list
```

## Running
```
python main.py
```
Place GGUF model files in the project root and update `model_path` in `main.py`.

## Key conventions
- Keep prompts in named constants at the top of the file
- Use `langchain_core` message/template abstractions instead of raw token strings
- `temperature` stays low (≤0.3) for deterministic, fact-constrained responses
- No external API calls — everything runs locally
