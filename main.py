import argparse
import yaml
from core import load_knowledge, load_model, ask

CONFIG_PATH = "./server/config.yaml"
KNOWLEDGE_PATH = "./data/knowledge_luigis.yaml"

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_model_cfg(config: dict, args: argparse.Namespace) -> dict:
    """Merge config model entry with CLI overrides. CLI wins."""
    name = args.model or config["default_model"]
    models = config.get("models", {})

    if name in models:
        cfg = dict(models[name])
    else:
        cfg = {"path": name, "n_ctx": 4096, "n_threads": 4, "temperature": 0.5, "max_tokens": 250}

    if args.n_ctx is not None:
        cfg["n_ctx"] = args.n_ctx
    if args.threads is not None:
        cfg["n_threads"] = args.threads
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if args.max_tokens is not None:
        cfg["max_tokens"] = args.max_tokens
    return cfg

def chat_loop(llm, system_prompt: str, cfg: dict, timezone: str = "UTC") -> None:
    print()
    history: list = []
    welcome = ask(llm, system_prompt, "Greet the customer with a warm welcome message.", history, cfg, timezone=timezone)
    print(f"Assistant: {welcome}\n")
    print("(type 'quit' or press Ctrl+C to exit)\n")
    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        print(f"Assistant: {ask(llm, system_prompt, query, history, cfg, timezone=timezone)}\n")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Luigi's Pizza local LLM assistant")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name from config.yaml or path to a .gguf file")
    parser.add_argument("--n-ctx", type=int, default=None, help="Context window size")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens per response")
    parser.add_argument("--knowledge", type=str, default=KNOWLEDGE_PATH, help="Path to knowledge.yaml")
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="Path to config.yaml")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    cfg = resolve_model_cfg(config, args)
    system_prompt, timezone = load_knowledge(args.knowledge)
    llm = load_model(cfg)
    chat_loop(llm, system_prompt, cfg, timezone=timezone)
