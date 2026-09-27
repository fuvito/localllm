from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from llama_cpp import Llama

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_PATH = "./google_gemma-4-E4B-it-Q4_K_M.gguf"
N_CTX = 8192
N_THREADS = 4
MAX_TOKENS = 150
TEMPERATURE = 0.5

# ── Knowledge base / system prompt ───────────────────────────────────────────
SYSTEM_PROMPT = """
You are a helpful assistant for Luigi's Pizza. Use ONLY the facts below to answer. Do not make up information.

FACTS:
- We are open Monday to Sunday, 11:00 AM to 10:00 PM.
- Standard pizzas start at $12.
- We offer a Gluten-Free crust for an extra $3.
- The Gluten-Free crust contains eggs. It is NOT vegan.

RULES:
- Always answer directly using the facts above.
- If the question cannot be answered from the facts, say: "I don't have that information."
- Never refuse to answer a question that the facts above can answer.
"""

FORBIDDEN_KEYWORDS = ["ignore", "override", "system prompt", "developer mode"]

# ── Prompt template ───────────────────────────────────────────────────────────
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content="{customer_query}"),
])

# ── Model loader ──────────────────────────────────────────────────────────────
def load_model() -> Llama:
    print("Loading model...")
    model = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        verbose=False,
    )
    print("Model ready.")
    return model

# ── Inference ─────────────────────────────────────────────────────────────────
def ask(llm: Llama, query: str) -> str:
    if any(kw in query.lower() for kw in FORBIDDEN_KEYWORDS):
        return "I can only assist you with restaurant menu, hours, and dietary questions."

    messages = prompt_template.format_messages(customer_query=query)
    # Gemma has no system role — merge system prompt into the user turn
    system_content = next(m.content for m in messages if m.type == "system")
    user_content = next(m.content for m in messages if m.type == "human")
    dict_messages = [
        {"role": "user", "content": f"{system_content}\n\n{user_content}"},
    ]

    response = llm.create_chat_completion(
        messages=dict_messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return response["choices"][0]["message"]["content"]

# ── Chat loop ─────────────────────────────────────────────────────────────────
def chat_loop(llm: Llama) -> None:
    print("\nLuigi's Pizza Assistant — type 'quit' or press Ctrl+C to exit.\n")
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
        print(f"Assistant: {ask(llm, query)}\n")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    llm = load_model()
    chat_loop(llm)
