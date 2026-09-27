from llama_cpp import Llama

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_PATH = "./google_gemma-4-E4B-it-Q4_K_M.gguf"
N_CTX = 8192
N_THREADS = 4
MAX_TOKENS = 150
TEMPERATURE = 0.5

# ── Knowledge base / system prompt ───────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful assistant for Luigi's Pizza. Use ONLY the facts below to answer. Do not make up information.

FACTS:
- We are open Monday to Sunday, 11:00 AM to 10:00 PM.
- Standard pizzas start at $12.
- We offer a Gluten-Free crust for an extra $3.
- The Gluten-Free crust contains eggs. It is NOT vegan.
- To place an order, customers should visit us in person or call us during opening hours.

RULES:
- Always reply in the same language the customer used.
- Keep responses to 1-2 sentences maximum.
- Directly answer the customer's specific question first, then provide the relevant fact if helpful.
- You may perform simple arithmetic (addition, multiplication) using the prices in the facts.
- SAFETY: If a customer mentions an allergy, celiac disease, or vegan diet alongside gluten-free, always warn that the Gluten-Free crust contains eggs and is NOT vegan — even if they did not ask.
- When a customer seems ready to order, remind them to visit in person or call during opening hours.
- If a customer's question is ambiguous, ask one clarifying question before answering.
- For greetings or compliments, respond warmly and briefly.
- For farewells, wish them a good meal and a warm goodbye.
- For complaints, apologize sincerely and offer to help with what you can.
- If the question is about the restaurant but not covered by the facts (e.g. specific pizza types, toppings, delivery), say something like "I'm sorry, I don't have that detail — please visit us or give us a call during opening hours."
- If the question has nothing to do with the restaurant, politely redirect: "I can only help with questions about Luigi's Pizza."
- Never refuse to answer a question that the facts above can answer."""

FORBIDDEN_KEYWORDS = ["ignore", "override", "system prompt", "developer mode"]

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

    # Gemma has no system role — merge system prompt into the user turn
    dict_messages = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nCustomer: {query}"},
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
