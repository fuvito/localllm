import yaml
from llama_cpp import Llama

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_PATH = "./google_gemma-4-E4B-it-Q4_K_M.gguf"
KNOWLEDGE_PATH = "./knowledge.yaml"
N_CTX = 8192
N_THREADS = 4
MAX_TOKENS = 250
TEMPERATURE = 0.5

FORBIDDEN_KEYWORDS = ["ignore", "override", "system prompt", "developer mode"]

# ── Knowledge loader ──────────────────────────────────────────────────────────
def load_knowledge(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        kb = yaml.safe_load(f)

    r = kb["restaurant"]
    gf_crust = next(c for c in kb["crusts"] if c["name"] == "Gluten-Free")

    lines = [
        f"You are a helpful assistant for {r['name']}. Use ONLY the facts below to answer. Do not make up information.",
        "",
        "FACTS:",
        "",
        f"Hours: {r['hours']}.",
        f"To order: {r['ordering']}.",
        "",
        "Crust options:",
        "- Standard crust (default)",
        f"- Gluten-Free crust: +${gf_crust['extra_cost']} — {gf_crust['note']}",
        "",
        "Pizzas (all can be made Gluten-Free for +$3):",
    ]

    for p in kb["pizzas"]:
        veg = "Yes" if p["vegetarian"] else "No"
        vegan = ("No" + (f" ({p['vegan_note']})" if p.get("vegan_note") else "")) if not p["vegan"] else "Yes"
        lines.append(f"- {p['name']} ${p['price']} — {p['description']} | Vegetarian: {veg} | Vegan: {vegan}")

    lines += ["", "Sides:"]
    for s in kb["sides"]:
        veg = "Yes" if s["vegetarian"] else "No"
        vegan = ("No" + (f" ({s['vegan_note']})" if s.get("vegan_note") else "")) if not s["vegan"] else "Yes"
        gf = "Yes" if s["gluten_free"] else "No"
        lines.append(f"- {s['name']} ${s['price']} | Vegetarian: {veg} | Vegan: {vegan} | Gluten-Free: {gf}")

    lines += [
        "",
        "RULES:",
        "- Always reply in the same language the customer used.",
        "- Keep responses to 1-2 sentences maximum.",
        "- Directly answer the customer's specific question first, then provide the relevant fact if helpful.",
        "- You may perform simple arithmetic (addition, multiplication) using the prices in the facts.",
        "- SAFETY: If a customer mentions an allergy, celiac disease, or vegan diet alongside gluten-free, always warn that the Gluten-Free crust contains eggs and is NOT vegan — even if they did not ask.",
        "- When a customer seems ready to order, remind them to visit in person or call during opening hours.",
        "- If a customer's question is ambiguous, ask one clarifying question before answering.",
        "- For greetings or compliments, respond warmly and briefly.",
        "- For farewells, wish them a good meal and a warm goodbye.",
        "- For complaints, apologize sincerely and offer to help with what you can.",
        "- If the question is about the restaurant but not covered by the facts, say: \"I'm sorry, I don't have that detail — please visit us or give us a call during opening hours.\"",
        "- If the question has nothing to do with the restaurant, politely redirect: \"I can only help with questions about Luigi's Pizza.\"",
        "- Never refuse to answer a question that the facts above can answer.",
    ]

    return "\n".join(lines)

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
def ask(llm: Llama, system_prompt: str, query: str) -> str:
    if any(kw in query.lower() for kw in FORBIDDEN_KEYWORDS):
        return "I can only assist you with restaurant menu, hours, and dietary questions."

    # Gemma has no system role — merge system prompt into the user turn
    dict_messages = [
        {"role": "user", "content": f"{system_prompt}\n\nCustomer: {query}"},
    ]

    response = llm.create_chat_completion(
        messages=dict_messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return response["choices"][0]["message"]["content"]

# ── Chat loop ─────────────────────────────────────────────────────────────────
def chat_loop(llm: Llama, system_prompt: str) -> None:
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
        print(f"Assistant: {ask(llm, system_prompt, query)}\n")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    system_prompt = load_knowledge(KNOWLEDGE_PATH)
    llm = load_model()
    chat_loop(llm, system_prompt)
