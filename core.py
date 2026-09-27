import yaml
from llama_cpp import Llama

FORBIDDEN_KEYWORDS = ["ignore", "override", "system prompt", "developer mode"]

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
        f"Address: {r['address']}.",
        f"Phone: {r['phone']}.",
        f"Email: {r['email']}.",
        f"Website: {r['website']}.",
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

    lines += ["", "Drinks (non-alcoholic):"]
    for d in kb["drinks"]:
        vegan = "Yes" if d["vegan"] else "No"
        gf = "Yes" if d["gluten_free"] else "No"
        lines.append(f"- {d['name']} ${d['price']} | Vegan: {vegan} | Gluten-Free: {gf}")

    lines += [
        "",
        "RULES:",
        "- Always reply in the same language the customer used.",
        "- Keep responses to 1-2 sentences maximum.",
        "- Directly answer the customer's specific question first, then provide the relevant fact if helpful.",
        "- You may perform simple arithmetic (addition, multiplication) using the prices in the facts.",
        "- SAFETY: If a customer mentions an allergy, celiac disease, or vegan diet alongside gluten-free, always warn that the Gluten-Free crust contains eggs and is NOT vegan — even if they did not ask.",
        "- Only mention how to place an order if the customer explicitly asks how to order or finalize their order.",
        "- If a customer's question is ambiguous, ask one clarifying question before answering.",
        "- For greetings or compliments, respond warmly and briefly.",
        "- For farewells, wish them a good meal and a warm goodbye.",
        "- For complaints, apologize sincerely and offer to help with what you can.",
        "- If the question is about the restaurant but not covered by the facts, say: \"I'm sorry, I don't have that detail — please visit us or give us a call during opening hours.\"",
        "- If the question has nothing to do with the restaurant, politely redirect: \"I can only help with questions about Luigi's Pizza.\"",
        "- Never refuse to answer a question that the facts above can answer.",
    ]

    return "\n".join(lines)


def load_model(cfg: dict) -> Llama:
    print(f"Loading model: {cfg['path']}")
    model = Llama(
        model_path=cfg["path"],
        n_ctx=cfg["n_ctx"],
        n_threads=cfg["n_threads"],
        verbose=False,
    )
    print("Model ready.")
    return model


def ask(llm: Llama, system_prompt: str, query: str, history: list, cfg: dict) -> str:
    if any(kw in query.lower() for kw in FORBIDDEN_KEYWORDS):
        return "I can only assist you with restaurant menu, hours, and dietary questions."

    # Gemma has no system role — merge system prompt into the first user turn only
    user_content = f"{system_prompt}\n\nCustomer: {query}" if not history else f"Customer: {query}"
    history.append({"role": "user", "content": user_content})

    response = llm.create_chat_completion(
        messages=history,
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
    )
    reply = response["choices"][0]["message"]["content"]
    history.append({"role": "assistant", "content": reply})
    return reply
