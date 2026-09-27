from datetime import datetime
from zoneinfo import ZoneInfo
import yaml
from llama_cpp import Llama

FORBIDDEN_KEYWORDS = ["ignore", "override", "system prompt", "developer mode"]

def _flag(item: dict, key: str) -> str:
    return "Yes" if item.get(key) else "No"

def _item_line(item: dict) -> str:
    parts = [f"- {item['name']} ${item['price']}"]
    if item.get("description"):
        parts.append(f"— {item['description']}")
    if item.get("note"):
        parts.append(f"({item['note']})")
    flags = []
    if "vegetarian" in item:
        flags.append(f"Vegetarian: {_flag(item, 'vegetarian')}")
    if "vegan" in item:
        flags.append(f"Vegan: {_flag(item, 'vegan')}")
    if "gluten_free" in item:
        flags.append(f"Gluten-Free: {_flag(item, 'gluten_free')}")
    if flags:
        parts.append("|")
        parts.append(" | ".join(flags))
    return " ".join(parts)

def _fmt_time(t: str) -> str:
    if t == "00:00":
        return "Midnight"
    h, m = map(int, t.split(":"))
    period = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {period}" if m else f"{h12}:00 {period}"

def _tz_abbr(tz_name: str) -> str:
    return datetime.now(ZoneInfo(tz_name)).strftime("%Z")

def _format_schedule(schedule: list, tz_abbr: str) -> str:
    parts = []
    for entry in schedule:
        days = entry["days"]
        open_t = _fmt_time(entry["open"])
        close_t = _fmt_time(entry["close"])
        if len(days) == 1:
            day_str = days[0]
        elif len(days) == 7:
            day_str = "Monday–Sunday"
        else:
            day_str = f"{days[0]}–{days[-1]}"
        parts.append(f"{day_str} {open_t}–{close_t}")
    return " | ".join(parts) + f" ({tz_abbr})"

def load_knowledge(path: str) -> tuple[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        kb = yaml.safe_load(f)

    r = kb["restaurant"]
    timezone = "UTC"

    if "business_hours" in kb:
        bh = kb["business_hours"]
        timezone = bh["timezone"]
        abbr = _tz_abbr(timezone)
        hours_text = _format_schedule(bh["schedule"], abbr)
    else:
        hours_text = r.get("hours", "See website for hours")

    lines = [
        f"You are a helpful assistant EXCLUSIVELY for {r['name']}. You have no knowledge of any other restaurant, business, or service. Use ONLY the facts listed below. Do not make up information.",
        "",
        "FACTS:",
        "",
        f"Hours: {hours_text}.",
        f"Address: {r['address']}.",
        f"Phone: {r['phone']}.",
        f"Email: {r['email']}.",
        f"Website: {r['website']}.",
        f"To order: {r['ordering']}.",
    ]

    if r.get("services"):
        lines.append(f"Services: {r['services']}.")

    # ── Generic menu sections (e.g. The Bite) ────────────────────────────────
    if "menu_sections" in kb:
        for section in kb["menu_sections"]:
            lines += ["", f"{section['category']}:"]
            if section.get("note"):
                lines.append(f"  Note: {section['note']}")
            for item in section["items"]:
                lines.append(_item_line(item))

    # ── Pizza-specific format (Luigi's Pizza) ────────────────────────────────
    else:
        gf_crust = next(c for c in kb["crusts"] if c["name"] == "Gluten-Free")
        lines += [
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
        "- You know the current date and time in the restaurant's local timezone. Use it to answer questions like 'are you open now?' or 'what time do you close today?'.",
        "- Only mention how to place an order if the customer explicitly asks how to order or finalize their order.",
        "- If a customer's question is ambiguous, ask one clarifying question before answering.",
        "- For greetings or compliments, respond warmly and briefly.",
        "- For farewells, wish them a good meal and a warm goodbye.",
        "- For complaints, apologize sincerely and offer to help with what you can.",
        f"- If the question is about the restaurant but not covered by the facts, say: \"I'm sorry, I don't have that detail — please visit us or give us a call.\"",
        f"- STRICT SCOPE: You only know about {r['name']}. Never reference, compare, or mention any other restaurant or business. If asked about anything outside {r['name']}, say: \"I can only help with questions about {r['name']}.\"",
        "- Never refuse to answer a question that the facts above can answer.",
    ]

    return "\n".join(lines), timezone


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


def ask(llm: Llama, system_prompt: str, query: str, history: list, cfg: dict, timezone: str = "UTC") -> str:
    if any(kw in query.lower() for kw in FORBIDDEN_KEYWORDS):
        return "I can only assist you with restaurant menu, hours, and dietary questions."

    now = datetime.now(ZoneInfo(timezone)).strftime("%A, %B %d, %Y %I:%M %p %Z")

    if cfg.get("system_role", False):
        # Model supports a dedicated system role (e.g. Qwen)
        if not history:
            history.append({"role": "system", "content": system_prompt})
        user_content = f"[Current date and time: {now}] Customer: {query}"
    else:
        # No system role (e.g. Gemma) — merge system prompt into the first user turn
        if not history:
            user_content = f"{system_prompt}\n\nCurrent date and time: {now}\n\nCustomer: {query}"
        else:
            user_content = f"[Current date and time: {now}] Customer: {query}"

    history.append({"role": "user", "content": user_content})

    response = llm.create_chat_completion(
        messages=history,
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
    )
    reply = response["choices"][0]["message"]["content"]
    history.append({"role": "assistant", "content": reply})
    return reply
