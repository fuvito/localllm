from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import yaml

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core import load_knowledge, load_model, ask

SERVER_DIR = Path(__file__).parent
CONFIG_PATH = SERVER_DIR / "config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = _load_config()

    # Load model
    model_name = config["default_model"]
    cfg = dict(config["models"][model_name])
    app.state.cfg = cfg
    app.state.llm = load_model(cfg)

    # Load all restaurant knowledge bases
    restaurants: dict[str, dict[str, Any]] = {}
    for entry in config.get("restaurants", []):
        rid = entry["id"]
        knowledge_path = (SERVER_DIR / entry["knowledge"]).resolve()
        restaurants[rid] = {
            "id": rid,
            "name": entry["name"],
            "icon": entry["icon"],
            "description": entry["description"],
            "system_prompt": load_knowledge(str(knowledge_path)),
        }
        print(f"  Loaded knowledge: {rid} ({entry['name']})")

    app.state.restaurants = restaurants
    yield


app = FastAPI(title="Local LLM Restaurant Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────

class RestaurantInfo(BaseModel):
    id: str
    name: str
    icon: str
    description: str

class ChatRequest(BaseModel):
    restaurant_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/restaurants", response_model=list[RestaurantInfo])
async def list_restaurants(request: Request):
    return [
        RestaurantInfo(id=r["id"], name=r["name"], icon=r["icon"], description=r["description"])
        for r in request.app.state.restaurants.values()
    ]


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    restaurants = request.app.state.restaurants
    if req.restaurant_id not in restaurants:
        raise HTTPException(status_code=404, detail=f"Restaurant '{req.restaurant_id}' not found.")

    restaurant = restaurants[req.restaurant_id]
    history: list = []
    reply = ask(
        request.app.state.llm,
        restaurant["system_prompt"],
        req.message,
        history,
        request.app.state.cfg,
    )
    return ChatResponse(reply=reply)
