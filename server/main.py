from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4
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
    # In-memory session store: session_id -> {restaurant_id, history, created_at}
    app.state.sessions: dict[str, dict] = {}
    yield


app = FastAPI(title="Local LLM Restaurant Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class RestaurantInfo(BaseModel):
    id: str
    name: str
    icon: str
    description: str

class SessionRequest(BaseModel):
    restaurant_id: str

class SessionResponse(BaseModel):
    session_id: str

class ChatRequest(BaseModel):
    session_id: str
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


@app.post("/sessions", response_model=SessionResponse)
async def create_session(req: SessionRequest, request: Request):
    if req.restaurant_id not in request.app.state.restaurants:
        raise HTTPException(status_code=404, detail=f"Restaurant '{req.restaurant_id}' not found.")
    session_id = str(uuid4())
    request.app.state.sessions[session_id] = {
        "restaurant_id": req.restaurant_id,
        "history": [],
        "created_at": datetime.utcnow().isoformat(),
    }
    return SessionResponse(session_id=session_id)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    sessions = request.app.state.sessions
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please start a new session.")

    session = sessions[req.session_id]
    restaurant = request.app.state.restaurants[session["restaurant_id"]]

    reply = ask(
        request.app.state.llm,
        restaurant["system_prompt"],
        req.message,
        session["history"],  # persistent history, grows each turn
        request.app.state.cfg,
    )
    return ChatResponse(reply=reply)
