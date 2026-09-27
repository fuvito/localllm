from contextlib import asynccontextmanager
from pathlib import Path
import yaml

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core import load_knowledge, load_model, ask

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.yaml"
KNOWLEDGE_PATH = ROOT / "knowledge.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = _load_config()
    model_name = config["default_model"]
    cfg = dict(config["models"][model_name])
    app.state.cfg = cfg
    app.state.system_prompt = load_knowledge(str(KNOWLEDGE_PATH))
    app.state.llm = load_model(cfg)
    yield


app = FastAPI(title="Luigi's Pizza Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    state = request.app.state
    history: list = []
    reply = ask(state.llm, state.system_prompt, req.message, history, state.cfg)
    return ChatResponse(reply=reply)
