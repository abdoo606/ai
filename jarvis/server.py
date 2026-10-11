"""Web server: a chat page plus a small JSON API. Needs FastAPI (pip install -r jarvis/requirements.txt)."""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .agent import Assistant
from .config import load_settings

WEB_DIR = Path(__file__).resolve().parent / "web"


class ChatIn(BaseModel):
    message: str


class LearnIn(BaseModel):
    topic: str
    max_sources: int = 24
    depth: int = 2


def create_app(assistant: Assistant | None = None) -> FastAPI:
    app = FastAPI(title="Jarvis")
    settings = load_settings()
    bot = assistant or Assistant.from_settings(settings)
    lock = threading.Lock()  # one conversation at a time

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/status")
    def status():
        model = settings.hf_model if settings.llm_mode == "huggingface-local" else None
        models = None
        if settings.llm_mode == "local-ensemble":
            models = {"intent": settings.hf_intent_model, "english": settings.hf_english_model,
                      "arabic": settings.hf_arabic_model}
        return {"llm": settings.llm_mode, "model": model, "models": models,
                "topics": bot.tools.call("list_topics", {}),
                "shell_enabled": settings.allow_shell}

    @app.post("/api/chat")
    def chat(body: ChatIn):
        with lock:
            return {"reply": bot.ask(body.message)}

    @app.post("/api/reset")
    def reset():
        with lock:
            bot.reset()
        return {"ok": True}

    @app.post("/api/learn")
    def learn(body: LearnIn):
        if not body.topic.strip():
            raise HTTPException(status_code=400, detail="topic is empty")
        with lock:
            return {"result": bot.tools.call("learn_deep", {
                "topic": body.topic, "max_sources": body.max_sources, "depth": body.depth
            })}

    @app.get("/api/recall")
    def recall(q: str):
        with lock:
            return {"result": bot.tools.call("recall", {"query": q})}

    return app


app = None


def get_app() -> FastAPI:
    global app
    if app is None:
        app = create_app()
    return app
