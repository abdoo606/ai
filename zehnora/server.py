"""Hardened local web server and JSON endpoints for Zehnora."""

from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .agent import Assistant
from .config import Settings, load_settings

WEB_DIR = Path(__file__).resolve().parent / "web"


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)


class LearnIn(BaseModel):
    topic: str = Field(min_length=1, max_length=300)
    max_sources: int = Field(default=24, ge=1, le=100)
    depth: int = Field(default=2, ge=1, le=4)


def create_app(assistant: Assistant | None = None, settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="Zehnora", version=__version__, docs_url=None, redoc_url=None)
    settings = settings or load_settings()
    bot = assistant or Assistant.from_settings(settings)
    conversation_lock = threading.Lock()
    rate_lock = threading.Lock()
    requests_by_client: dict[str, deque[float]] = defaultdict(deque)
    failed_auth_by_client: dict[str, deque[float]] = defaultdict(deque)

    @app.middleware("http")
    async def security(request: Request, call_next):
        path = request.url.path
        identity = request.client.host if request.client else "unknown"
        if path.startswith("/api/") and path != "/api/health" and settings.access_token:
            supplied = request.headers.get("x-zehnora-token", "")
            authorization = request.headers.get("authorization", "")
            if authorization.lower().startswith("bearer "):
                supplied = authorization[7:].strip()
            now = time.monotonic()
            with rate_lock:
                failures = failed_auth_by_client[identity]
                while failures and failures[0] <= now - 60:
                    failures.popleft()
                if len(failures) >= 10:
                    return JSONResponse({"detail": "too many authentication failures"}, status_code=429,
                                        headers={"Retry-After": "60"})
                authenticated = hmac.compare_digest(supplied, settings.access_token)
                if not authenticated:
                    failures.append(now)
            if not authenticated:
                return JSONResponse({"detail": "authentication required"}, status_code=401)

        if path in {"/api/chat", "/api/learn"}:
            now = time.monotonic()
            with rate_lock:
                history = requests_by_client[identity]
                while history and history[0] <= now - 60:
                    history.popleft()
                if len(history) >= settings.rate_limit_per_minute:
                    return JSONResponse({"detail": "rate limit exceeded"}, status_code=429,
                                        headers={"Retry-After": "60"})
                history.append(now)

        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), geolocation=(), microphone=(self)"
        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": app.version}

    @app.get("/api/status")
    def status():
        model = settings.hf_model if settings.llm_mode == "huggingface-local" else None
        models = None
        if settings.llm_mode == "local-ensemble":
            models = {"intent": settings.hf_intent_model, "english": settings.hf_english_model,
                      "arabic": settings.hf_arabic_model}
        elif settings.llm_mode == "local-deliberative":
            models = {"intent": settings.hf_intent_model, "reasoner": settings.hf_reasoning_model,
                      "critic": settings.hf_critic_model, "english": settings.hf_english_model,
                      "arabic": settings.hf_arabic_model}
        return {"llm": settings.llm_mode, "model": model, "models": models,
                "topics": bot.tools.call("list_topics", {}),
                "shell_enabled": "run_command" in bot.tools.tools,
                "authentication_enabled": bool(settings.access_token)}

    @app.post("/api/chat")
    def chat(body: ChatIn):
        with conversation_lock:
            return {"reply": bot.ask(body.message)}

    @app.post("/api/reset")
    def reset():
        with conversation_lock:
            bot.reset()
        return {"ok": True}

    @app.post("/api/learn")
    def learn(body: LearnIn):
        with conversation_lock:
            return {"result": bot.tools.call("learn_deep", {
                "topic": body.topic, "max_sources": body.max_sources, "depth": body.depth,
            })}

    @app.get("/api/recall")
    def recall(q: str = Query(min_length=1, max_length=2_000)):
        with conversation_lock:
            return {"result": bot.tools.call("recall", {"query": q})}

    return app


app = None


def get_app() -> FastAPI:
    global app
    if app is None:
        app = create_app()
    return app
