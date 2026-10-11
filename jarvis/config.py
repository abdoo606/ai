"""Jarvis settings for built-in and fully local model brains.

There are no hosted-model/API settings.  The optional ensemble uses three
local model folders: a fast multilingual intent model, an English model, and
an Arabic model. Models load lazily and can only download when explicitly
allowed during initial setup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    backend: str
    home: Path
    workspace: Path
    allow_shell: bool
    hf_model: str = "jarvis/models/brain"
    hf_english_model: str = "jarvis/models/reasoning"
    hf_arabic_model: str = "jarvis/models/reasoning"
    hf_intent_model: str = "jarvis/models/intent"
    hf_reasoning_model: str = "jarvis/models/reasoning"
    hf_critic_model: str = "jarvis/models/reasoning"
    hf_allow_download: bool = False
    hf_device: str = "auto"
    hf_max_new_tokens: int = 1024
    hf_intent_max_new_tokens: int = 256
    reasoning_rounds: int = 1

    @property
    def db_path(self) -> Path:
        return self.home / "knowledge.sqlite3"

    @property
    def llm_mode(self) -> str:
        if self.backend in {"powerful", "deliberate", "reasoning"}:
            return "local-deliberative"
        if self.backend in {"ensemble", "local-ensemble", "jarvis"}:
            return "local-ensemble"
        if self.backend in {"hf", "huggingface", "transformers", "local-model"}:
            return "huggingface-local"
        return "offline-rules"


def _integer(env: dict, name: str, default: int, minimum: int = 32) -> int:
    try:
        return max(minimum, int(env.get(name, str(default))))
    except ValueError:
        return default


def load_settings(env: dict | None = None) -> Settings:
    env = dict(os.environ if env is None else env)
    here = Path(__file__).resolve().parent
    return Settings(
        backend=env.get("JARVIS_BACKEND", "offline").strip().lower(),
        home=Path(env.get("JARVIS_HOME", here / "data")).expanduser(),
        workspace=Path(env.get("JARVIS_WORKSPACE", Path.home() / "jarvis_workspace")).expanduser(),
        allow_shell=_truthy(env.get("JARVIS_ALLOW_SHELL", "0")),
        hf_model=env.get("JARVIS_HF_MODEL", "jarvis/models/brain").strip(),
        hf_english_model=env.get("JARVIS_HF_ENGLISH_MODEL", "jarvis/models/reasoning").strip(),
        hf_arabic_model=env.get("JARVIS_HF_ARABIC_MODEL", "jarvis/models/reasoning").strip(),
        hf_intent_model=env.get("JARVIS_HF_INTENT_MODEL", "jarvis/models/intent").strip(),
        hf_reasoning_model=env.get("JARVIS_HF_REASONING_MODEL", "jarvis/models/reasoning").strip(),
        hf_critic_model=env.get("JARVIS_HF_CRITIC_MODEL", "jarvis/models/reasoning").strip(),
        hf_allow_download=_truthy(env.get("JARVIS_HF_ALLOW_DOWNLOAD", "0")),
        hf_device=env.get("JARVIS_HF_DEVICE", "auto").strip(),
        hf_max_new_tokens=_integer(env, "JARVIS_HF_MAX_NEW_TOKENS", 1024),
        hf_intent_max_new_tokens=_integer(env, "JARVIS_HF_INTENT_MAX_NEW_TOKENS", 256),
        reasoning_rounds=min(_integer(env, "JARVIS_REASONING_ROUNDS", 1, minimum=1), 3),
    )
