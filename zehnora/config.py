"""Zehnora settings for built-in and fully local model brains.

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
    access_token: str = ""
    allowed_commands: tuple[str, ...] = ()
    allow_private_web: bool = False
    rate_limit_per_minute: int = 60
    hf_model: str = "zehnora/models/brain"
    hf_english_model: str = "zehnora/models/reasoning"
    hf_arabic_model: str = "zehnora/models/reasoning"
    hf_intent_model: str = "zehnora/models/intent"
    hf_reasoning_model: str = "zehnora/models/reasoning"
    hf_critic_model: str = "zehnora/models/reasoning"
    hf_allow_download: bool = False
    hf_revision: str = ""
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
        if self.backend in {"ensemble", "local-ensemble", "zehnora"}:
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
        backend=env.get("ZEHNORA_BACKEND", "offline").strip().lower(),
        home=Path(env.get("ZEHNORA_HOME", here / "data")).expanduser(),
        workspace=Path(env.get("ZEHNORA_WORKSPACE", Path.home() / "zehnora_workspace")).expanduser(),
        allow_shell=_truthy(env.get("ZEHNORA_ALLOW_SHELL", "0")),
        access_token=env.get("ZEHNORA_ACCESS_TOKEN", "").strip(),
        allowed_commands=tuple(command.strip() for command in
                               env.get("ZEHNORA_ALLOWED_COMMANDS", "").split(",") if command.strip()),
        allow_private_web=_truthy(env.get("ZEHNORA_ALLOW_PRIVATE_WEB", "0")),
        rate_limit_per_minute=min(_integer(env, "ZEHNORA_RATE_LIMIT_PER_MINUTE", 60, minimum=1), 600),
        hf_model=env.get("ZEHNORA_HF_MODEL", "zehnora/models/brain").strip(),
        hf_english_model=env.get("ZEHNORA_HF_ENGLISH_MODEL", "zehnora/models/reasoning").strip(),
        hf_arabic_model=env.get("ZEHNORA_HF_ARABIC_MODEL", "zehnora/models/reasoning").strip(),
        hf_intent_model=env.get("ZEHNORA_HF_INTENT_MODEL", "zehnora/models/intent").strip(),
        hf_reasoning_model=env.get("ZEHNORA_HF_REASONING_MODEL", "zehnora/models/reasoning").strip(),
        hf_critic_model=env.get("ZEHNORA_HF_CRITIC_MODEL", "zehnora/models/reasoning").strip(),
        hf_allow_download=_truthy(env.get("ZEHNORA_HF_ALLOW_DOWNLOAD", "0")),
        hf_revision=env.get("ZEHNORA_HF_REVISION", "").strip(),
        hf_device=env.get("ZEHNORA_HF_DEVICE", "auto").strip(),
        hf_max_new_tokens=_integer(env, "ZEHNORA_HF_MAX_NEW_TOKENS", 1024),
        hf_intent_max_new_tokens=_integer(env, "ZEHNORA_HF_INTENT_MAX_NEW_TOKENS", 256),
        reasoning_rounds=min(_integer(env, "ZEHNORA_REASONING_ROUNDS", 1, minimum=1), 3),
    )
