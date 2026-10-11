"""Jarvis settings.

Jarvis has no hosted-model/API backend. Its brain is either the built-in
zero-dependency offline engine or an open model running on this machine through
Hugging Face Transformers. Secrets and API keys are intentionally unsupported.
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
    hf_allow_download: bool = False
    hf_device: str = "auto"
    hf_max_new_tokens: int = 512

    @property
    def db_path(self) -> Path:
        return self.home / "knowledge.sqlite3"

    @property
    def llm_mode(self) -> str:
        if self.backend in {"hf", "huggingface", "transformers", "local-model"}:
            return "huggingface-local"
        return "offline-rules"


def load_settings(env: dict | None = None) -> Settings:
    env = dict(os.environ if env is None else env)
    here = Path(__file__).resolve().parent
    try:
        max_tokens = max(32, int(env.get("JARVIS_HF_MAX_NEW_TOKENS", "512")))
    except ValueError:
        max_tokens = 512
    return Settings(
        backend=env.get("JARVIS_BACKEND", "offline").strip().lower(),
        home=Path(env.get("JARVIS_HOME", here / "data")).expanduser(),
        workspace=Path(env.get("JARVIS_WORKSPACE", Path.home() / "jarvis_workspace")).expanduser(),
        allow_shell=_truthy(env.get("JARVIS_ALLOW_SHELL", "0")),
        hf_model=env.get("JARVIS_HF_MODEL", "jarvis/models/brain").strip(),
        hf_allow_download=_truthy(env.get("JARVIS_HF_ALLOW_DOWNLOAD", "0")),
        hf_device=env.get("JARVIS_HF_DEVICE", "auto").strip(),
        hf_max_new_tokens=max_tokens,
    )
