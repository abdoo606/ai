"""Settings, read from environment variables.

Secrets are never stored in the repository. Set them in the environment:

    JARVIS_BACKEND        'offline' (default, no network, no key) or 'openai'
    JARVIS_LLM_API_KEY    API key, used only when JARVIS_BACKEND=openai (optional)
    JARVIS_LLM_BASE_URL   default https://api.openai.com/v1
    JARVIS_LLM_MODEL      default gpt-4o-mini
    JARVIS_HOME           where the knowledge base lives (default ./jarvis/data)
    JARVIS_WORKSPACE      folder the file tools may use (default ~/jarvis_workspace)
    JARVIS_ALLOW_SHELL    set to 1 to allow the run_command tool (off by default)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    backend: str
    home: Path
    workspace: Path
    allow_shell: bool

    @property
    def db_path(self) -> Path:
        return self.home / "knowledge.sqlite3"

    @property
    def llm_mode(self) -> str:
        if self.backend == "openai" and self.llm_api_key:
            return "openai-compatible"
        return "offline-rules"


def load_settings(env: dict | None = None) -> Settings:
    env = dict(os.environ if env is None else env)
    here = Path(__file__).resolve().parent
    return Settings(
        llm_api_key=env.get("JARVIS_LLM_API_KEY", ""),
        llm_base_url=env.get("JARVIS_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        llm_model=env.get("JARVIS_LLM_MODEL", "gpt-4o-mini"),
        backend=env.get("JARVIS_BACKEND", "offline").strip().lower(),
        home=Path(env.get("JARVIS_HOME", here / "data")),
        workspace=Path(env.get("JARVIS_WORKSPACE", Path.home() / "jarvis_workspace")),
        allow_shell=env.get("JARVIS_ALLOW_SHELL", "0") == "1",
    )
