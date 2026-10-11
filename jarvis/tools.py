"""Tools the assistant can call.

Each tool has a JSON schema (so an LLM can call it) and a Python function.
Results are plain strings, capped in length, so one tool cannot flood the
model's context. File tools stay inside the workspace. Shell commands are
off unless JARVIS_ALLOW_SHELL=1.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import web
from .config import Settings
from .knowledge import KnowledgeBase

MAX_RESULT_CHARS = 4000


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., str]

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


@dataclass
class ToolRegistry:
    tools: Dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self.tools[tool.name] = tool

    def schemas(self) -> List[dict]:
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, arguments: Any) -> str:
        if name not in self.tools:
            return f"error: unknown tool {name}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                return "error: arguments are not valid JSON"
        try:
            result = self.tools[name].func(**arguments)
        except TypeError as e:
            return f"error: bad arguments for {name}: {e}"
        except Exception as e:  # tools must never crash the assistant
            return f"error: {name} failed: {e}"
        text = str(result)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + "\n...[truncated]"
        return text


def _obj(props: dict, required: List[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


def build_registry(kb: KnowledgeBase, settings: Settings) -> ToolRegistry:
    reg = ToolRegistry()
    workspace = settings.workspace
    workspace.mkdir(parents=True, exist_ok=True)

    def _inside_workspace(path: str) -> Path:
        p = (workspace / path).resolve()
        if workspace.resolve() not in p.parents and p != workspace.resolve():
            raise PermissionError("path is outside the workspace")
        return p

    # --- time ---
    reg.register(Tool(
        "get_time", "Current local date and time.", _obj({}, []),
        lambda: _dt.datetime.now().strftime("%A %Y-%m-%d %H:%M"),
    ))

    # --- web ---
    def web_search(query: str, limit: int = 5) -> str:
        results = web.search(query, limit=int(limit))
        if not results:
            return "no results"
        return "\n".join(f"- {r['title']} | {r['url']}\n  {r['snippet']}" for r in results)

    reg.register(Tool(
        "web_search", "Search the web. Returns titles, URLs and snippets.",
        _obj({"query": {"type": "string"}, "limit": {"type": "integer", "default": 5}}, ["query"]),
        web_search,
    ))

    def fetch_url(url: str) -> str:
        title, text = web.fetch_text(url)
        return f"{title}\n{url}\n\n{text}"

    reg.register(Tool(
        "fetch_url", "Read the readable text of a web page.",
        _obj({"url": {"type": "string"}}, ["url"]), fetch_url,
    ))

    # --- knowledge ---
    def learn_topic(topic: str, sources: int = 3, query: Optional[str] = None) -> str:
        """Search the web for a topic, read the top pages, and store them."""
        q = query or topic
        kb.add_topic(topic)
        added, log = 0, []
        for hit in web.search(q, limit=int(sources) + 3):
            if added >= int(sources):
                break
            if kb.has_source(topic, hit["url"]):
                continue
            try:
                title, text = web.fetch_text(hit["url"])
            except Exception as e:
                log.append(f"skipped {hit['url']}: {e}")
                continue
            chunks = kb.add_source(topic, hit["url"], title or hit["title"], text)
            if chunks:
                added += 1
                log.append(f"stored {chunks} chunks from {hit['url']}")
        summary = f"learned '{topic.strip().lower()}' from {added} source(s)"
        return summary + ("\n" + "\n".join(log) if log else "")

    reg.register(Tool(
        "learn_topic",
        "Learn a topic: search the web, read sources and store them in memory for later answers.",
        _obj({"topic": {"type": "string"}, "sources": {"type": "integer", "default": 3},
              "query": {"type": "string", "description": "optional search query if different from topic"}},
             ["topic"]),
        learn_topic,
    ))

    def recall(query: str, topic: Optional[str] = None, limit: int = 5) -> str:
        hits = kb.search(query, topic=topic, limit=int(limit))
        if not hits:
            return "nothing relevant in memory"
        return "\n\n".join(f"[{h.topic}] {h.title} ({h.url})\n{h.text}" for h in hits)

    reg.register(Tool(
        "recall", "Search what has been learned and stored in memory.",
        _obj({"query": {"type": "string"}, "topic": {"type": "string"},
              "limit": {"type": "integer", "default": 5}}, ["query"]),
        recall,
    ))

    reg.register(Tool(
        "list_topics", "List the topics stored in memory.", _obj({}, []),
        lambda: ", ".join(kb.topics()) or "no topics yet",
    ))

    reg.register(Tool(
        "forget_topic", "Delete a topic and all its stored sources.",
        _obj({"topic": {"type": "string"}}, ["topic"]),
        lambda topic: f"removed {kb.forget_topic(topic)} topic(s)",
    ))

    # --- files (inside the workspace only) ---
    reg.register(Tool(
        "read_file", "Read a text file from the workspace.",
        _obj({"path": {"type": "string"}}, ["path"]),
        lambda path: _inside_workspace(path).read_text(encoding="utf-8")[:MAX_RESULT_CHARS],
    ))

    def write_file(path: str, content: str) -> str:
        p = _inside_workspace(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} characters to {p.relative_to(workspace.resolve())}"

    reg.register(Tool(
        "write_file", "Write a text file in the workspace.",
        _obj({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
        write_file,
    ))

    reg.register(Tool(
        "list_files", "List files in the workspace.", _obj({}, []),
        lambda: "\n".join(sorted(str(p.relative_to(workspace.resolve()))
                                 for p in workspace.resolve().rglob("*") if p.is_file())) or "empty",
    ))

    # --- shell (opt-in) ---
    if settings.allow_shell:
        def run_command(command: str, timeout: int = 20) -> str:
            proc = subprocess.run(command, shell=True, cwd=str(workspace), capture_output=True,
                                  text=True, timeout=int(timeout))
            return f"exit {proc.returncode}\n{proc.stdout}{proc.stderr}"

        reg.register(Tool(
            "run_command", "Run a shell command in the workspace and return its output.",
            _obj({"command": {"type": "string"}, "timeout": {"type": "integer", "default": 20}},
                 ["command"]),
            run_command,
        ))

    return reg
