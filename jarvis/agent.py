"""The assistant: keeps a conversation, calls tools, and returns the final answer."""

from __future__ import annotations

import json
from typing import List

from .config import Settings, load_settings
from .knowledge import KnowledgeBase
from .llm import LLM, LLMError, make_llm
from .tools import ToolRegistry, build_registry

SYSTEM_PROMPT = """You are Jarvis, a capable personal assistant.
- Answer in the language the user writes in (Arabic or English).
- Use tools when they help: web search for current facts, recall for stored knowledge,
  learn_deep when the user asks you to learn a field, files only inside the workspace.
- Your model runs locally. Never ask for an API key or suggest a hosted AI service.
- When you answer from memory or the web, name your sources (URLs) briefly.
- Be direct and concise. If you are not sure, say so instead of guessing.
- Ask before anything destructive. Never claim you did something a tool did not confirm."""


class Assistant:
    MAX_STEPS = 6
    MAX_HISTORY = 40

    def __init__(self, llm: LLM, tools: ToolRegistry) -> None:
        self.llm = llm
        self.tools = tools
        self.history: List[dict] = []

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "Assistant":
        settings = settings or load_settings()
        kb = KnowledgeBase(settings.db_path)
        return cls(make_llm(settings), build_registry(kb, settings))

    def reset(self) -> None:
        self.history.clear()

    def ask(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "Say something and I will help."
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history
        messages.append({"role": "user", "content": text})
        try:
            reply = self._run(messages)
        except LLMError as e:
            return f"Sorry, the language model is unavailable: {e}"
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        self.history = self.history[-self.MAX_HISTORY:]
        return reply

    def _run(self, messages: List[dict]) -> str:
        for _ in range(self.MAX_STEPS):
            reply = self.llm.chat(messages, self.tools.schemas())
            calls = reply.get("tool_calls") or []
            if not calls:
                return (reply.get("content") or "").strip() or "Done."
            messages.append({
                "role": "assistant",
                "content": reply.get("content"),
                "tool_calls": [{"id": c["id"], "type": "function",
                                "function": {"name": c["name"],
                                             "arguments": c["arguments"] if isinstance(c["arguments"], str)
                                             else json.dumps(c["arguments"], ensure_ascii=False)}}
                               for c in calls],
            })
            for c in calls:
                result = self.tools.call(c["name"], c["arguments"])
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
        return "I stopped after several steps without finishing. Please narrow the request."
