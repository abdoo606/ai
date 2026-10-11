"""Language-model backends.

Two backends share one interface, ``chat(messages, tools) -> reply``:

* ``OpenAICompatibleLLM`` talks to any OpenAI-compatible endpoint with tool
  calling (OpenAI, Azure, OpenRouter, local servers such as vLLM or LM Studio).
* ``RuleBasedLLM`` works offline with no key. It understands a small set of
  commands in English and Arabic and calls the matching tools. It is a
  fallback, not a replacement for a real model.

``reply`` is a dict: ``{"content": str | None, "tool_calls": [ {id, name, arguments} ]}``.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Dict, List, Optional, Protocol

from .config import Settings


class LLM(Protocol):
    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> Dict[str, Any]:
        ...


class LLMError(RuntimeError):
    pass


class OpenAICompatibleLLM:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 60) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise LLMError(f"model request failed: {e}") from e
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected response: {str(data)[:200]}") from e
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            calls.append({"id": tc.get("id", ""), "name": fn.get("name", ""),
                          "arguments": fn.get("arguments", "{}"),
                          "raw": tc})
        return {"content": msg.get("content"), "tool_calls": calls, "raw_message": msg}


class RuleBasedLLM:
    """Offline fallback. Maps simple commands to tools and reports tool results."""

    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> Dict[str, Any]:
        # Second step: a tool just ran, so report its result.
        if messages and messages[-1].get("role") == "tool":
            return {"content": self._report(messages[-1]["content"]), "tool_calls": []}

        text = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        call = self._route(text.strip())
        if call is None:
            return {"content": self._help(), "tool_calls": []}
        name, args = call
        return {"content": None, "tool_calls": [{
            "id": f"rule-{abs(hash((name, json.dumps(args, sort_keys=True))))}",
            "name": name, "arguments": json.dumps(args, ensure_ascii=False),
        }]}

    @staticmethod
    def _route(text: str) -> Optional[tuple]:
        low = text.lower()
        if not text:
            return None
        m = re.match(r"^(learn|study|تعلم|تعلّم|ادرس)\s+(.+)$", low)
        if m:
            return ("learn_topic", {"topic": text[len(m.group(1)):].strip()})
        m = re.match(r"^(recall|search memory|استرجع|ابحث في معلوماتي)\s+(.+)$", low)
        if m:
            return ("recall", {"query": text[len(m.group(1)):].strip()})
        m = re.match(r"^(what do you know about|ماذا تعرف عن|ماذا تعرف)\s+(.+)$", low)
        if m:
            return ("recall", {"query": text[len(m.group(1)):].strip()})
        if low in ("topics", "list topics", "المواضيع", "المواضيع اللي تعرفها"):
            return ("list_topics", {})
        if re.search(r"\b(time|date)\b", low) or "الوقت" in low or "الساعة" in low or "التاريخ" in low:
            return ("get_time", {})
        m = re.match(r"^(search|google|ابحث عن|ابحث)\s+(.+)$", low)
        if m:
            return ("web_search", {"query": text[len(m.group(1)):].strip()})
        m = re.match(r"^(forget|انس)\s+(.+)$", low)
        if m:
            return ("forget_topic", {"topic": text[len(m.group(1)):].strip()})
        return None

    @staticmethod
    def _report(result: str) -> str:
        return result if result.strip() else "done"

    @staticmethod
    def _help() -> str:
        return (
            "Offline mode. Commands I understand:\n"
            "- learn <topic>  /  تعلم <موضوع>: read the web and remember it\n"
            "- recall <question>  /  استرجع <سؤال>: answer from memory, with sources\n"
            "- search <query>  /  ابحث عن <شي>: web search\n"
            "- topics  /  المواضيع: what I have learned\n"
            "- time  /  الوقت\n"
            "Set JARVIS_LLM_API_KEY to enable a full language model."
        )


def make_llm(settings: Settings) -> LLM:
    if settings.llm_api_key:
        return OpenAICompatibleLLM(settings.llm_api_key, settings.llm_base_url, settings.llm_model)
    return RuleBasedLLM()
