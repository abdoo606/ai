"""Language-model backends for Jarvis.

All backends are local and share ``chat(messages, tools) -> reply``:

* ``RuleBasedLLM`` is the fast zero-dependency default with fuzzy commands.
* ``HuggingFaceTransformersLLM`` runs one open model locally.
* ``LocalModelEnsembleLLM`` combines multilingual intent with Arabic/English writers.
* ``DeliberativeLocalLLM`` adds senior reasoning, adversarial critique, and revision.
  Logical roles can share one strong model in memory. Everything is lazy and keyless.
"""

from __future__ import annotations

import json
import re
import threading
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Protocol

from .config import Settings
from .knowledge import normalize_text


class LLM(Protocol):
    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> Dict[str, Any]: ...


class LLMError(RuntimeError):
    pass


class HuggingFaceTransformersLLM:
    """Local text-generation backend using Hugging Face Transformers.

    Instances pointing at the same path share one loaded pipeline. This lets
    Arabic writer, English writer, reasoner, and critic be distinct logical
    brains without loading identical multi-billion-parameter weights four times.

    Model loading is delayed until the first message, so importing and serving
    Jarvis stays fast. Downloads are disabled by default: point
    ``JARVIS_HF_MODEL`` at a local model folder, or explicitly opt in once with
    ``JARVIS_HF_ALLOW_DOWNLOAD=1``.
    """

    _shared_generators: Dict[tuple, Any] = {}
    _load_lock = threading.Lock()

    def __init__(self, model: str, allow_download: bool = False, revision: str = "",
                 device: str = "auto", max_new_tokens: int = 512,
                 purpose: str = "general") -> None:
        self.model = model
        self.allow_download = allow_download
        self.revision = revision
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.purpose = purpose
        self._generator = None

    def _load(self):
        if self._generator is not None:
            return self._generator
        cache_key = (self.model, self.device, self.allow_download, self.revision)
        with self._load_lock:
            if cache_key in self._shared_generators:
                self._generator = self._shared_generators[cache_key]
                return self._generator
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            except ImportError as error:
                raise LLMError(
                    "Hugging Face backend needs optional packages: "
                    "pip install -r jarvis/requirements-huggingface.txt"
                ) from error
            kwargs = {"local_files_only": not self.allow_download, "low_cpu_mem_usage": True}
            if self.allow_download and not self.revision:
                raise LLMError(
                    "runtime model downloads require an immutable JARVIS_HF_REVISION; "
                    "prefer jarvis/scripts/install_models.py"
                )
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model, revision=self.revision or None, **kwargs,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model, revision=self.revision or None, torch_dtype="auto", **kwargs,
                )
                if self.device == "auto":
                    device: Any = 0 if torch.cuda.is_available() else (
                        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else -1
                    )
                else:
                    device = int(self.device) if self.device.lstrip("-").isdigit() else self.device
                generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
                self._shared_generators[cache_key] = generator
                self._generator = generator
            except Exception as error:
                hint = (
                    "Install a local model profile with jarvis/scripts/install_models.py, or explicitly set "
                    "JARVIS_HF_ALLOW_DOWNLOAD=1 for initial setup."
                )
                raise LLMError(f"could not load local model '{self.model}': {error}. {hint}") from error
        return self._generator

    def _system_prompt(self, tools: Optional[List[dict]]) -> str:
        schemas = json.dumps(tools or [], ensure_ascii=False, separators=(",", ":"))
        if self.purpose == "intent":
            role = (
                "You are Jarvis's multilingual intent brain. Infer what the user means even when Arabic or "
                "English is dialectal, misspelled, abbreviated, missing punctuation, or incomplete. Prefer a "
                "tool call whenever intent and arguments can be recovered. Never invent a missing destructive "
                "argument; ask one short clarification instead."
            )
        elif self.purpose == "arabic":
            role = (
                "You are Jarvis's Arabic reasoning brain. Understand Modern Standard Arabic and Levantine "
                "dialect, including informal and imperfect spelling. Answer naturally in the user's dialect."
            )
        elif self.purpose == "english":
            role = "You are Jarvis's English answer writer. Produce precise, natural English."
        elif self.purpose == "reasoner":
            role = (
                "You are Jarvis's senior reasoning brain. Decompose difficult tasks, use tools for missing "
                "evidence, connect facts across sources, test assumptions, calculate carefully, and draft a "
                "complete answer. Prefer correctness over speed."
            )
        elif self.purpose == "critic":
            role = (
                "You are Jarvis's adversarial verifier. Find unsupported claims, missed constraints, logical "
                "gaps, contradictions, unsafe actions, and citation problems. Give concise actionable fixes."
            )
        else:
            role = "You are Jarvis, a concise local personal assistant. Reply in the user's language."
        return (
            role + " For complex questions, reason internally in steps: retrieve evidence, inspect concept "
            "links, compare sources and contradictions, then provide only the conclusion and supporting "
            "evidence. Never expose private chain-of-thought or mistake co-occurrence for causation. "
            "You may call one or more tools. Return exactly one JSON object and no markdown: "
            '{"content":"answer or null","tool_calls":['
            '{"id":"call-1","name":"tool_name","arguments":{"arg":"value"}}]}. '
            "When a tool is needed, content must be null. When no tool is needed, tool_calls must be []. "
            f"Available tool schemas: {schemas}"
        )

    @classmethod
    def _parse_reply(cls, generated: str) -> Dict[str, Any]:
        decoder = json.JSONDecoder()
        parsed = None
        for start, char in enumerate(generated):
            if char != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(generated[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and ("content" in candidate or "tool_calls" in candidate):
                parsed = candidate
                break
        if parsed is None:
            return {"content": generated.strip(), "tool_calls": []}

        calls = []
        for index, call in enumerate(parsed.get("tool_calls") or []):
            if not isinstance(call, dict) or not call.get("name"):
                continue
            arguments = call.get("arguments", {})
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            calls.append({
                "id": str(call.get("id") or f"hf-{index + 1}"),
                "name": str(call["name"]),
                "arguments": arguments,
            })
        content = parsed.get("content")
        return {"content": str(content) if content is not None else None, "tool_calls": calls}

    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> Dict[str, Any]:
        generator = self._load()
        conversation = [{"role": "system", "content": self._system_prompt(tools)}]
        for message in messages[-24:]:
            role = message.get("role", "user")
            if role == "system":
                continue  # the stricter JSON/tool system prompt above is authoritative
            content = message.get("content") or ""
            if role == "tool":
                content = f"Tool result ({message.get('name', 'tool')}): {content}"
                role = "user"
            conversation.append({"role": role if role in {"user", "assistant"} else "user", "content": content})
        tokenizer = generator.tokenizer
        try:
            prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = "\n".join(f"{item['role'].upper()}: {item['content']}" for item in conversation) + "\nASSISTANT:"
        try:
            output = generator(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_full_text=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = output[0]["generated_text"]
        except Exception as error:
            raise LLMError(f"local Hugging Face generation failed: {error}") from error
        return self._parse_reply(generated)


def detect_language(text: str) -> str:
    """Return ``arabic`` or ``english`` for model routing, including mixed text."""
    arabic = len(re.findall(r"[\u0600-\u06ff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return "arabic" if arabic and arabic >= latin * 0.35 else "english"


class LocalModelEnsembleLLM:
    """Three-brain local ensemble: intent understanding + Arabic/English response.

    A deterministic fuzzy router handles obvious commands first for speed. The
    intent model is used only for ambiguous/incomplete input, then exactly one
    language model handles reasoning and response. All models are lazy-loaded.
    """

    def __init__(self, intent: HuggingFaceTransformersLLM,
                 english: HuggingFaceTransformersLLM,
                 arabic: HuggingFaceTransformersLLM) -> None:
        self.intent = intent
        self.english = english
        self.arabic = arabic

    @staticmethod
    def _latest_user(messages: List[dict]) -> str:
        return next((str(message.get("content") or "") for message in reversed(messages)
                     if message.get("role") == "user"), "")

    def _response_model(self, messages: List[dict]) -> HuggingFaceTransformersLLM:
        return self.arabic if detect_language(self._latest_user(messages)) == "arabic" else self.english

    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> Dict[str, Any]:
        # A tool just returned: let the matching language brain synthesize the evidence.
        if messages and messages[-1].get("role") == "tool":
            return self._response_model(messages).chat(messages, tools)

        text = self._latest_user(messages)
        deterministic = RuleBasedLLM._route(text)
        if deterministic is not None:
            name, arguments = deterministic
            return {"content": None, "tool_calls": [{
                "id": f"fast-{abs(hash((name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))))}",
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            }]}

        # Ambiguous text goes through the small intent model. If its local files
        # are unavailable, the language model still gets a chance to understand it.
        try:
            decision = self.intent.chat(messages, tools)
            if decision.get("tool_calls"):
                return decision
        except LLMError:
            pass
        return self._response_model(messages).chat(messages, tools)


class DeliberativeLocalLLM(LocalModelEnsembleLLM):
    """A slower, stronger local pipeline: understand, reason, critique, revise.

    The response writers, reasoner, and critic may share one physical model
    pipeline while receiving different role prompts. This provides multiple
    independent passes without multiplying model memory.
    """

    def __init__(self, intent: HuggingFaceTransformersLLM,
                 reasoner: HuggingFaceTransformersLLM,
                 critic: HuggingFaceTransformersLLM,
                 english: HuggingFaceTransformersLLM,
                 arabic: HuggingFaceTransformersLLM,
                 rounds: int = 1) -> None:
        super().__init__(intent, english, arabic)
        self.reasoner = reasoner
        self.critic = critic
        self.rounds = min(max(int(rounds), 1), 3)

    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> Dict[str, Any]:
        text = self._latest_user(messages)
        if not (messages and messages[-1].get("role") == "tool"):
            deterministic = RuleBasedLLM._route(text)
            if deterministic is not None:
                name, arguments = deterministic
                return {"content": None, "tool_calls": [{
                    "id": f"fast-{abs(hash((name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))))}",
                    "name": name, "arguments": json.dumps(arguments, ensure_ascii=False),
                }]}
            try:
                intent = self.intent.chat(messages, tools)
                if intent.get("tool_calls"):
                    return intent
            except LLMError:
                pass

        # The senior reasoner may request more evidence. The outer Assistant
        # loop executes those tools and calls us again with their results.
        draft_reply = self.reasoner.chat(messages, tools)
        if draft_reply.get("tool_calls"):
            return draft_reply
        draft = (draft_reply.get("content") or "").strip()
        if not draft:
            return self._response_model(messages).chat(messages, tools)

        writer = self._response_model(messages)
        for _ in range(self.rounds):
            review_messages = [*messages, {"role": "assistant", "content": draft}, {
                "role": "user",
                "content": (
                    "Internal verification pass: critique the draft against the user's exact request and all "
                    "available evidence. List only concrete errors, omissions, unsupported claims, and fixes."
                ),
            }]
            try:
                review = self.critic.chat(review_messages, [])
                critique = (review.get("content") or "No specific issue found.").strip()
            except LLMError:
                critique = "Verification model unavailable; preserve evidence and avoid unsupported claims."
            revision_messages = [*messages, {"role": "assistant", "content": draft}, {
                "role": "user",
                "content": (
                    "Internal reviewer feedback follows. Revise the draft silently. Return only the final "
                    "answer in my language, with concise reasoning, uncertainty, and source citations where "
                    f"available. Feedback: {critique}"
                ),
            }]
            try:
                revised = writer.chat(revision_messages, tools)
            except LLMError:
                break
            if revised.get("tool_calls"):
                return revised
            revised_text = (revised.get("content") or "").strip()
            if revised_text:
                draft = revised_text
        return {"content": draft, "tool_calls": []}


class RuleBasedLLM:
    """Offline router with useful source-grounded memory answers."""

    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> Dict[str, Any]:
        if messages and messages[-1].get("role") == "tool":
            return {"content": self._report(messages[-1]["content"]), "tool_calls": []}
        text = next((message["content"] for message in reversed(messages)
                     if message.get("role") == "user"), "")
        call = self._route(text.strip())
        if call is None:
            return {"content": self._help(), "tool_calls": []}
        name, arguments = call
        return {"content": None, "tool_calls": [{
            "id": f"rule-{abs(hash((name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))))}",
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        }]}

    @staticmethod
    def _clean_command(text: str) -> str:
        low = normalize_text(text.strip())
        low = re.sub(r"[،,:;!]+", " ", low)
        low = re.sub(r"\s+", " ", low).strip()
        fillers = re.compile(
            r"^(?:لو سمحت|من فضلك|بدي ياه|بدي ياك|بديك|بدي|ممكن|فيني اطلب|"
            r"روح|يلا|please|can you|could you|would you|i want you to)\s+"
        )
        previous = None
        while low != previous:
            previous = low
            low = fillers.sub("", low).strip()
        return low

    @staticmethod
    def _fuzzy_command(low: str) -> Optional[tuple]:
        """Recover common command intent from misspelled or conversational text."""
        aliases = {
            "learn_deep": ("topic", {"learn", "study", "تعلم", "اتعلم", "يتعلم", "ادرس", "دروس"}),
            "recall": ("query", {"recall", "تذكر", "استرجع", "جاوب"}),
            "summarize_topic": ("topic", {"summarize", "لخص", "تلخيص", "ملخص"}),
            "web_search": ("query", {"search", "google", "ابحث", "دور"}),
            "connect_ideas": ("query", {"connect", "link", "اربط", "علاقه"}),
        }
        words = low.split()
        for index, word in enumerate(words[:-1]):
            best_tool, best_argument, best_score = None, None, 0.0
            for tool, (argument, commands) in aliases.items():
                for command in commands:
                    score = SequenceMatcher(None, word, command).ratio()
                    if score > best_score:
                        best_tool, best_argument, best_score = tool, argument, score
            if best_score >= 0.72:
                value = " ".join(words[index + 1:]).strip()
                if value:
                    return best_tool, {best_argument: value}
        return None

    @staticmethod
    def _route(text: str) -> Optional[tuple]:
        low = RuleBasedLLM._clean_command(text)
        if not low:
            return None
        patterns = [
            (r"^(learn|study|تعلم|اتعلم|يتعلم|ادرس)\s+(.+)$", "learn_deep", "topic"),
            (r"^(quick learn|تعلم بسرعه)\s+(.+)$", "learn_topic", "topic"),
            (r"^(recall|search memory|استرجع|ابحث في معلوماتي)\s+(.+)$", "recall", "query"),
            (r"^(what do you know about|ماذا تعرف عن|ماذا تعرف)\s+(.+)$", "recall", "query"),
            (r"^(summarize|summary of|لخص|ملخص)\s+(.+)$", "summarize_topic", "topic"),
            (r"^(sources for|list sources|مصادر|المصادر)\s+(.+)$", "list_sources", "topic"),
            (r"^(connect|link|اربط|شو العلاقه بين|ما العلاقه بين)\s+(.+)$", "connect_ideas", "query"),
            (r"^(search|google|ابحث عن|ابحث)\s+(.+)$", "web_search", "query"),
            (r"^(forget|انسي|انس)\s+(.+)$", "forget_topic", "topic"),
        ]
        for pattern, tool, argument in patterns:
            match = re.match(pattern, low)
            if match:
                return tool, {argument: match.group(2).strip()}
        if low in {"topics", "list topics", "المواضيع", "المواضيع اللي تعرفها"}:
            return "list_topics", {}
        if low in {"rebuild connections", "rebuild graph", "اعد بناء الروابط", "حدث الروابط"}:
            return "rebuild_connections", {}
        if re.search(r"\b(time|date)\b", low) or any(word in low for word in ("الوقت", "الساعة", "الساعه", "التاريخ")):
            return "get_time", {}
        # Question punctuation is stronger evidence than a fuzzy command match
        # (for example Arabic "تعمل" must not be mistaken for "تعلم").
        if text.rstrip().endswith(("?", "؟")):
            return "recall", {"query": text.rstrip("?؟ ")}
        fuzzy = RuleBasedLLM._fuzzy_command(low)
        if fuzzy is not None:
            return fuzzy
        return None

    @staticmethod
    def _report(result: str) -> str:
        return result if result.strip() else "done"

    @staticmethod
    def _help() -> str:
        return (
            "Offline mode. Commands I understand:\n"
            "- learn <topic> / اتعلم <موضوع>: map the field deeply and archive its sources\n"
            "- recall <question> / استرجع <سؤال>: concise answer with citations\n"
            "- summarize <topic> / لخص <موضوع>\n"
            "- sources for <topic> / مصادر <موضوع>\n"
            "- search <query> / ابحث عن <شي>\n"
            "- topics / المواضيع; time / الوقت\n"
            "You can also ask a direct question ending with ?."
        )


def make_llm(settings: Settings) -> LLM:
    common = {"allow_download": settings.hf_allow_download, "revision": settings.hf_revision,
              "device": settings.hf_device}
    if settings.llm_mode == "local-deliberative":
        return DeliberativeLocalLLM(
            intent=HuggingFaceTransformersLLM(
                settings.hf_intent_model, max_new_tokens=settings.hf_intent_max_new_tokens,
                purpose="intent", **common,
            ),
            reasoner=HuggingFaceTransformersLLM(
                settings.hf_reasoning_model, max_new_tokens=settings.hf_max_new_tokens,
                purpose="reasoner", **common,
            ),
            critic=HuggingFaceTransformersLLM(
                settings.hf_critic_model, max_new_tokens=settings.hf_max_new_tokens,
                purpose="critic", **common,
            ),
            english=HuggingFaceTransformersLLM(
                settings.hf_english_model, max_new_tokens=settings.hf_max_new_tokens,
                purpose="english", **common,
            ),
            arabic=HuggingFaceTransformersLLM(
                settings.hf_arabic_model, max_new_tokens=settings.hf_max_new_tokens,
                purpose="arabic", **common,
            ),
            rounds=settings.reasoning_rounds,
        )
    if settings.llm_mode == "local-ensemble":
        return LocalModelEnsembleLLM(
            intent=HuggingFaceTransformersLLM(
                settings.hf_intent_model, max_new_tokens=settings.hf_intent_max_new_tokens,
                purpose="intent", **common,
            ),
            english=HuggingFaceTransformersLLM(
                settings.hf_english_model, max_new_tokens=settings.hf_max_new_tokens,
                purpose="english", **common,
            ),
            arabic=HuggingFaceTransformersLLM(
                settings.hf_arabic_model, max_new_tokens=settings.hf_max_new_tokens,
                purpose="arabic", **common,
            ),
        )
    if settings.llm_mode == "huggingface-local":
        return HuggingFaceTransformersLLM(
            settings.hf_model, max_new_tokens=settings.hf_max_new_tokens, **common,
        )
    return RuleBasedLLM()
