"""Offline tests for Jarvis. No network, no API key, no real model.

Run: pytest -q jarvis/tests
"""

import json
import sqlite3
from pathlib import Path

import pytest

from jarvis import web
from jarvis.agent import Assistant
from jarvis.config import load_settings
from jarvis.knowledge import KnowledgeBase, chunk_text, normalize_text
from jarvis.llm import (
    DeliberativeLocalLLM,
    HuggingFaceTransformersLLM,
    LLMError,
    LocalModelEnsembleLLM,
    RuleBasedLLM,
    detect_language,
    make_llm,
)
from jarvis.research import DeepResearcher, canonical_url, curriculum_queries, discover_subtopics
from jarvis.tools import build_registry

# ---------- fixtures -------------------------------------------------------

@pytest.fixture
def settings(tmp_path):
    return load_settings({
        "JARVIS_HOME": str(tmp_path / "home"),
        "JARVIS_WORKSPACE": str(tmp_path / "ws"),
    })


@pytest.fixture
def kb():
    k = KnowledgeBase(":memory:")
    yield k
    k.close()


@pytest.fixture
def fake_web(monkeypatch):
    """Replace network calls with fixtures."""
    pages = {
        "https://a.example/qubits": ("Qubits", "<html><title>Qubits</title><body><p>A qubit is a quantum bit that can be in superposition.</p>"
                                                "<script>var x=1;</script></body></html>"),
        "https://b.example/entangle": ("Entanglement", "<p>Entanglement links two qubits so measuring one tells you about the other.</p>"),
    }
    results = [
        {"url": "https://a.example/qubits", "title": "Qubits", "snippet": "qubit basics"},
        {"url": "https://b.example/entangle", "title": "Entanglement", "snippet": "entanglement"},
    ]
    monkeypatch.setattr(web, "search", lambda q, limit=5: results[:limit])

    def fetch(url, allow_private=False):
        if url not in pages:
            raise OSError("no such fixture page")
        title, body = pages[url]
        return web.html_to_text(body if body.startswith("<") else f"<html><title>{title}</title><body>{body}</body></html>")

    monkeypatch.setattr(web, "fetch_text", fetch)
    return pages


# ---------- chunking -------------------------------------------------------

def test_chunk_text_short_text_is_one_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_chunk_text_respects_size_and_keeps_all_words():
    text = "\n\n".join(f"paragraph {i} " + "word " * 80 for i in range(20))
    chunks = chunk_text(text, size=300)
    assert chunks and all(len(c) <= 300 for c in chunks)
    joined = " ".join(chunks)
    for i in range(20):
        assert f"paragraph {i}" in joined


def test_chunk_text_hard_splits_very_long_paragraph():
    chunks = chunk_text("x" * 2000, size=800)
    assert [len(c) for c in chunks] == [800, 800, 400]


def test_chunk_text_empty():
    assert chunk_text("   \n\n  ") == []


# ---------- knowledge base: FTS5 search -----------------------------------

def test_search_ranks_relevant_chunk_first(kb):
    kb.add_source("physics", "https://p.example/a", "Quantum", "Quantum entanglement connects particles across distance.")
    kb.add_source("physics", "https://p.example/b", "Cooking", "Bake the bread for forty minutes at high heat.")
    hits = kb.search("entanglement particles")
    assert hits and hits[0].url == "https://p.example/a"
    assert hits[0].score <= 0  # bm25: lower is better, all matches are negative or zero


def test_search_is_safe_against_fts_syntax(kb):
    kb.add_source("t", "https://x.example", "X", "Some ordinary text about rivers.")
    # Quotes, operators and colons must not crash the FTS parser.
    assert kb.search('rivers" OR NEAR(') is not None
    assert kb.search("   ") == []


def test_search_filters_by_topic(kb):
    kb.add_source("one", "https://1.example", "1", "shared word here")
    kb.add_source("two", "https://2.example", "2", "shared word there")
    hits = kb.search("shared", topic="two")
    assert [h.topic for h in hits] == ["two"]


def test_search_finds_arabic_text(kb):
    kb.add_source("ar", "https://ar.example", "عربي", "الحوسبة الكمومية تستخدم الكيوبت للحساب.")
    assert kb.search("الكيوبت")


def test_arabic_search_normalizes_diacritics_alef_and_ya(kb):
    kb.add_source("إدارة", "https://ar.example/admin", "إدارة", "إِدارة المشاريع مسؤولية كبرى.")
    assert normalize_text("إِدارة") == "ادارة"
    assert kb.search("ادارة المشاريع", topic="ادارة")


def test_recall_builds_concise_answer_with_citations(kb):
    kb.add_source("physics", "https://p.example", "Quantum", (
        "A qubit stores quantum information. It may exist in a superposition of states. "
        "Measurement produces a classical result."
    ))
    answer = kb.answer("How does qubit superposition work?")
    assert "superposition" in answer
    assert "Sources:" in answer
    assert "https://p.example" in answer


def test_concept_graph_links_points_and_expands_retrieval(kb):
    kb.add_source("physics", "https://p.example/mechanics", "Mechanics", (
        "Force changes motion and acceleration. Energy describes the ability to do work. "
        "Force, motion, energy, and acceleration are central mechanics concepts."
    ))
    kb.add_source("physics", "https://p.example/relativity", "Relativity", (
        "Energy and mass are related in relativity. Spacetime geometry describes gravity."
    ))
    related = kb.related_terms("force", topic="physics")
    assert "motion" in related or "acceleration" in related
    links = kb.connections("energy", topic="physics", limit=30)
    assert links and any("mass" in (link.source, link.target) for link in links)
    answer = kb.answer("How is force connected to motion?", topic="physics")
    assert "Connections / الروابط" in answer
    assert "https://p.example/mechanics" in answer


def test_rebuild_connections_indexes_legacy_chunks(kb):
    kb.add_source("biology", "https://b.example", "Cells", "Cells contain DNA and proteins. " * 8)
    kb.db.execute("DELETE FROM concepts")
    kb.db.execute("DELETE FROM concept_relations")
    kb.db.commit()
    kb._related_cache.clear()
    assert kb.connections("cells", topic="biology") == []
    assert kb.rebuild_connections("biology") >= 1
    assert kb.connections("cells", topic="biology")


def test_duplicate_source_is_not_stored_twice(kb):
    assert kb.add_source("t", "https://d.example", "D", "content here") == 1
    assert kb.add_source("t", "https://d.example", "D", "content here") == 0
    assert len(kb.sources("t")) == 1


def test_forget_topic_removes_sources_and_chunks(kb):
    kb.add_source("gone", "https://g.example", "G", "delete me please")
    assert kb.forget_topic("gone") == 1
    assert "gone" not in kb.topics()
    assert kb.search("delete") == []
    assert kb.sources("gone") == []


def test_knowledge_persists_on_disk(tmp_path):
    path = tmp_path / "k.sqlite3"
    a = KnowledgeBase(path)
    a.add_source("p", "https://p.example", "P", "persistent memory works")
    a.close()
    b = KnowledgeBase(path)
    assert b.search("persistent")[0].url == "https://p.example"
    b.close()


def test_old_fts_schema_is_migrated_without_losing_chunks(tmp_path):
    path = tmp_path / "old.sqlite3"
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE topics(name TEXT PRIMARY KEY, created REAL NOT NULL);
        CREATE TABLE sources(id INTEGER PRIMARY KEY, topic TEXT NOT NULL, url TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '', fetched REAL NOT NULL, UNIQUE(topic, url));
        CREATE VIRTUAL TABLE chunks USING fts5(text, topic UNINDEXED, url UNINDEXED, title UNINDEXED);
        INSERT INTO topics VALUES ('arabic', 1);
        INSERT INTO sources(topic, url, title, fetched) VALUES ('arabic', 'https://old.example', 'Old', 1);
        INSERT INTO chunks(text, topic, url, title)
            VALUES ('إِدارة المشاريع مهمة', 'arabic', 'https://old.example', 'Old');
    """)
    db.close()
    kb = KnowledgeBase(path)
    assert kb.search("ادارة")[0].url == "https://old.example"
    kb.close()


def test_empty_topic_name_rejected(kb):
    with pytest.raises(ValueError):
        kb.add_topic("   ")


# ---------- web: parsing ---------------------------------------------------

def test_html_to_text_drops_scripts_and_styles():
    title, text = web.html_to_text(
        "<html><head><title>Hi &amp; bye</title><style>p{}</style></head>"
        "<body><script>evil()</script><p>Visible &lt;text&gt;</p></body></html>")
    assert title == "Hi & bye"
    assert "Visible <text>" in text
    assert "evil" not in text and "p{}" not in text


def test_parse_duckduckgo_results_decodes_redirect_urls():
    page = """
    <div class="result"><h2><a class="result__a"
      href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fpage&amp;rut=abc">Example &amp; Co</a></h2>
      <a class="result__snippet">A short snippet.</a></div>
    <div class="result"><h2><a class="result__a" href="javascript:void(0)">bad</a></h2></div>
    """
    results = web.parse_search_results(page)
    assert results == [{"url": "https://example.org/page", "title": "Example & Co", "snippet": "A short snippet."}]


def test_fetch_text_rejects_non_http_urls():
    with pytest.raises(ValueError):
        web.fetch_text("file:///etc/passwd")


def test_web_url_security_blocks_credentials_and_private_networks(monkeypatch):
    with pytest.raises(ValueError, match="credentials"):
        web.validate_web_url("https://user:password@example.com/secret")
    monkeypatch.setattr(web.socket, "getaddrinfo", lambda *args: [
        (2, 1, 6, "", ("127.0.0.1", 80)),
    ])
    with pytest.raises(ValueError, match="private"):
        web.validate_web_url("http://internal.example/admin")
    # Explicit private-web opt-in is available for intentional local integrations.
    assert web.validate_web_url("http://127.0.0.1:9000", allow_private=True)


def test_web_url_security_accepts_public_resolution(monkeypatch):
    monkeypatch.setattr(web.socket, "getaddrinfo", lambda *args: [
        (2, 1, 6, "", ("93.184.216.34", 443)),
    ])
    assert web.validate_web_url("https://example.com/page") == "https://example.com/page"


# ---------- autonomous deep research --------------------------------------

def test_curriculum_is_broad_and_multilingual():
    english = curriculum_queries("physics")
    arabic = curriculum_queries("الفيزياء")
    assert len(english) >= 10 and any("current research" in query for query in english)
    assert len(arabic) >= 10 and any("أحدث الأبحاث" in query for query in arabic)


def test_research_url_cleanup_and_subtopic_discovery():
    assert canonical_url("HTTPS://Example.org/x/?utm_source=a&ok=1#part") == "https://example.org/x?ok=1"
    results = [
        {"title": "Quantum mechanics and particles", "snippet": "Quantum theory"},
        {"title": "Quantum mechanics experiments", "snippet": "Quantum particles"},
    ]
    assert "Quantum" in discover_subtopics("physics", results)


def test_deep_research_maps_queries_deduplicates_and_stores(kb):
    pages = {
        "https://learn.example/mechanics": ("Classical mechanics", "Mechanics explains motion force and energy. " * 25),
        "https://learn.example/quantum": ("Quantum mechanics", "Quantum theory explains particles measurement and uncertainty. " * 25),
    }

    def search(query, limit=5):
        return [
            {"url": "https://learn.example/mechanics?utm_source=test", "title": "Classical mechanics", "snippet": "mechanics motion"},
            {"url": "https://learn.example/quantum", "title": "Quantum mechanics", "snippet": "mechanics particles"},
        ][:limit]

    def fetch(url):
        return pages[url]

    report = DeepResearcher(kb, search=search, fetch=fetch).learn(
        "physics", max_sources=2, depth=2, results_per_query=2
    )
    assert report.stored_sources == 2
    assert report.stored_chunks >= 2
    assert len(report.queries) >= 1
    assert kb.search("uncertainty", topic="physics")
    assert "stored 2 source(s)" in report.render()


# ---------- tools ----------------------------------------------------------

def test_registry_has_expected_tools(kb, settings):
    names = set(build_registry(kb, settings).tools)
    assert {"get_time", "web_search", "fetch_url", "learn_topic", "learn_deep", "recall",
            "list_sources", "summarize_topic", "connect_ideas", "rebuild_connections",
            "list_topics", "forget_topic",
            "read_file", "write_file", "list_files"} <= names
    assert "run_command" not in names  # shell is off by default


def test_shell_only_when_enabled(kb, tmp_path):
    on = load_settings({"JARVIS_HOME": str(tmp_path), "JARVIS_WORKSPACE": str(tmp_path / "w"),
                        "JARVIS_ALLOW_SHELL": "1", "JARVIS_ALLOWED_COMMANDS": "echo,python"})
    registry = build_registry(kb, on)
    assert "run_command" in registry.tools
    assert "hello" in registry.call("run_command", {"command": "echo hello"})
    assert "not in JARVIS_ALLOWED_COMMANDS" in registry.call(
        "run_command", {"command": "sh -c 'touch escaped'"})
    shell_without_allowlist = load_settings({"JARVIS_HOME": str(tmp_path),
                                             "JARVIS_ALLOW_SHELL": "1"})
    assert "run_command" not in build_registry(kb, shell_without_allowlist).tools


def test_file_tools_are_confined_to_workspace(kb, settings):
    reg = build_registry(kb, settings)
    assert "wrote" in reg.call("write_file", {"path": "notes/a.txt", "content": "hello"})
    assert reg.call("read_file", {"path": "notes/a.txt"}) == "hello"
    assert "notes/a.txt" in reg.call("list_files", {})
    for bad in ("../escape.txt", "/etc/passwd", "notes/../../escape.txt"):
        out = reg.call("write_file", {"path": bad, "content": "x"})
        assert out.startswith("error:") and "outside the workspace" in out
        assert "outside the workspace" in reg.call("read_file", {"path": bad})
    assert not (Path(settings.workspace).parent / "escape.txt").exists()


def test_tool_errors_are_returned_not_raised(kb, settings):
    reg = build_registry(kb, settings)
    assert reg.call("no_such_tool", {}).startswith("error: unknown tool")
    assert reg.call("recall", {}).startswith("error: bad arguments")
    assert reg.call("recall", "{not json").startswith("error: arguments are not valid JSON")


def test_tool_output_is_capped(kb, settings):
    reg = build_registry(kb, settings)
    reg.call("write_file", {"path": "big.txt", "content": "y" * 10000})
    out = reg.call("read_file", {"path": "big.txt"})
    assert len(out) <= 4000


def test_learn_topic_reads_fixture_pages_and_recall_finds_them(kb, settings, fake_web):
    reg = build_registry(kb, settings)
    summary = reg.call("learn_topic", {"topic": "Quantum", "sources": 2})
    assert "learned 'quantum' from 2 source(s)" in summary
    assert "quantum" in reg.call("list_topics", {})
    answer = reg.call("recall", {"query": "superposition qubit"})
    assert "https://a.example/qubits" in answer and "Sources:" in answer
    assert "https://a.example/qubits" in reg.call("list_sources", {"topic": "quantum"})
    assert "https://a.example/qubits" in reg.call("summarize_topic", {"topic": "quantum"})
    # Second run must not duplicate sources.
    again = reg.call("learn_topic", {"topic": "Quantum", "sources": 2})
    assert "from 0 source(s)" in again


def test_learn_topic_skips_failing_pages(kb, settings, fake_web, monkeypatch):
    results = [{"url": "https://broken.example", "title": "x", "snippet": ""}]
    monkeypatch.setattr(web, "search", lambda q, limit=5: results)
    reg = build_registry(kb, settings)
    out = reg.call("learn_topic", {"topic": "broken"})
    assert "skipped https://broken.example" in out
    assert "from 0 source(s)" in out


# ---------- rule-based intents --------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("learn quantum computing", ("learn_deep", {"topic": "quantum computing"})),
    ("تعلم الحوسبة الكمومية", ("learn_deep", {"topic": "الحوسبة الكمومية"})),
    ("اتعلم الفيزياء", ("learn_deep", {"topic": "الفيزياء"})),
    ("بدي ياه يتعلم فيزيا", ("learn_deep", {"topic": "فيزيا"})),
    ("لو سمحت اتعالم الفيزيا", ("learn_deep", {"topic": "الفيزيا"})),
    ("plese serch quantum gravity", ("web_search", {"query": "quantum gravity"})),
    ("recall what is a qubit", ("recall", {"query": "what is a qubit"})),
    ("ماذا تعرف عن الكيوبت", ("recall", {"query": "الكيوبت"})),
    ("what time is it", ("get_time", {})),
    ("ما هي الساعة", ("get_time", {})),
    ("topics", ("list_topics", {})),
    ("search python sqlite", ("web_search", {"query": "python sqlite"})),
    ("summarize quantum", ("summarize_topic", {"topic": "quantum"})),
    ("لخّص الحوسبة", ("summarize_topic", {"topic": "الحوسبة"})),
    ("مصادر quantum", ("list_sources", {"topic": "quantum"})),
    ("اربط الطاقة والكتلة", ("connect_ideas", {"query": "الطاقة والكتلة"})),
    ("كيف تعمل الكيوبتات؟", ("recall", {"query": "كيف تعمل الكيوبتات"})),
    ("forget quantum", ("forget_topic", {"topic": "quantum"})),
])
def test_rule_intents(text, expected):
    name, args = RuleBasedLLM._route(text)
    assert (name, args) == expected


def test_rule_unknown_text_returns_help():
    reply = RuleBasedLLM().chat([{"role": "user", "content": "tell me a joke"}])
    assert reply["tool_calls"] == [] and "learn <topic>" in reply["content"]


# ---------- agent loop -----------------------------------------------------

class ScriptedLLM:
    """Returns pre-set replies in order, to test the loop deterministically."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, tools=None):
        self.seen.append(list(messages))
        return self.replies.pop(0)


class FakeTools:
    def __init__(self):
        self.calls = []

    def schemas(self):
        return []

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        return f"result of {name}"


def test_agent_runs_tool_then_answers():
    llm = ScriptedLLM([
        {"content": None, "tool_calls": [{"id": "1", "name": "get_time", "arguments": "{}"}]},
        {"content": "It is noon.", "tool_calls": []},
    ])
    tools = FakeTools()
    bot = Assistant(llm, tools)
    assert bot.ask("time?") == "It is noon."
    assert tools.calls == [("get_time", "{}")]
    # The tool result was fed back to the model.
    assert any(m.get("role") == "tool" and m["content"] == "result of get_time" for m in llm.seen[1])


def test_agent_stops_after_max_steps():
    looping = {"content": None, "tool_calls": [{"id": "x", "name": "get_time", "arguments": "{}"}]}
    bot = Assistant(ScriptedLLM([looping] * Assistant.MAX_STEPS), FakeTools())
    assert "stopped" in bot.ask("loop forever")


def test_agent_keeps_history_and_reset_clears_it():
    llm = ScriptedLLM([
        {"content": "first", "tool_calls": []},
        {"content": "second", "tool_calls": []},
    ])
    bot = Assistant(llm, FakeTools())
    bot.ask("one")
    bot.ask("two")
    assert any(m["content"] == "one" for m in llm.seen[1])
    bot.reset()
    assert bot.history == []


def test_agent_reports_llm_failure_gracefully():
    class Broken:
        def chat(self, messages, tools=None):
            raise LLMError("network down")

    bot = Assistant(Broken(), FakeTools())
    assert "network down" in bot.ask("hi")
    assert bot.history == []


def test_end_to_end_offline_learn_then_recall(settings, fake_web):
    kb = KnowledgeBase(settings.db_path)
    bot = Assistant(RuleBasedLLM(), build_registry(kb, settings))
    assert "learned" in bot.ask("quick learn quantum")
    assert "superposition" in bot.ask("recall superposition")
    kb.close()


def test_settings_default_to_offline_and_has_no_api_configuration(tmp_path):
    settings = load_settings({"JARVIS_HOME": str(tmp_path), "JARVIS_LLM_API_KEY": "ignored"})
    assert settings.llm_mode == "offline-rules" and settings.allow_shell is False
    assert not hasattr(settings, "llm_api_key")
    # Hosted backend names cannot activate a network model.
    assert load_settings({"JARVIS_BACKEND": "openai"}).llm_mode == "offline-rules"


def test_huggingface_backend_is_opt_in_local_and_lazy(tmp_path):
    settings = load_settings({
        "JARVIS_HOME": str(tmp_path),
        "JARVIS_BACKEND": "huggingface",
        "JARVIS_HF_MODEL": str(tmp_path / "model"),
    })
    assert settings.llm_mode == "huggingface-local"
    assert settings.hf_allow_download is False
    model = make_llm(settings)
    assert isinstance(model, HuggingFaceTransformersLLM)
    assert model._generator is None  # importing Jarvis does not load a multi-GB model


def test_three_model_ensemble_configuration_is_local_and_lazy(tmp_path):
    settings = load_settings({
        "JARVIS_HOME": str(tmp_path), "JARVIS_BACKEND": "ensemble",
        "JARVIS_HF_ENGLISH_MODEL": str(tmp_path / "en"),
        "JARVIS_HF_ARABIC_MODEL": str(tmp_path / "ar"),
        "JARVIS_HF_INTENT_MODEL": str(tmp_path / "intent"),
    })
    model = make_llm(settings)
    assert settings.llm_mode == "local-ensemble"
    assert isinstance(model, LocalModelEnsembleLLM)
    assert model.english.model.endswith("en") and model.english.purpose == "english"
    assert model.arabic.model.endswith("ar") and model.arabic.purpose == "arabic"
    assert model.intent.model.endswith("intent") and model.intent.purpose == "intent"
    assert all(brain._generator is None for brain in (model.intent, model.english, model.arabic))


def test_language_detection_routes_mixed_and_incomplete_text():
    assert detect_language("مرحبا Jarvis بدي help") == "arabic"
    assert detect_language("please explain physics") == "english"


class StubBrain:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append((messages, tools))
        return self.reply


def test_ensemble_uses_fast_fuzzy_intent_before_loading_models():
    intent = StubBrain({"content": "unused", "tool_calls": []})
    english = StubBrain({"content": "unused", "tool_calls": []})
    arabic = StubBrain({"content": "unused", "tool_calls": []})
    model = LocalModelEnsembleLLM(intent, english, arabic)
    reply = model.chat([{"role": "user", "content": "لو سمحت اتعالم الفيزيا"}], [])
    assert reply["tool_calls"][0]["name"] == "learn_deep"
    assert json.loads(reply["tool_calls"][0]["arguments"])["topic"] == "الفيزيا"
    assert not intent.calls and not english.calls and not arabic.calls


def test_ensemble_uses_intent_brain_then_matching_language_brain():
    intent = StubBrain({"content": "conversation", "tool_calls": []})
    english = StubBrain({"content": "hello", "tool_calls": []})
    arabic = StubBrain({"content": "اهلا", "tool_calls": []})
    model = LocalModelEnsembleLLM(intent, english, arabic)
    reply = model.chat([{"role": "user", "content": "مرحبا شو فينا نعمل"}], [])
    assert reply["content"] == "اهلا"
    assert len(intent.calls) == 1 and len(arabic.calls) == 1 and not english.calls


def test_ensemble_accepts_intent_model_tool_decision():
    decision = {"content": None, "tool_calls": [{"id": "i", "name": "recall", "arguments": "{}"}]}
    intent = StubBrain(decision)
    english = StubBrain({"content": "unused", "tool_calls": []})
    arabic = StubBrain({"content": "unused", "tool_calls": []})
    model = LocalModelEnsembleLLM(intent, english, arabic)
    assert model.chat([{"role": "user", "content": "find my old thing"}], [])["tool_calls"]
    assert not english.calls and not arabic.calls


def test_powerful_backend_builds_shared_weight_deliberative_brains(tmp_path):
    settings = load_settings({
        "JARVIS_HOME": str(tmp_path), "JARVIS_BACKEND": "powerful",
        "JARVIS_HF_REASONING_MODEL": str(tmp_path / "reasoning"),
        "JARVIS_HF_CRITIC_MODEL": str(tmp_path / "reasoning"),
        "JARVIS_HF_ENGLISH_MODEL": str(tmp_path / "reasoning"),
        "JARVIS_HF_ARABIC_MODEL": str(tmp_path / "reasoning"),
    })
    model = make_llm(settings)
    assert settings.llm_mode == "local-deliberative"
    assert isinstance(model, DeliberativeLocalLLM)
    assert {model.reasoner.model, model.critic.model, model.english.model, model.arabic.model} == {
        str(tmp_path / "reasoning")
    }
    assert all(brain._generator is None for brain in
               (model.intent, model.reasoner, model.critic, model.english, model.arabic))


def test_deliberative_pipeline_reasons_critiques_and_revises():
    intent = StubBrain({"content": "general question", "tool_calls": []})
    reasoner = StubBrain({"content": "draft answer", "tool_calls": []})
    critic = StubBrain({"content": "add missing evidence", "tool_calls": []})
    english = StubBrain({"content": "verified final answer", "tool_calls": []})
    arabic = StubBrain({"content": "جواب نهائي", "tool_calls": []})
    model = DeliberativeLocalLLM(intent, reasoner, critic, english, arabic, rounds=1)
    reply = model.chat([{"role": "user", "content": "Explain this complex idea"}], [])
    assert reply == {"content": "verified final answer", "tool_calls": []}
    assert all(brain.calls for brain in (intent, reasoner, critic, english))
    assert not arabic.calls
    revision_messages = english.calls[0][0]
    assert any("add missing evidence" in message.get("content", "") for message in revision_messages)


def test_deliberative_reasoner_can_request_more_evidence_before_writing():
    tool_reply = {"content": None, "tool_calls": [{"id": "r", "name": "recall", "arguments": "{}"}]}
    model = DeliberativeLocalLLM(
        StubBrain({"content": "ambiguous", "tool_calls": []}), StubBrain(tool_reply),
        StubBrain({"content": "unused", "tool_calls": []}),
        StubBrain({"content": "unused", "tool_calls": []}),
        StubBrain({"content": "unused", "tool_calls": []}),
    )
    assert model.chat([{"role": "user", "content": "Explain an unusual concept"}], [])["tool_calls"]
    assert not model.critic.calls and not model.english.calls and not model.arabic.calls


def test_huggingface_reply_parser_supports_tool_calls_and_plain_text():
    reply = HuggingFaceTransformersLLM._parse_reply(
        'prefix {"content":null,"tool_calls":[{"name":"recall",'
        '"arguments":{"query":"qubits"}}]} suffix'
    )
    assert reply["content"] is None
    assert reply["tool_calls"][0]["name"] == "recall"
    assert json.loads(reply["tool_calls"][0]["arguments"]) == {"query": "qubits"}
    assert HuggingFaceTransformersLLM._parse_reply("ordinary answer") == {
        "content": "ordinary answer", "tool_calls": []
    }
