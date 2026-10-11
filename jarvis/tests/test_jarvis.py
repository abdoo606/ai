"""Offline tests for Jarvis. No network, no API key, no real model.

Run: pytest -q jarvis/tests
"""

import json
from pathlib import Path

import pytest

from jarvis import web
from jarvis.agent import Assistant
from jarvis.config import load_settings
from jarvis.knowledge import KnowledgeBase, chunk_text
from jarvis.llm import LLMError, RuleBasedLLM
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

    def fetch(url):
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


# ---------- tools ----------------------------------------------------------

def test_registry_has_expected_tools(kb, settings):
    names = set(build_registry(kb, settings).tools)
    assert {"get_time", "web_search", "fetch_url", "learn_topic", "recall",
            "list_topics", "forget_topic", "read_file", "write_file", "list_files"} <= names
    assert "run_command" not in names  # shell is off by default


def test_shell_only_when_enabled(kb, tmp_path):
    on = load_settings({"JARVIS_HOME": str(tmp_path), "JARVIS_WORKSPACE": str(tmp_path / "w"),
                        "JARVIS_ALLOW_SHELL": "1"})
    assert "run_command" in build_registry(kb, on).tools


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
    assert "https://a.example/qubits" in answer
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
    ("learn quantum computing", ("learn_topic", {"topic": "quantum computing"})),
    ("تعلم الحوسبة الكمومية", ("learn_topic", {"topic": "الحوسبة الكمومية"})),
    ("recall what is a qubit", ("recall", {"query": "what is a qubit"})),
    ("ماذا تعرف عن الكيوبت", ("recall", {"query": "الكيوبت"})),
    ("what time is it", ("get_time", {})),
    ("ما هي الساعة", ("get_time", {})),
    ("topics", ("list_topics", {})),
    ("search python sqlite", ("web_search", {"query": "python sqlite"})),
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
    assert "learned" in bot.ask("learn quantum")
    assert "superposition" in bot.ask("recall superposition")
    kb.close()


def test_settings_default_to_offline_mode(tmp_path):
    s = load_settings({"JARVIS_HOME": str(tmp_path)})
    assert s.llm_mode == "offline-rules" and s.allow_shell is False
    assert load_settings({"JARVIS_LLM_API_KEY": "k"}).llm_mode == "openai-compatible"
