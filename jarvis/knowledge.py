"""Persistent, source-aware knowledge retrieval for Jarvis.

SQLite FTS5 keeps the default installation small and completely local.  Text
is indexed in a normalized form (including Arabic letter/diacritic
normalization), while the original text is retained for answers and citations.
"""

from __future__ import annotations

import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

CHUNK_CHARS = 800
_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?؟])\s+|\n+")


@dataclass(frozen=True)
class Hit:
    topic: str
    url: str
    title: str
    text: str
    score: float


def normalize_text(text: str) -> str:
    """Normalize text for retrieval without changing displayed source text."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _ARABIC_DIACRITICS.sub("", text)
    return text.translate(str.maketrans({
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
        "ى": "ي", "ؤ": "و", "ئ": "ي", "ـ": "",
    }))


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(normalize_text(text))


def chunk_text(text: str, size: int = CHUNK_CHARS) -> List[str]:
    """Split text into chunks of about ``size`` characters."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        p = paragraph
        while len(p) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(p[:size])
            p = p[size:]
        if len(current) + len(p) + 1 > size and current:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n{p}" if current else p
    if current:
        chunks.append(current)
    return chunks


def _fts_query(query: str) -> str:
    """Turn free text into a safe, normalized FTS5 OR query."""
    words = _tokens(query)
    return " OR ".join(f'"{word}"' for word in words) if words else ""


def extractive_answer(query: str, hits: List[Hit], max_sentences: int = 5) -> str:
    """Build a concise answer from relevant source sentences, with citations."""
    if not hits:
        return "nothing relevant in memory"

    query_words = set(_tokens(query))
    candidates = []
    for rank, hit in enumerate(hits):
        sentences = [s.strip() for s in _SENTENCE_RE.split(hit.text) if len(s.strip()) >= 25]
        if not sentences and hit.text.strip():
            sentences = [hit.text.strip()]
        for position, sentence in enumerate(sentences):
            words = set(_tokens(sentence))
            overlap = len(query_words & words)
            # BM25 has already selected/ranked chunks. Token overlap chooses the
            # clearest sentences inside those chunks; rank is a stable fallback.
            score = overlap * 100 - rank * 3 - position * 0.01
            candidates.append((score, rank, sentence))

    chosen = []
    seen = set()
    for _, rank, sentence in sorted(candidates, key=lambda item: item[0], reverse=True):
        fingerprint = normalize_text(sentence)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        chosen.append((rank, sentence))
        if len(chosen) >= max(1, int(max_sentences)):
            break

    used_ranks = []
    lines = []
    for rank, sentence in chosen:
        if rank not in used_ranks:
            used_ranks.append(rank)
        citation = used_ranks.index(rank) + 1
        lines.append(f"- {sentence} [{citation}]")
    lines.append("\nSources:")
    for number, rank in enumerate(used_ranks, 1):
        hit = hits[rank]
        label = hit.title or hit.topic
        lines.append(f"[{number}] {label} — {hit.url}")
    return "\n".join(lines)


class KnowledgeBase:
    def __init__(self, path: Union[str, Path] = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS topics(
                name TEXT PRIMARY KEY,
                created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sources(
                id INTEGER PRIMARY KEY,
                topic TEXT NOT NULL REFERENCES topics(name) ON DELETE CASCADE,
                url TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                fetched REAL NOT NULL,
                UNIQUE(topic, url)
            );
            """
        )
        self._ensure_normalized_index()

    def _ensure_normalized_index(self) -> None:
        """Create the current FTS schema and migrate the first Jarvis schema."""
        columns = [row["name"] for row in self.db.execute("PRAGMA table_info(chunks)")]
        if not columns:
            self.db.execute(
                "CREATE VIRTUAL TABLE chunks USING fts5("
                "text UNINDEXED, search_text, topic UNINDEXED, url UNINDEXED, title UNINDEXED)"
            )
            return
        if "search_text" in columns:
            return

        old_rows = list(self.db.execute("SELECT text, topic, url, title FROM chunks"))
        self.db.execute("DROP TABLE chunks")
        self.db.execute(
            "CREATE VIRTUAL TABLE chunks USING fts5("
            "text UNINDEXED, search_text, topic UNINDEXED, url UNINDEXED, title UNINDEXED)"
        )
        self.db.executemany(
            "INSERT INTO chunks(text, search_text, topic, url, title) VALUES (?, ?, ?, ?, ?)",
            [(row["text"], normalize_text(row["text"]), row["topic"], row["url"], row["title"])
             for row in old_rows],
        )
        self.db.commit()

    def add_topic(self, name: str) -> None:
        name = normalize_text(name.strip())
        if not name:
            raise ValueError("topic name is empty")
        self.db.execute("INSERT OR IGNORE INTO topics(name, created) VALUES (?, ?)", (name, time.time()))
        self.db.commit()

    def topics(self) -> List[str]:
        return [row["name"] for row in self.db.execute("SELECT name FROM topics ORDER BY name")]

    def forget_topic(self, name: str) -> int:
        name = normalize_text(name.strip())
        self.db.execute("DELETE FROM chunks WHERE topic = ?", (name,))
        self.db.execute("DELETE FROM sources WHERE topic = ?", (name,))
        cur = self.db.execute("DELETE FROM topics WHERE name = ?", (name,))
        self.db.commit()
        return cur.rowcount

    def has_source(self, topic: str, url: str) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM sources WHERE topic = ? AND url = ?",
            (normalize_text(topic.strip()), url),
        ).fetchone()
        return row is not None

    def add_source(self, topic: str, url: str, title: str, text: str) -> int:
        """Store a document under a topic. Return the number of chunks added."""
        topic = normalize_text(topic.strip())
        self.add_topic(topic)
        if self.has_source(topic, url):
            return 0
        chunks = chunk_text(text)
        if not chunks:
            return 0
        self.db.execute(
            "INSERT INTO sources(topic, url, title, fetched) VALUES (?, ?, ?, ?)",
            (topic, url, title, time.time()),
        )
        self.db.executemany(
            "INSERT INTO chunks(text, search_text, topic, url, title) VALUES (?, ?, ?, ?, ?)",
            [(chunk, normalize_text(chunk), topic, url, title) for chunk in chunks],
        )
        self.db.commit()
        return len(chunks)

    def sources(self, topic: Optional[str] = None) -> List[dict]:
        if topic:
            rows = self.db.execute(
                "SELECT topic, url, title, fetched FROM sources WHERE topic = ? ORDER BY fetched DESC",
                (normalize_text(topic.strip()),),
            )
        else:
            rows = self.db.execute(
                "SELECT topic, url, title, fetched FROM sources ORDER BY fetched DESC"
            )
        return [dict(row) for row in rows]

    def search(self, query: str, topic: Optional[str] = None, limit: int = 5) -> List[Hit]:
        fts = _fts_query(query)
        if not fts:
            return []
        sql = (
            "SELECT text, topic, url, title, bm25(chunks) AS score "
            "FROM chunks WHERE search_text MATCH ?"
        )
        params: list = [fts]
        if topic:
            sql += " AND topic = ?"
            params.append(normalize_text(topic.strip()))
        sql += " ORDER BY score LIMIT ?"
        params.append(max(1, int(limit)))
        return [
            Hit(row["topic"], row["url"], row["title"], row["text"], float(row["score"]))
            for row in self.db.execute(sql, params)
        ]

    def answer(self, query: str, topic: Optional[str] = None, limit: int = 8,
               max_sentences: int = 5) -> str:
        return extractive_answer(query, self.search(query, topic=topic, limit=limit), max_sentences)

    def close(self) -> None:
        self.db.close()
