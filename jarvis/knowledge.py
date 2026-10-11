"""Knowledge base: topics, sources and searchable text chunks.

Stored in SQLite with the FTS5 full-text index (built into Python's sqlite3,
so no extra packages). Search ranks with BM25. Every chunk keeps its source
URL, so answers can cite where they came from.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

CHUNK_CHARS = 800


@dataclass(frozen=True)
class Hit:
    topic: str
    url: str
    title: str
    text: str
    score: float


def chunk_text(text: str, size: int = CHUNK_CHARS) -> List[str]:
    """Split text into chunks of about ``size`` characters, on paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""
    for p in paragraphs:
        while len(p) > size:  # very long paragraph: hard split
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
    """Turn free text into a safe FTS5 query: each word quoted, joined with OR."""
    words = re.findall(r"[\w\u0600-\u06FF]+", query.lower())
    return " OR ".join(f'"{w}"' for w in words) if words else ""


class KnowledgeBase:
    def __init__(self, path: Union[str, Path] = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
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
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                text,
                topic UNINDEXED,
                url UNINDEXED,
                title UNINDEXED
            );
            """
        )

    # --- topics -------------------------------------------------------------
    def add_topic(self, name: str) -> None:
        name = name.strip().lower()
        if not name:
            raise ValueError("topic name is empty")
        self.db.execute("INSERT OR IGNORE INTO topics(name, created) VALUES (?, ?)", (name, time.time()))
        self.db.commit()

    def topics(self) -> List[str]:
        return [r["name"] for r in self.db.execute("SELECT name FROM topics ORDER BY name")]

    def forget_topic(self, name: str) -> int:
        name = name.strip().lower()
        self.db.execute("DELETE FROM chunks WHERE topic = ?", (name,))
        cur = self.db.execute("DELETE FROM topics WHERE name = ?", (name,))
        self.db.execute("DELETE FROM sources WHERE topic = ?", (name,))
        self.db.commit()
        return cur.rowcount

    def has_source(self, topic: str, url: str) -> bool:
        row = self.db.execute("SELECT 1 FROM sources WHERE topic = ? AND url = ?",
                              (topic.strip().lower(), url)).fetchone()
        return row is not None

    # --- sources and chunks --------------------------------------------------
    def add_source(self, topic: str, url: str, title: str, text: str) -> int:
        """Store a document under a topic. Returns the number of chunks added."""
        topic = topic.strip().lower()
        self.add_topic(topic)
        if self.has_source(topic, url):
            return 0
        chunks = chunk_text(text)
        if not chunks:
            return 0
        self.db.execute("INSERT INTO sources(topic, url, title, fetched) VALUES (?, ?, ?, ?)",
                        (topic, url, title, time.time()))
        for c in chunks:
            self.db.execute("INSERT INTO chunks(text, topic, url, title) VALUES (?, ?, ?, ?)",
                            (c, topic, url, title))
        self.db.commit()
        return len(chunks)

    def sources(self, topic: Optional[str] = None) -> List[dict]:
        if topic:
            rows = self.db.execute("SELECT topic, url, title, fetched FROM sources WHERE topic = ?",
                                   (topic.strip().lower(),))
        else:
            rows = self.db.execute("SELECT topic, url, title, fetched FROM sources")
        return [dict(r) for r in rows]

    def search(self, query: str, topic: Optional[str] = None, limit: int = 5) -> List[Hit]:
        fts = _fts_query(query)
        if not fts:
            return []
        sql = "SELECT text, topic, url, title, bm25(chunks) AS score FROM chunks WHERE chunks MATCH ?"
        params: list = [fts]
        if topic:
            sql += " AND topic = ?"
            params.append(topic.strip().lower())
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        return [Hit(topic=r["topic"], url=r["url"], title=r["title"], text=r["text"],
                    score=float(r["score"])) for r in self.db.execute(sql, params)]

    def close(self) -> None:
        self.db.close()
