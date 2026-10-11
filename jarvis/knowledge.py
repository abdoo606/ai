"""Fast local memory, concept graph, and source-grounded reasoning.

Jarvis combines SQLite FTS5/BM25 retrieval with a lightweight concept graph.
The graph links terms that repeatedly occur in the same evidence chunks. Query
expansion and reciprocal-rank fusion use those links to retrieve multi-hop
evidence without a remote model or a slow embedding service.
"""

from __future__ import annotations

import itertools
import re
import sqlite3
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

CHUNK_CHARS = 800
_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?؟])\s+|\n+")
_STOP_WORDS = {
    "about", "after", "also", "and", "are", "because", "been", "before", "being", "between",
    "both", "but", "can", "could", "does", "each", "for", "from", "had", "has", "have", "how",
    "into", "its", "more", "most", "not", "other", "our", "over", "such", "than", "that", "the",
    "their", "these", "they", "this", "through", "under", "use", "used", "using", "was", "were",
    "what", "when", "where", "which", "while", "will", "with", "would", "you", "your",
    "اذا", "الى", "التي", "الذي", "الذين", "على", "عليها", "عنه", "عن", "عند", "غير", "فان",
    "في", "فيها", "كان", "كما", "كل", "لا", "ما", "ماذا", "من", "منها", "هذا", "هذه", "هو",
    "هي", "يمكن", "يكون", "وهو", "وهي", "ذلك", "تلك", "بين", "بعد", "قبل", "حول",
}


@dataclass(frozen=True)
class Hit:
    topic: str
    url: str
    title: str
    text: str
    score: float


@dataclass(frozen=True)
class Connection:
    source: str
    target: str
    weight: int
    topic: str


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


def _concept_counts(text: str, limit: int = 10) -> Counter[str]:
    counts = Counter(
        token for token in _tokens(text)
        if len(token) >= 3 and token not in _STOP_WORDS and not token.isdigit()
    )
    return Counter(dict(counts.most_common(limit)))


def chunk_text(text: str, size: int = CHUNK_CHARS) -> List[str]:
    """Split text into chunks of about ``size`` characters."""
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n|\n", text) if paragraph.strip()]
    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        remaining = paragraph
        while len(remaining) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(remaining[:size])
            remaining = remaining[size:]
        if len(current) + len(remaining) + 1 > size and current:
            chunks.append(current)
            current = remaining
        else:
            current = f"{current}\n{remaining}" if current else remaining
    if current:
        chunks.append(current)
    return chunks


def _fts_query(query: str) -> str:
    words = _tokens(query)
    return " OR ".join(f'"{word}"' for word in words) if words else ""


def extractive_answer(query: str, hits: List[Hit], max_sentences: int = 5) -> str:
    """Select high-coverage, non-redundant evidence sentences with citations."""
    if not hits:
        return "nothing relevant in memory"
    query_words = set(_tokens(query))
    candidates = []
    for rank, hit in enumerate(hits):
        sentences = [sentence.strip() for sentence in _SENTENCE_RE.split(hit.text)
                     if len(sentence.strip()) >= 25]
        if not sentences and hit.text.strip():
            sentences = [hit.text.strip()]
        for position, sentence in enumerate(sentences):
            words = set(_tokens(sentence))
            coverage = len(query_words & words)
            candidates.append((coverage * 100 - rank * 3 - position * 0.01, rank, sentence, words))

    chosen = []
    covered: set[str] = set()
    seen = set()
    remaining = sorted(candidates, key=lambda item: item[0], reverse=True)
    while remaining and len(chosen) < max(1, int(max_sentences)):
        # Maximal marginal relevance: reward new query coverage and penalize
        # sentences that merely repeat already selected evidence.
        best_index, best_value = 0, float("-inf")
        for index, (base_score, rank, sentence, words) in enumerate(remaining):
            novelty = len((words & query_words) - covered)
            redundancy = max((len(words & old_words) / max(len(words | old_words), 1)
                              for _, _, old_words in chosen), default=0.0)
            value = base_score + novelty * 35 - redundancy * 30
            if value > best_value:
                best_index, best_value = index, value
        _, rank, sentence, words = remaining.pop(best_index)
        fingerprint = normalize_text(sentence)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        covered.update(words & query_words)
        chosen.append((rank, sentence, words))

    used_ranks: List[int] = []
    lines = []
    for rank, sentence, _ in chosen:
        if rank not in used_ranks:
            used_ranks.append(rank)
        lines.append(f"- {sentence} [{used_ranks.index(rank) + 1}]")
    lines.append("\nSources:")
    for number, rank in enumerate(used_ranks, 1):
        hit = hits[rank]
        lines.append(f"[{number}] {hit.title or hit.topic} — {hit.url}")
    return "\n".join(lines)


class KnowledgeBase:
    def __init__(self, path: Union[str, Path] = ":memory:") -> None:
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA temp_store = MEMORY")
        self.db.execute("PRAGMA cache_size = -32768")  # 32 MiB page cache
        if path != ":memory:":
            self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS topics(name TEXT PRIMARY KEY, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS sources(
                id INTEGER PRIMARY KEY,
                topic TEXT NOT NULL REFERENCES topics(name) ON DELETE CASCADE,
                url TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                fetched REAL NOT NULL,
                UNIQUE(topic, url)
            );
            CREATE TABLE IF NOT EXISTS concepts(
                topic TEXT NOT NULL,
                term TEXT NOT NULL,
                frequency INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(topic, term)
            );
            CREATE TABLE IF NOT EXISTS concept_relations(
                topic TEXT NOT NULL,
                left_term TEXT NOT NULL,
                right_term TEXT NOT NULL,
                weight INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(topic, left_term, right_term)
            );
            CREATE INDEX IF NOT EXISTS idx_relations_left ON concept_relations(topic, left_term, weight DESC);
            CREATE INDEX IF NOT EXISTS idx_relations_right ON concept_relations(topic, right_term, weight DESC);
            """
        )
        self._ensure_normalized_index()
        self._related_cache: Dict[tuple, List[str]] = {}

    def _ensure_normalized_index(self) -> None:
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
        self.db.execute("DELETE FROM concepts WHERE topic = ?", (name,))
        self.db.execute("DELETE FROM concept_relations WHERE topic = ?", (name,))
        cur = self.db.execute("DELETE FROM topics WHERE name = ?", (name,))
        self.db.commit()
        self._related_cache.clear()
        return cur.rowcount

    def has_source(self, topic: str, url: str) -> bool:
        return self.db.execute(
            "SELECT 1 FROM sources WHERE topic = ? AND url = ?",
            (normalize_text(topic.strip()), url),
        ).fetchone() is not None

    def _index_connections(self, topic: str, chunks: List[str]) -> None:
        concept_totals: Counter[str] = Counter()
        relation_totals: Counter[tuple] = Counter()
        for chunk in chunks:
            counts = _concept_counts(chunk)
            concept_totals.update(counts)
            for left, right in itertools.combinations(sorted(counts), 2):
                relation_totals[(left, right)] += min(counts[left], counts[right])
        self.db.executemany(
            "INSERT INTO concepts(topic, term, frequency) VALUES (?, ?, ?) "
            "ON CONFLICT(topic, term) DO UPDATE SET frequency = frequency + excluded.frequency",
            [(topic, term, frequency) for term, frequency in concept_totals.items()],
        )
        self.db.executemany(
            "INSERT INTO concept_relations(topic, left_term, right_term, weight) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(topic, left_term, right_term) DO UPDATE SET weight = weight + excluded.weight",
            [(topic, pair[0], pair[1], weight) for pair, weight in relation_totals.items()],
        )

    def add_source(self, topic: str, url: str, title: str, text: str) -> int:
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
        self._index_connections(topic, chunks)
        self.db.commit()
        self._related_cache.clear()
        return len(chunks)

    def rebuild_connections(self, topic: Optional[str] = None) -> int:
        """Rebuild graph data, useful once for databases created by older Jarvis versions."""
        normalized_topic = normalize_text(topic.strip()) if topic else None
        if normalized_topic:
            self.db.execute("DELETE FROM concepts WHERE topic = ?", (normalized_topic,))
            self.db.execute("DELETE FROM concept_relations WHERE topic = ?", (normalized_topic,))
            rows = self.db.execute("SELECT topic, text FROM chunks WHERE topic = ?", (normalized_topic,))
        else:
            self.db.execute("DELETE FROM concepts")
            self.db.execute("DELETE FROM concept_relations")
            rows = self.db.execute("SELECT topic, text FROM chunks")
        grouped: Dict[str, List[str]] = defaultdict(list)
        count = 0
        for row in rows:
            grouped[row["topic"]].append(row["text"])
            count += 1
        for name, chunks in grouped.items():
            self._index_connections(name, chunks)
        self.db.commit()
        self._related_cache.clear()
        return count

    def sources(self, topic: Optional[str] = None) -> List[dict]:
        if topic:
            rows = self.db.execute(
                "SELECT topic, url, title, fetched FROM sources WHERE topic = ? ORDER BY fetched DESC",
                (normalize_text(topic.strip()),),
            )
        else:
            rows = self.db.execute("SELECT topic, url, title, fetched FROM sources ORDER BY fetched DESC")
        return [dict(row) for row in rows]

    def search(self, query: str, topic: Optional[str] = None, limit: int = 5) -> List[Hit]:
        fts = _fts_query(query)
        if not fts:
            return []
        sql = "SELECT text, topic, url, title, bm25(chunks) AS score FROM chunks WHERE search_text MATCH ?"
        params: list = [fts]
        if topic:
            sql += " AND topic = ?"
            params.append(normalize_text(topic.strip()))
        sql += " ORDER BY score LIMIT ?"
        params.append(max(1, int(limit)))
        return [Hit(row["topic"], row["url"], row["title"], row["text"], float(row["score"]))
                for row in self.db.execute(sql, params)]

    def connections(self, query: str, topic: Optional[str] = None, limit: int = 8) -> List[Connection]:
        terms = list(dict.fromkeys(_tokens(query)))[:12]
        if not terms:
            return []
        placeholders = ",".join("?" for _ in terms)
        sql = (
            "SELECT topic, left_term, right_term, weight FROM concept_relations "
            f"WHERE (left_term IN ({placeholders}) OR right_term IN ({placeholders}))"
        )
        params: list = [*terms, *terms]
        if topic:
            sql += " AND topic = ?"
            params.append(normalize_text(topic.strip()))
        sql += " ORDER BY weight DESC LIMIT ?"
        params.append(max(1, int(limit)))
        return [Connection(row["left_term"], row["right_term"], int(row["weight"]), row["topic"])
                for row in self.db.execute(sql, params)]

    def related_terms(self, query: str, topic: Optional[str] = None, limit: int = 6) -> List[str]:
        cache_key = (normalize_text(query), normalize_text(topic or ""), int(limit))
        if cache_key in self._related_cache:
            return self._related_cache[cache_key]
        query_terms = set(_tokens(query))
        scores: Counter[str] = Counter()
        for connection in self.connections(query, topic=topic, limit=max(limit * 4, 16)):
            if connection.source in query_terms and connection.target not in query_terms:
                scores[connection.target] += connection.weight
            if connection.target in query_terms and connection.source not in query_terms:
                scores[connection.source] += connection.weight
        result = [term for term, _ in scores.most_common(max(1, int(limit)))]
        self._related_cache[cache_key] = result
        return result

    def connected_search(self, query: str, topic: Optional[str] = None, limit: int = 8) -> tuple[List[Hit], List[str]]:
        """Fuse direct BM25 and graph-expanded BM25 with reciprocal-rank fusion."""
        direct = self.search(query, topic=topic, limit=limit)
        related = self.related_terms(query, topic=topic, limit=6)
        expanded = self.search(" ".join([query, *related]), topic=topic, limit=limit * 2) if related else []
        scores: Counter[tuple] = Counter()
        hits: Dict[tuple, Hit] = {}
        for weight, ranked in ((2.0, direct), (1.0, expanded)):
            for rank, hit in enumerate(ranked, 1):
                key = (hit.topic, hit.url, hit.text)
                hits[key] = hit
                scores[key] += weight / (60 + rank)
        ordered = sorted(scores, key=scores.get, reverse=True)[:max(1, int(limit))]
        return [hits[key] for key in ordered], related

    def answer(self, query: str, topic: Optional[str] = None, limit: int = 8,
               max_sentences: int = 5) -> str:
        hits, related = self.connected_search(query, topic=topic, limit=limit)
        expanded_query = " ".join([query, *related])
        answer = extractive_answer(expanded_query, hits, max_sentences)
        if not hits:
            return answer
        links = self.connections(query, topic=topic, limit=4)
        if links:
            answer += "\n\nConnections / الروابط (co-occurrence, not causation):\n"
            answer += "\n".join(
                f"- {link.source} ↔ {link.target} (evidence weight: {link.weight})"
                for link in links
            )
        return answer

    def close(self) -> None:
        self.db.close()
