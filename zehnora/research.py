"""Autonomous, API-free deep web learning.

The researcher builds a broad curriculum, searches each part, reads pages,
deduplicates them, expands promising subtopics, and stores source-grounded text
in Zehnora's SQLite memory.  It deliberately has a finite source/depth budget:
"all human knowledge" is not a finishable job, while resumable bounded passes
are useful and honest.
"""

from __future__ import annotations

import hashlib
import re
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Tuple

from . import web
from .knowledge import KnowledgeBase, normalize_text

_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_WORD_RE = re.compile(r"[\w\u0600-\u06ff]{3,}", re.UNICODE)
_EN_STOP = {
    "about", "after", "also", "and", "are", "basic", "beginners", "com", "from", "guide",
    "history", "how", "introduction", "into", "latest", "learn", "main", "overview", "the",
    "their", "this", "through", "understanding", "what", "with", "www", "your",
}
_AR_STOP = {
    "إلى", "الى", "التي", "الذي", "على", "عن", "في", "من", "ما", "ماذا", "هذا", "هذه",
    "شرح", "مقدمة", "دليل", "تعلم", "حول", "كيف", "مع", "كل", "أهم", "اهم",
}


@dataclass
class ResearchReport:
    topic: str
    queries: List[str] = field(default_factory=list)
    stored_sources: int = 0
    stored_chunks: int = 0
    skipped_duplicates: int = 0
    failed_pages: int = 0
    discovered_subtopics: List[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"deep learning pass finished for '{self.topic}'",
            f"stored {self.stored_sources} source(s) / {self.stored_chunks} chunks",
            f"searched {len(self.queries)} curriculum queries",
        ]
        if self.discovered_subtopics:
            lines.append("mapped subtopics: " + ", ".join(self.discovered_subtopics[:12]))
        if self.skipped_duplicates:
            lines.append(f"skipped {self.skipped_duplicates} duplicate source(s)")
        if self.failed_pages:
            lines.append(f"could not read {self.failed_pages} page(s)")
        lines.append("Run the command again later to refresh and extend this knowledge.")
        return "\n".join(lines)


def curriculum_queries(topic: str) -> List[str]:
    """Create broad, domain-independent study queries in the topic's language."""
    topic = topic.strip()
    if _ARABIC_RE.search(topic):
        aspects = [
            "مقدمة وأساسيات", "الفروع والمجالات", "المفاهيم والمصطلحات الرئيسية",
            "التاريخ والتطور", "المناهج والأدوات", "أمثلة وتجارب", "تطبيقات عملية",
            "مواضيع متقدمة", "أحدث الأبحاث", "الأسئلة المفتوحة والقيود",
            "مراجع تعليمية شاملة", "منهج جامعي",
        ]
    else:
        aspects = [
            "introduction fundamentals", "branches fields", "key concepts terminology",
            "history development", "methods tools", "examples experiments", "real world applications",
            "advanced topics", "current research", "open questions limitations",
            "comprehensive reference", "university curriculum syllabus",
        ]
    return [f"{topic} {aspect}" for aspect in aspects]


def canonical_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query = [(key, value) for key, value in query
             if not key.lower().startswith("utm_") and key.lower() not in {"ref", "source", "fbclid", "gclid"}]
    return urllib.parse.urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"),
                                    urllib.parse.urlencode(query), ""))


def _fingerprint(text: str) -> str:
    sample = " ".join(normalize_text(text).split())[:6000]
    return hashlib.sha256(sample.encode("utf-8")).hexdigest()


def discover_subtopics(topic: str, results: Iterable[dict], limit: int = 10) -> List[str]:
    """Extract useful recurring terms from result titles/snippets."""
    topic_words = set(_WORD_RE.findall(normalize_text(topic)))
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    for result in results:
        text = f"{result.get('title', '')} {result.get('snippet', '')}"
        for raw in _WORD_RE.findall(text):
            key = normalize_text(raw)
            if key in topic_words or key in _EN_STOP or raw in _AR_STOP or key in _AR_STOP:
                continue
            if key.isdigit() or len(key) < 3:
                continue
            counts[key] += 1
            display.setdefault(key, raw)
    recurring = [word for word, count in counts.most_common() if count >= 2]
    if not recurring:  # sparse result sets still get a few expansion candidates
        recurring = [word for word, _ in counts.most_common(limit)]
    return [display[word] for word in recurring[:limit]]


class DeepResearcher:
    """Run a bounded, resumable learning pass using ordinary public web pages."""

    def __init__(self, kb: KnowledgeBase,
                 search: Callable[[str, int], List[dict]] = web.search,
                 fetch: Callable[[str], Tuple[str, str]] = web.fetch_text) -> None:
        self.kb = kb
        self.search = search
        self.fetch = fetch

    def learn(self, topic: str, max_sources: int = 24, depth: int = 2,
              results_per_query: int = 5) -> ResearchReport:
        topic = topic.strip()
        if not topic:
            raise ValueError("topic is empty")
        max_sources = min(max(int(max_sources), 1), 100)
        depth = min(max(int(depth), 1), 4)
        results_per_query = min(max(int(results_per_query), 1), 10)
        self.kb.add_topic(topic)
        report = ResearchReport(topic=topic)
        queue = curriculum_queries(topic)
        seen_queries = set()
        seen_urls = {canonical_url(source["url"]) for source in self.kb.sources(topic)}
        seen_content = set()

        for _level in range(depth):
            if report.stored_sources >= max_sources or not queue:
                break
            current, queue = queue, []
            level_results: List[dict] = []
            # Keep each level bounded even after automatic expansion.
            for query in current[:16]:
                normalized_query = normalize_text(query)
                if normalized_query in seen_queries:
                    continue
                seen_queries.add(normalized_query)
                report.queries.append(query)
                try:
                    hits = self.search(query, limit=results_per_query)
                except Exception:
                    hits = []  # provider failures are isolated; later curriculum queries still run
                level_results.extend(hits)
                for hit in hits:
                    if report.stored_sources >= max_sources:
                        break
                    url = canonical_url(hit.get("url", ""))
                    if not url or url in seen_urls:
                        report.skipped_duplicates += 1
                        continue
                    seen_urls.add(url)
                    try:
                        title, text = self.fetch(url)
                    except Exception:
                        report.failed_pages += 1
                        continue
                    # Reject navigation stubs and almost-empty/garbage pages.
                    if len(text) < 200 or len(_WORD_RE.findall(text)) < 40:
                        report.failed_pages += 1
                        continue
                    fingerprint = _fingerprint(text)
                    if fingerprint in seen_content:
                        report.skipped_duplicates += 1
                        continue
                    seen_content.add(fingerprint)
                    chunks = self.kb.add_source(topic, url, title or hit.get("title", ""), text)
                    if chunks:
                        report.stored_sources += 1
                        report.stored_chunks += chunks
            subtopics = discover_subtopics(topic, level_results, limit=10)
            for subtopic in subtopics:
                if normalize_text(subtopic) not in {normalize_text(item) for item in report.discovered_subtopics}:
                    report.discovered_subtopics.append(subtopic)
                    queue.append(f"{topic} {subtopic}")
        return report
