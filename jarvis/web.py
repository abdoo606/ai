"""Web access: search, fetch and readable text extraction. Standard library only."""

from __future__ import annotations

import html
import ipaddress
import re
import socket
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import List, Tuple

USER_AGENT = "Mozilla/5.0 (compatible; JarvisAssistant/0.1; +local)"
MAX_BYTES = 2_000_000
TIMEOUT = 15


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in ("p", "br", "li", "h1", "h2", "h3", "h4", "div", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(page: str) -> Tuple[str, str]:
    """Return (title, readable text) from an HTML page."""
    parser = _TextExtractor()
    parser.feed(page)
    text = html.unescape("".join(parser.parts))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return parser.title.strip(), text


def validate_web_url(url: str, allow_private: bool = False) -> str:
    """Validate a URL and block credentials plus local/private network targets."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("only absolute http and https URLs are allowed")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed")
    if allow_private:
        return url
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
    except socket.gaierror as error:
        raise ValueError(f"could not resolve web host: {parsed.hostname}") from error
    if not addresses:
        raise ValueError("web host resolved to no addresses")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("private, loopback, link-local, and reserved web targets are blocked")
    return url


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, allow_private: bool = False) -> None:
        super().__init__()
        self.allow_private = allow_private

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_web_url(newurl, allow_private=self.allow_private)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _get(url: str, allow_private: bool = False) -> str:
    validate_web_url(url, allow_private=allow_private)
    # Scheme, credentials, DNS targets, and every redirect are validated above/below.
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310  # nosec B310
    opener = urllib.request.build_opener(_SafeRedirectHandler(allow_private=allow_private))
    with opener.open(req, timeout=TIMEOUT) as resp:
        content_type = (resp.headers.get_content_type() or "").lower()
        if content_type and content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
            raise ValueError(f"unsupported web content type: {content_type}")
        raw = resp.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError(f"web page exceeds the {MAX_BYTES}-byte limit")
        charset = resp.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


def fetch_text(url: str, allow_private: bool = False) -> Tuple[str, str]:
    """Download safe readable web text. Local/private addresses are blocked by default."""
    return html_to_text(_get(url, allow_private=allow_private))


class _ResultParser(HTMLParser):
    """Parses DuckDuckGo's HTML results page."""

    def __init__(self) -> None:
        super().__init__()
        self.results: List[dict] = []
        self._current: dict | None = None
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "") or ""
        if tag == "a" and "result__a" in cls:
            href = a.get("href", "")
            self._current = {"url": _clean_ddg_url(href), "title": "", "snippet": ""}
            self.results.append(self._current)
            self._in_title = True
        elif "result__snippet" in cls and self.results:
            self._in_snippet = True

    def handle_endtag(self, tag):
        if tag == "a":
            self._in_title = False
        if tag in ("a", "div", "td") and self._in_snippet:
            self._in_snippet = False

    def handle_data(self, data):
        if self._in_title and self._current is not None:
            self._current["title"] += data
        elif self._in_snippet and self.results:
            self.results[-1]["snippet"] += data


def _clean_ddg_url(href: str) -> str:
    parsed = urllib.parse.urlparse(href)
    qs = urllib.parse.parse_qs(parsed.query)
    if "uddg" in qs:
        return qs["uddg"][0]
    if href.startswith("//"):
        return "https:" + href
    return href


def parse_search_results(page: str, limit: int = 5) -> List[dict]:
    parser = _ResultParser()
    parser.feed(page)
    out = []
    for r in parser.results:
        if r["url"].startswith(("http://", "https://")):
            out.append({"url": r["url"], "title": html.unescape(r["title"]).strip(),
                        "snippet": html.unescape(r["snippet"]).strip()})
        if len(out) >= limit:
            break
    return out


def search(query: str, limit: int = 5) -> List[dict]:
    """Web search through DuckDuckGo's HTML endpoint."""
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    return parse_search_results(_get(url), limit=limit)
