"""Scrape ollama.com/library for the live catalog of pullable models.

Cached on disk so a `/model browse` doesn't hit the network on every call.
The page returns ~200 models with structured `x-test-*` attributes that
make extraction stable enough for a CLI helper.

If the scrape fails (no network, page redesign, etc.) callers should fall
back to `agent.ollama_catalog.RECOMMENDED_MODELS`.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path

LIBRARY_URL = "https://ollama.com/library"
CACHE_DIR = Path.home() / ".cache" / "free-agent"
CACHE_FILE = CACHE_DIR / "library.json"
DEFAULT_TTL_SECONDS = 6 * 60 * 60  # 6 hours

_CARD_RE = re.compile(
    r'<a\s+href="/library/([a-z0-9._-]+)"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_DESC_RE = re.compile(
    r'<p[^>]*class="[^"]*max-w-lg[^"]*"[^>]*>([^<]+)</p>',
    re.DOTALL,
)
_CAP_RE = re.compile(r'x-test-capability[^>]*>([^<]+)<')
_SIZE_RE = re.compile(r'x-test-size[^>]*>([^<]+)<')


@dataclass(frozen=True)
class LibraryEntry:
    name: str
    description: str
    capabilities: tuple[str, ...]
    sizes: tuple[str, ...]

    @property
    def supports_tools(self) -> bool:
        return "tools" in self.capabilities


def fetch_library(*, timeout: float = 10.0) -> list[LibraryEntry]:
    """Fetch and parse ollama.com/library. Raises RuntimeError on network failure."""
    req = urllib.request.Request(
        LIBRARY_URL,
        headers={"User-Agent": "free-agent/0.1 (+https://github.com/Robert-Gomez-AI/free-agent)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} fetching ollama.com/library") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach ollama.com: {exc.reason}") from exc
    except TimeoutError:
        raise RuntimeError(f"ollama.com/library timed out after {timeout}s") from None
    return _parse(html)


def cached_library(
    *,
    max_age_seconds: float = DEFAULT_TTL_SECONDS,
    force_refresh: bool = False,
) -> tuple[list[LibraryEntry], str]:
    """Return (entries, source_label).

    `source_label` is one of: "fresh", "cached:<age>", "stale-fallback:<age>".
    On total failure (no network and no cache), raises RuntimeError.
    """
    if not force_refresh and CACHE_FILE.exists():
        age = time.time() - CACHE_FILE.stat().st_mtime
        if age < max_age_seconds:
            try:
                return _load_cache(), f"cached:{_humanize_age(age)}"
            except (OSError, json.JSONDecodeError, TypeError, KeyError):
                pass

    try:
        entries = fetch_library()
    except RuntimeError:
        if CACHE_FILE.exists():
            try:
                age = time.time() - CACHE_FILE.stat().st_mtime
                return _load_cache(), f"stale-fallback:{_humanize_age(age)}"
            except (OSError, json.JSONDecodeError, TypeError, KeyError):
                pass
        raise

    _save_cache(entries)
    return entries, "fresh"


# ─── parsing ────────────────────────────────────────────────────────────────


def _parse(html: str) -> list[LibraryEntry]:
    entries: list[LibraryEntry] = []
    seen: set[str] = set()
    for m in _CARD_RE.finditer(html):
        name = m.group(1)
        if name in seen:
            continue
        body = m.group(2)

        desc_m = _DESC_RE.search(body)
        if not desc_m:
            # Skip non-card matches (the regex can hit nav / unrelated anchors).
            continue
        desc = re.sub(r"\s+", " ", unescape(desc_m.group(1)).strip())

        caps = tuple(c.strip() for c in _CAP_RE.findall(body))
        sizes = tuple(s.strip() for s in _SIZE_RE.findall(body))

        entries.append(
            LibraryEntry(
                name=name, description=desc, capabilities=caps, sizes=sizes
            )
        )
        seen.add(name)
    return entries


# ─── cache I/O ──────────────────────────────────────────────────────────────


def _save_cache(entries: list[LibraryEntry]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = [
        {**asdict(e), "capabilities": list(e.capabilities), "sizes": list(e.sizes)}
        for e in entries
    ]
    CACHE_FILE.write_text(json.dumps(data), encoding="utf-8")


def _load_cache() -> list[LibraryEntry]:
    raw = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return [
        LibraryEntry(
            name=d["name"],
            description=d["description"],
            capabilities=tuple(d.get("capabilities") or []),
            sizes=tuple(d.get("sizes") or []),
        )
        for d in raw
    ]


def _humanize_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    return f"{seconds / 3600:.1f}h"


# ─── filtering ──────────────────────────────────────────────────────────────


def filter_library(entries: list[LibraryEntry], query: str) -> list[LibraryEntry]:
    q = query.strip().lower()
    if not q:
        return entries
    out = []
    for e in entries:
        haystack = " ".join(
            (e.name, e.description, *e.capabilities, *e.sizes)
        ).lower()
        if q in haystack:
            out.append(e)
    return out
