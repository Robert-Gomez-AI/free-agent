"""Curated catalog of Ollama models known to support tool calling.

This list is intentionally short — it's what deepagents users actually need,
not the entire ollama.com/library (most of which are non-instruct or
non-tool-calling fine-tunes that won't work here).

`/model pull <name>` accepts any name from ollama.com/library; the catalog is
a starting point, not a hard list. To update: edit this file or `git pull`
the upstream repo. PRs welcome.

Each entry is `(name, params_b, approx_size_gb, blurb, tags)`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    params_b: float        # parameters in billions
    size_gb: float         # approximate disk footprint
    blurb: str             # one-line description
    tags: tuple[str, ...]  # for filtering: small/medium/large, recommended, etc.


RECOMMENDED_MODELS: list[CatalogEntry] = [
    # ── Qwen — excellent tool calling, the default recommendation ────────
    CatalogEntry("qwen3.5:9b",   9,  6.0, "Default pick — strong tool calling, modest VRAM",  ("recommended", "small", "qwen")),
    CatalogEntry("qwen3.5:14b", 14,  9.0, "Bigger Qwen — better reasoning, still fast",       ("medium", "qwen")),
    CatalogEntry("qwen3.5:32b", 32, 20.0, "Largest Qwen — production-grade",                  ("large", "qwen")),
    CatalogEntry("qwen2.5:7b",   7,  4.7, "Older Qwen — proven, very lightweight",            ("small", "qwen")),

    # ── Llama 3.1+ — Meta's tool-capable family ─────────────────────────
    CatalogEntry("llama3.1:8b",  8,  4.7, "Meta flagship 8B with native tool calling",        ("recommended", "small", "llama")),
    CatalogEntry("llama3.1:70b", 70, 40.0, "Llama 70B — needs serious VRAM (>=48 GB)",        ("large", "llama")),

    # ── Mistral family ────────────────────────────────────────────────────
    CatalogEntry("mistral-nemo", 12, 7.1, "Mistral 12B with 128k context window",             ("medium", "mistral", "long-context")),
    CatalogEntry("mistral-small", 22, 13.0, "Mistral 22B — strong reasoning",                 ("medium", "mistral")),

    # ── Function-calling specialists ──────────────────────────────────────
    CatalogEntry("hermes3:8b",   8,  4.7, "Hermes 3 — fine-tuned for function calling",       ("small", "tools-tuned")),
    CatalogEntry("hermes3:70b", 70, 40.0, "Hermes 3 70B — heavyweight tool specialist",       ("large", "tools-tuned")),

    # ── Cohere Command R ──────────────────────────────────────────────────
    CatalogEntry("command-r",     35, 18.0, "Cohere — strong RAG and tool routing",           ("medium", "rag")),
    CatalogEntry("command-r-plus", 104, 60.0, "Cohere 104B — flagship multilingual",          ("xlarge", "rag")),

    # ── OpenAI open weights ───────────────────────────────────────────────
    CatalogEntry("gpt-oss:20b",   20, 13.0, "OpenAI open weights 20B — excellent reasoning",  ("medium", "openai")),
    CatalogEntry("gpt-oss:120b", 120, 65.0, "OpenAI open weights 120B — top quality",         ("xlarge", "openai")),

    # ── Multilingual ──────────────────────────────────────────────────────
    CatalogEntry("glm-4.7-flash",  9,  6.0, "Zhipu GLM — strong Chinese + multilingual",      ("small",)),
]


def filter_catalog(query: str) -> list[CatalogEntry]:
    """Substring match (case-insensitive) on name, blurb, and tags."""
    q = query.strip().lower()
    if not q:
        return list(RECOMMENDED_MODELS)
    out = []
    for e in RECOMMENDED_MODELS:
        haystack = " ".join((e.name, e.blurb, *e.tags)).lower()
        if q in haystack:
            out.append(e)
    return out
