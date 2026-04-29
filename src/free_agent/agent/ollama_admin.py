"""Wrappers around the Ollama HTTP API for in-app model management.

We use `ollama.AsyncClient` so pulls (which can take minutes) don't block the
event loop. All errors are translated to RuntimeError with a single clear
message — callers don't need to know about ollama-package internals.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import ollama


def is_ollama_reachable(base_url: str) -> bool:
    """Quick liveness check — True if the daemon answers `list`."""
    try:
        ollama.Client(host=base_url).list()
        return True
    except Exception:
        return False


def list_models(base_url: str) -> list[dict[str, Any]]:
    """Return a list of {name, size_bytes, modified} dicts."""
    try:
        resp = ollama.Client(host=base_url).list()
    except Exception as exc:
        raise RuntimeError(_unreachable_msg(base_url, exc)) from exc

    out: list[dict[str, Any]] = []
    models = getattr(resp, "models", None) or (
        resp.get("models", []) if isinstance(resp, dict) else []
    )
    for m in models:
        name = getattr(m, "model", None) or (m.get("model") if isinstance(m, dict) else None)
        size = getattr(m, "size", None) or (m.get("size") if isinstance(m, dict) else None)
        modified = getattr(m, "modified_at", None) or (
            m.get("modified_at") if isinstance(m, dict) else None
        )
        if name:
            out.append({"name": name, "size_bytes": size or 0, "modified": modified})
    return out


async def pull_model(base_url: str, name: str) -> AsyncIterator[dict[str, Any]]:
    """Async generator yielding progress dicts from `ollama pull <name>`.

    Each dict has at least `status` (str). When downloading layers, also has
    `total` (int) and `completed` (int) in bytes. The final dict has
    `status == "success"`.
    """
    client = ollama.AsyncClient(host=base_url)
    try:
        async for chunk in await client.pull(name, stream=True):
            yield _chunk_to_dict(chunk)
    except ollama.ResponseError as exc:
        raise RuntimeError(f"pull failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(_unreachable_msg(base_url, exc)) from exc


def delete_model(base_url: str, name: str) -> None:
    """Remove a model. Raises RuntimeError on failure."""
    try:
        ollama.Client(host=base_url).delete(name)
    except ollama.ResponseError as exc:
        raise RuntimeError(f"could not remove {name!r}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(_unreachable_msg(base_url, exc)) from exc


# ─── helpers ────────────────────────────────────────────────────────────────


def _chunk_to_dict(chunk: Any) -> dict[str, Any]:
    if isinstance(chunk, dict):
        return chunk
    out: dict[str, Any] = {}
    for field in ("status", "digest", "total", "completed"):
        val = getattr(chunk, field, None)
        if val is not None:
            out[field] = val
    return out


def _unreachable_msg(base_url: str, exc: Exception) -> str:
    return (
        f"cannot reach Ollama at {base_url}. is the daemon running? "
        f"install: https://ollama.com  ·  start: `ollama serve`. ({exc})"
    )
