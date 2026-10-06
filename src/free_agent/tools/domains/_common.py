"""Shared helpers for the domain (mode) tool packs.

Kept stdlib-only so every mode works on a fresh install without extra deps.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from free_agent import __version__

_USER_AGENT = f"free-agent/{__version__} (+https://github.com/Robert-Gomez-AI/free-agent)"
_OUTPUT_LIMIT = 12_000

# Root the code tools resolve relative paths against. Set by the agent
# builder on every assemble: the writable root when real-disk mode is on,
# otherwise None (read tools fall back to cwd, write tools refuse).
_writable_root: Path | None = None


def set_writable_root(root: Path | None) -> None:
    global _writable_root
    _writable_root = root.resolve() if root is not None else None


def writable_root() -> Path | None:
    return _writable_root


def base_dir() -> Path:
    return _writable_root or Path.cwd().resolve()


def resolve_path(path: str, *, for_write: bool = False) -> Path:
    """Resolve `path` against the base dir.

    Writes are confined to the writable root (no escape via `..`, `~` or
    absolute paths); they are refused entirely when real-disk mode is off.
    """
    if for_write and _writable_root is None:
        raise PermissionError(
            "real-disk writes are disabled — ask the user to run `/writable on` "
            "(or start free-agent with --writable) and retry."
        )
    raw = Path(path).expanduser()
    p = (raw if raw.is_absolute() else base_dir() / raw).resolve()
    if for_write:
        root = _writable_root
        assert root is not None
        if p != root and root not in p.parents:
            raise PermissionError(f"{p} is outside the writable root {root}")
    return p


def truncate(text: str, limit: int = _OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[…truncated, {len(text) - limit} more chars]"


def http_get(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = 20.0,
    headers: dict[str, str] | None = None,
) -> bytes:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data: bytes = resp.read()
        return data


def http_json(url: str, params: dict[str, Any] | None = None, **kw: Any) -> Any:
    return json.loads(http_get(url, params, **kw).decode("utf-8"))


def net_error(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"[http error {exc.code}: {exc.reason}]"
    if isinstance(exc, urllib.error.URLError):
        return f"[network error: {exc.reason}]"
    return f"[error: {type(exc).__name__}: {exc}]"


def parse_numbers(values: str) -> list[float]:
    """Parse '1, 2 3;4\\n5' (commas, whitespace, semicolons) into floats."""
    tokens = values.replace(";", " ").replace(",", " ").split()
    try:
        return [float(t) for t in tokens]
    except ValueError as exc:
        raise ValueError(f"could not parse numbers: {exc}") from exc
