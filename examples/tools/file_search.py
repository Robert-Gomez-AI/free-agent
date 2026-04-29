"""Sample tool: grep-like search restricted to a project directory.

To enable, copy this file into ./free_agent_tools/.

The path argument is resolved against the current working directory and any
attempt to escape it (via `..` or absolute paths outside cwd) is rejected.
Adjust _ROOT below if you want a different sandbox boundary.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

_ROOT = Path.cwd().resolve()
_MAX_HITS = 50
_MAX_BYTES_PER_FILE = 200_000


@tool
def file_search(pattern: str, glob: str = "**/*") -> str:
    """Search files under the working directory for `pattern` (plain substring, case-sensitive).

    Use this when the user asks to locate text in their codebase or notes.
    `glob` narrows the file set (e.g. `**/*.py`, `**/*.md`). Returns up to
    50 hits as `path:line: text` lines. Never raises — errors come back as text.
    """
    try:
        results: list[str] = []
        for path in _ROOT.glob(glob):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if _ROOT not in resolved.parents and resolved != _ROOT:
                continue
            try:
                if path.stat().st_size > _MAX_BYTES_PER_FILE:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for n, line in enumerate(text.splitlines(), 1):
                if pattern in line:
                    rel = path.relative_to(_ROOT)
                    results.append(f"{rel}:{n}: {line.strip()[:160]}")
                    if len(results) >= _MAX_HITS:
                        return "\n".join(results) + f"\n[truncated at {_MAX_HITS} hits]"
        return "\n".join(results) if results else "(no matches)"
    except Exception as exc:
        return f"file_search failed: {exc}"
