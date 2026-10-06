"""Coding tools — a Claude-Code-style toolkit over the user's real disk.

Reads resolve against the writable root (or cwd when real-disk mode is off).
Writes require real-disk mode and never escape the writable root.
"""
from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from free_agent.tools.domains._common import base_dir, resolve_path, truncate

_SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".tox", ".idea",
}

_PROJECT_MARKERS = {
    "pyproject.toml": "Python (pyproject)",
    "setup.py": "Python (setuptools)",
    "requirements.txt": "Python (pip)",
    "package.json": "JavaScript/TypeScript (npm)",
    "Cargo.toml": "Rust (cargo)",
    "go.mod": "Go (modules)",
    "pom.xml": "Java (maven)",
    "build.gradle": "JVM (gradle)",
    "Gemfile": "Ruby (bundler)",
    "composer.json": "PHP (composer)",
    "CMakeLists.txt": "C/C++ (cmake)",
    "Makefile": "make",
    "Dockerfile": "Docker",
}


def _walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in filenames:
            yield Path(dirpath) / name


@tool
def project_overview(path: str = "", max_depth: int = 3) -> str:
    """Summarize a codebase: detected stack, build/test files, and a directory tree.

    Call this FIRST when starting work on an unfamiliar repository.

    Args:
      path: Directory to inspect. Empty = project root.
      max_depth: How many directory levels to print in the tree (default 3).
    """
    try:
        root = resolve_path(path or ".")
    except PermissionError as exc:
        return f"[{exc}]"
    if not root.is_dir():
        return f"[not a directory: {root}]"

    stack = [label for marker, label in _PROJECT_MARKERS.items() if (root / marker).exists()]
    lines = [f"root: {root}", f"stack: {', '.join(stack) or 'unknown'}"]
    for doc in ("README.md", "CLAUDE.md", "AGENTS.md", "CONTRIBUTING.md"):
        if (root / doc).is_file():
            lines.append(f"doc: {doc}")

    lines.append("tree:")
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts)
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith("."))
        if depth >= max_depth:
            dirnames[:] = []
        indent = "  " * depth
        if depth:
            lines.append(f"{indent[:-2]}{rel.name}/")
        for name in sorted(filenames)[:40]:
            lines.append(f"{indent}{name}")
            count += 1
        if count > 400:
            lines.append("[…tree truncated]")
            break
    return truncate("\n".join(lines))


@tool
def view_file(path: str, offset: int = 1, limit: int = 400) -> str:
    """Read a file from disk with line numbers (like `cat -n`).

    Always view a file before editing it so `str_replace` gets an exact match.

    Args:
      path: File path, relative to the project root or absolute.
      offset: 1-based line to start from (default 1).
      limit: Max number of lines to return (default 400).
    """
    try:
        p = resolve_path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError) as exc:
        return f"[cannot read {path}: {exc}]"
    lines = text.splitlines()
    start = max(1, int(offset))
    end = min(len(lines), start - 1 + max(1, int(limit)))
    body = "\n".join(f"{i:>6}\t{lines[i - 1]}" for i in range(start, end + 1))
    footer = f"\n[lines {start}-{end} of {len(lines)}]" if end < len(lines) or start > 1 else ""
    return truncate(body + footer) if body else f"[{p} is empty]"


@tool
def glob_files(pattern: str, path: str = "") -> str:
    """Find files by glob pattern (e.g. `**/*.py`, `src/**/test_*.ts`), newest first.

    Args:
      pattern: Glob pattern relative to `path`.
      path: Directory to search from. Empty = project root.
    """
    try:
        root = resolve_path(path or ".")
    except PermissionError as exc:
        return f"[{exc}]"
    matches = [
        p for p in root.glob(pattern)
        if p.is_file() and not any(part in _SKIP_DIRS for part in p.relative_to(root).parts)
    ]
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        return f"[no files match {pattern!r} under {root}]"
    out = [str(p.relative_to(root)) for p in matches[:300]]
    if len(matches) > 300:
        out.append(f"[…{len(matches) - 300} more]")
    return "\n".join(out)


@tool
def grep_code(
    pattern: str,
    path: str = "",
    file_glob: str = "",
    ignore_case: bool = False,
    max_results: int = 200,
) -> str:
    """Search file contents with a regular expression (uses ripgrep when installed).

    Returns `file:line: text` matches. Use it to locate definitions, call sites,
    config keys or error messages before reading files.

    Args:
      pattern: Regular expression to search for.
      path: File or directory to search. Empty = project root.
      file_glob: Restrict to files matching this glob (e.g. `*.py`). Empty = all.
      ignore_case: Case-insensitive search.
      max_results: Cap on returned matches (default 200).
    """
    try:
        root = resolve_path(path or ".")
    except PermissionError as exc:
        return f"[{exc}]"
    max_results = max(1, min(int(max_results), 2000))

    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "--line-number", "--no-heading", "--color=never", "-m", "50"]
        if ignore_case:
            cmd.append("-i")
        if file_glob:
            cmd += ["--glob", file_glob]
        cmd += ["--", pattern, str(root)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return "[grep timed out]"
        if res.returncode not in (0, 1):
            return f"[rg error: {res.stderr.strip()}]"
        lines = res.stdout.splitlines()
    else:
        try:
            rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            return f"[invalid regex: {exc}]"
        files = [root] if root.is_file() else _walk(root)
        lines = []
        for f in files:
            if file_glob and not fnmatch.fnmatch(f.name, file_glob):
                continue
            try:
                with f.open(encoding="utf-8", errors="strict") as fh:
                    for n, line in enumerate(fh, 1):
                        if rx.search(line):
                            lines.append(f"{f}:{n}:{line.rstrip()}")
            except (UnicodeDecodeError, OSError):
                continue
            if len(lines) >= max_results:
                break

    base = str(base_dir()) + os.sep
    lines = [ln.replace(base, "", 1) for ln in lines]
    if not lines:
        return f"[no matches for {pattern!r}]"
    extra = len(lines) - max_results
    out = "\n".join(lines[:max_results])
    return truncate(out + (f"\n[…{extra} more matches]" if extra > 0 else ""))


@tool
def str_replace(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Edit a file by exact string replacement (requires real-disk mode).

    `old_string` must match the file exactly (including indentation) and be
    unique unless `replace_all` is true. View the file first.

    Args:
      path: File to edit.
      old_string: Exact text to replace.
      new_string: Replacement text (must differ from old_string).
      replace_all: Replace every occurrence instead of requiring uniqueness.
    """
    try:
        p = resolve_path(path, for_write=True)
        text = p.read_text(encoding="utf-8")
    except (OSError, PermissionError) as exc:
        return f"[cannot edit {path}: {exc}]"
    if old_string == new_string:
        return "[old_string and new_string are identical — nothing to do]"
    count = text.count(old_string)
    if count == 0:
        return "[old_string not found — view the file and copy the exact text]"
    if count > 1 and not replace_all:
        return f"[old_string matches {count} times — add surrounding context or set replace_all]"
    new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
    p.write_text(new_text, encoding="utf-8")
    return f"edited {p} ({count if replace_all else 1} replacement(s))"


@tool
def create_file(path: str, content: str, overwrite: bool = False) -> str:
    """Create a new file with the given content (requires real-disk mode).

    Refuses to clobber an existing file unless `overwrite` is true — prefer
    `str_replace` for changing existing files.

    Args:
      path: Destination path; parent directories are created.
      content: Full file content.
      overwrite: Allow replacing an existing file.
    """
    try:
        p = resolve_path(path, for_write=True)
    except PermissionError as exc:
        return f"[{exc}]"
    if p.exists() and not overwrite:
        return f"[{p} already exists — use str_replace, or set overwrite=true]"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {p} ({len(content.splitlines())} lines)"


_GIT_READONLY = {"status", "diff", "log", "show", "blame", "branch", "remote", "ls-files", "rev-parse", "shortlog", "tag"}


@tool
def git_inspect(subcommand: str, extra_args: str = "") -> str:
    """Run a READ-ONLY git command in the project (status, diff, log, show, blame, branch, …).

    For commits, pushes or other mutations use the `shell` tool and only when
    the user asked for it.

    Args:
      subcommand: One of status, diff, log, show, blame, branch, remote, ls-files, rev-parse, shortlog, tag.
      extra_args: Extra arguments, e.g. `--stat HEAD~3` or `-n 10 --oneline`.
    """
    sub = subcommand.strip()
    if sub not in _GIT_READONLY:
        return f"[git {sub!r} not allowed here — allowed: {', '.join(sorted(_GIT_READONLY))}]"
    import shlex

    try:
        extra = shlex.split(extra_args)
    except ValueError as exc:
        return f"[bad args: {exc}]"
    try:
        res = subprocess.run(
            ["git", "--no-pager", sub, *extra],
            capture_output=True, text=True, timeout=60, cwd=base_dir(),
        )
    except FileNotFoundError:
        return "[git is not installed]"
    except subprocess.TimeoutExpired:
        return "[git timed out]"
    out = res.stdout or ""
    if res.returncode != 0:
        out += f"\n[exit {res.returncode}] {res.stderr.strip()}"
    return truncate(out.strip() or "(no output)")


CODE_TOOLS = [project_overview, view_file, glob_files, grep_code, str_replace, create_file, git_inspect]
