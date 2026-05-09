from __future__ import annotations

import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool


@tool
def current_time(timezone: str = "UTC") -> str:
    """Return the current date and time in the given IANA timezone (e.g. 'UTC', 'America/Bogota').

    Use this whenever the user asks about the current time, today's date, or anything
    time-relative ('how long until X', 'what day is it'). Do not guess.
    """
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        return f"Unknown timezone: {timezone!r}. Try an IANA name like 'UTC' or 'America/Bogota'."
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


_SHELL_OUTPUT_LIMIT = 8_000  # bytes per stream (stdout, stderr) before truncation
_SHELL_TIMEOUT_MAX = 600     # hard upper bound (seconds) per call


@tool
def shell(command: str, timeout: int = 60, cwd: str = "") -> str:
    """Run a shell command on the user's real machine and return its output.

    This is the agent's escape hatch to the actual terminal: use it for git,
    ls, find, grep, cat, python, npm, builds, tests, anything CLI. Pipes,
    redirects, `&&`, and `$VAR` substitution all work because the command is
    handed to `bash -lc`.

    Prefer this over the virtual-filesystem tools (read_file/ls/...) when the
    user wants you to touch their actual disk.

    Args:
      command: The exact command line to run (e.g. `git status` or
               `grep -rn "TODO" src | head`).
      timeout: Hard timeout in seconds. Default 60, capped at 600.
      cwd:     Working directory. Empty string (default) means the agent's
               launch directory. Avoid `Optional[str]` — gpt-oss-style chat
               templates choke on schemas with empty type arrays.

    Returns a single string with the exit code, stdout, and stderr (each
    capped at ~8 KB; longer streams are truncated with a marker).
    """
    timeout = max(1, min(int(timeout), _SHELL_TIMEOUT_MAX))
    try:
        result = subprocess.run(  # noqa: S603 — bash -lc is the intended shell entry
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or None,
        )
    except subprocess.TimeoutExpired as exc:
        out = _truncate(exc.stdout if isinstance(exc.stdout, str) else "")
        err = _truncate(exc.stderr if isinstance(exc.stderr, str) else "")
        return f"[timed out after {timeout}s]\nstdout:\n{out}\nstderr:\n{err}".rstrip()
    except FileNotFoundError:
        return "[shell unavailable: /bin/bash not found on this system]"
    except OSError as exc:
        return f"[shell error: {exc}]"

    parts = [f"exit={result.returncode}"]
    out = _truncate(result.stdout or "")
    err = _truncate(result.stderr or "")
    if out:
        parts.append(f"stdout:\n{out}")
    if err:
        parts.append(f"stderr:\n{err}")
    return "\n".join(parts)


def _truncate(stream: str) -> str:
    if len(stream) <= _SHELL_OUTPUT_LIMIT:
        return stream
    extra = len(stream) - _SHELL_OUTPUT_LIMIT
    return stream[:_SHELL_OUTPUT_LIMIT] + f"\n[…truncated, {extra} more bytes]"


# Authoritative list of tools shipped with the package.
# User tools (created via /tool new) are merged on top by tools.registry at runtime.
BUILTIN_TOOLS = [current_time, shell]
