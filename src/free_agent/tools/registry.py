"""Mutable tool registry.

`TOOLS` is the single list every other module reads from. It's mutated
in place by `reload_tools()` so existing references stay valid after
the user creates or removes a tool through the wizard.

Discovery sources (in order, later overrides earlier on name collision):
    1. built-in tools shipped with the package
    2. global   — ~/.config/free-agent/tools/*.py
    3. workspace — <active workspace>/tools/*.py

Workspaces replaced the previous cwd-based `./free_agent_tools/` source.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

from langchain_core.tools import BaseTool

from free_agent.tools.basic import BUILTIN_TOOLS
from free_agent.workspace import active as _active_workspace

log = logging.getLogger(__name__)

# Kept for back-compat with /tool open & error messages — no longer the
# discovery source (the active workspace is).
USER_TOOLS_DIRNAME = "tools"
GLOBAL_TOOLS_PATH = Path.home() / ".config" / "free-agent" / "tools"
_USER_MODULE_PREFIX = "free_agent_user_tools"

# Mutable — read by everyone, mutated by reload_tools().
TOOLS: list[BaseTool] = []

# tool name → file it came from. None for built-ins.
_origins: dict[str, Path | None] = {}

# path → captured exception text from the last reload that failed to import it.
# Wizards read this to surface the *actual* error to the user instead of a
# generic "no @tool found" message.
_load_errors: dict[Path, str] = {}

# Tools that deepagents adds automatically via its default middleware stack
# (FilesystemMiddleware + TodoListMiddleware).
DEEPAGENTS_BUILTINS: list[tuple[str, str]] = [
    ("read_file", "read a file from the agent's virtual filesystem"),
    ("write_file", "write a file in the virtual filesystem"),
    ("ls", "list virtual filesystem entries"),
    ("edit_file", "patch a file in the virtual filesystem"),
    ("write_todos", "maintain a structured todo list for multi-step tasks"),
]


def user_tools_dir() -> Path:
    """Active workspace's tools directory (per-workspace, no longer cwd-based)."""
    ws = _active_workspace()
    if ws is None:
        # Fallback for callers that run before the workspace is bound (eg.
        # boot-time module load). Returns a non-existent placeholder; the
        # discovery walk skips missing directories.
        return GLOBAL_TOOLS_PATH.parent / "workspaces" / "_unbound" / "tools"
    return ws.tools_dir


def global_tools_dir() -> Path:
    """User-global tool directory — loaded regardless of active workspace."""
    return GLOBAL_TOOLS_PATH


def reload_tools() -> list[str]:
    """Re-discover all tools. Mutates TOOLS in place.

    Discovery order (later sources override earlier ones on name collision):
      1. Built-in   (shipped with the package)
      2. Global     (~/.config/free-agent/tools/*.py)
      3. Workspace  (<active workspace>/tools/*.py)

    Returns the names of file-sourced tools that loaded successfully.
    """
    # Re-import the package's built-in tool module so edits to basic.py
    # (e.g. tightening a schema) are picked up by /tool reload without a
    # full process restart. No-op on first load.
    import importlib

    from free_agent.tools import basic as _basic_module

    try:
        importlib.reload(_basic_module)
    except Exception as exc:
        log.warning("could not reload tools.basic: %s", exc)
    builtins = getattr(_basic_module, "BUILTIN_TOOLS", BUILTIN_TOOLS)

    by_name: dict[str, BaseTool] = {t.name: t for t in builtins}
    origins: dict[str, Path | None] = {t.name: None for t in builtins}
    _load_errors.clear()

    for source in (global_tools_dir(), user_tools_dir()):
        if not source.is_dir():
            continue
        for path in sorted(source.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                discovered = _load_tools_from_file(path)
            except Exception as exc:
                # Capture both for terse log + verbose surfacing in wizards.
                import traceback as _tb

                _load_errors[path.resolve()] = (
                    f"{type(exc).__name__}: {exc}\n"
                    + "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
                )
                log.warning("failed to load tool file %s: %s", path, exc)
                continue
            for tool in discovered:
                if tool.name in origins and origins[tool.name] is not None:
                    log.info(
                        "tool %r at %s shadows earlier definition at %s",
                        tool.name,
                        path,
                        origins[tool.name],
                    )
                by_name[tool.name] = tool
                origins[tool.name] = path

    TOOLS[:] = list(by_name.values())
    _origins.clear()
    _origins.update(origins)
    return [name for name, origin in origins.items() if origin is not None]


def last_load_error(path: Path) -> str | None:
    """Return the last import-error trace captured for `path`, if any."""
    return _load_errors.get(path.resolve())


def origin_of(name: str) -> Path | None:
    """The file a user-created tool came from. None for built-ins or unknown."""
    return _origins.get(name)


def is_user_tool(name: str) -> bool:
    return _origins.get(name) is not None


def _load_tools_from_file(path: Path) -> list[BaseTool]:
    """Import `path` and return every BaseTool defined at module level."""
    module_name = f"{_USER_MODULE_PREFIX}.{path.stem}"
    # Ensure a fresh load on reload.
    sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return []
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # so `from <name> import x` works inside the file
    spec.loader.exec_module(module)

    found: list[BaseTool] = []
    for attr in dir(module):
        if attr.startswith("_"):
            continue
        obj = getattr(module, attr)
        if isinstance(obj, BaseTool):
            found.append(obj)
    return found


# Boot-time discovery.
reload_tools()
