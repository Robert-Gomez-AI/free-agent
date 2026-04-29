"""Mutable tool registry.

`TOOLS` is the single list every other module reads from. It's mutated
in place by `reload_tools()` so existing references stay valid after
the user creates or removes a tool through the wizard.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

from langchain_core.tools import BaseTool

from free_agent.tools.basic import BUILTIN_TOOLS

log = logging.getLogger(__name__)

USER_TOOLS_DIRNAME = "free_agent_tools"
GLOBAL_TOOLS_PATH = Path.home() / ".config" / "free-agent" / "tools"
_USER_MODULE_PREFIX = "free_agent_user_tools"

# Mutable — read by everyone, mutated by reload_tools().
TOOLS: list[BaseTool] = []

# tool name → file it came from. None for built-ins.
_origins: dict[str, Path | None] = {}

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
    """Project-local user tool directory (relative to cwd)."""
    return Path.cwd() / USER_TOOLS_DIRNAME


def global_tools_dir() -> Path:
    """User-global tool directory — loaded regardless of cwd."""
    return GLOBAL_TOOLS_PATH


def reload_tools() -> list[str]:
    """Re-discover all tools. Mutates TOOLS in place.

    Discovery order:
      1. Built-in (shipped with the package)
      2. Global   (~/.config/free-agent/tools/*.py)
      3. Project  (./free_agent_tools/*.py)

    Later sources override earlier ones on name collision (so a project tool
    can shadow a global tool with the same name). Returns the names of
    user-sourced tools that loaded successfully.
    """
    by_name: dict[str, BaseTool] = {t.name: t for t in BUILTIN_TOOLS}
    origins: dict[str, Path | None] = {t.name: None for t in BUILTIN_TOOLS}

    for source in (global_tools_dir(), user_tools_dir()):
        if not source.is_dir():
            continue
        for path in sorted(source.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                discovered = _load_tools_from_file(path)
            except Exception as exc:
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
