"""Workspaces — self-contained directories that bundle profile + tools + skills.

A workspace replaces the previous cwd-based discovery: instead of reading
`./free-agent.yaml`, `./free_agent_tools/`, and `./free_agent_skills/`
from the current working directory, the agent reads those from the
*active workspace*:

    ~/.config/free-agent/workspaces/<name>/
        free-agent.yaml          (AgentProfile, optional)
        tools/*.py               (langchain @tool definitions)
        skills/<name>/SKILL.md   (deepagents skills)

Global tools (`~/.config/free-agent/tools/`) and global skills
(`~/.config/free-agent/skills/`) remain shared across workspaces — useful
for utility tools you want everywhere.

The active workspace is persisted in `~/.config/free-agent/state.json`.
Switching = mutating that file + reloading tools/skills/profile.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

CONFIG_ROOT = Path.home() / ".config" / "free-agent"
WORKSPACES_ROOT = CONFIG_ROOT / "workspaces"
STATE_FILE = CONFIG_ROOT / "state.json"
DEFAULT_WORKSPACE = "default"
PROFILE_FILENAME = "free-agent.yaml"

_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,40}$")


@dataclass(frozen=True)
class Workspace:
    """Filesystem layout for a single workspace.

    All paths are absolute and resolved. Existence of the subdirectories
    is not guaranteed — readers should call `.is_dir()` before iterating.
    """

    name: str
    root: Path

    @property
    def config_file(self) -> Path:
        return self.root / PROFILE_FILENAME

    @property
    def tools_dir(self) -> Path:
        return self.root / "tools"

    @property
    def skills_dir(self) -> Path:
        return self.root / "skills"


# ─── discovery / state ──────────────────────────────────────────────────────


def workspaces_root() -> Path:
    return WORKSPACES_ROOT


def list_workspaces() -> list[Workspace]:
    if not WORKSPACES_ROOT.is_dir():
        return []
    out: list[Workspace] = []
    for child in sorted(WORKSPACES_ROOT.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            out.append(Workspace(name=child.name, root=child.resolve()))
    return out


def get_workspace(name: str) -> Workspace | None:
    target = WORKSPACES_ROOT / name
    return Workspace(name=name, root=target.resolve()) if target.is_dir() else None


def _read_state() -> dict:
    if not STATE_FILE.is_file():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("failed to read state file %s: %s", STATE_FILE, exc)
        return {}


def _write_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def get_active_name() -> str | None:
    return _read_state().get("active_workspace")


def set_active(name: str) -> None:
    if get_workspace(name) is None:
        raise ValueError(f"workspace {name!r} does not exist")
    state = _read_state()
    state["active_workspace"] = name
    _write_state(state)


# ─── lifecycle ──────────────────────────────────────────────────────────────


def create_workspace(name: str, *, seed_from: Workspace | None = None) -> Workspace:
    """Create a new workspace. If `seed_from` is given, its contents are copied."""
    _validate_name(name)
    target = WORKSPACES_ROOT / name
    if target.exists():
        raise ValueError(f"workspace {name!r} already exists at {target}")

    target.mkdir(parents=True)
    (target / "tools").mkdir()
    (target / "skills").mkdir()

    if seed_from is not None and seed_from.root.is_dir():
        if seed_from.config_file.is_file():
            shutil.copy2(seed_from.config_file, target / PROFILE_FILENAME)
        if seed_from.tools_dir.is_dir():
            for py in seed_from.tools_dir.glob("*.py"):
                shutil.copy2(py, target / "tools" / py.name)
        if seed_from.skills_dir.is_dir():
            for sub in seed_from.skills_dir.iterdir():
                if sub.is_dir():
                    shutil.copytree(sub, target / "skills" / sub.name, dirs_exist_ok=True)

    return Workspace(name=name, root=target.resolve())


def delete_workspace(name: str) -> None:
    """Delete a workspace. Refuses to delete the active or last workspace."""
    target = WORKSPACES_ROOT / name
    if not target.is_dir():
        raise ValueError(f"workspace {name!r} does not exist")
    if get_active_name() == name:
        raise ValueError(
            f"workspace {name!r} is active — switch with /ws use <other> first"
        )
    if len(list_workspaces()) <= 1:
        raise ValueError("cannot delete the last workspace")
    shutil.rmtree(target)


def ensure_default(*, migrate_cwd: bool = True) -> Workspace:
    """Make sure at least one workspace exists, and return the active one.

    On first run (no `~/.config/free-agent/workspaces/` directory), creates
    a `default` workspace. If `migrate_cwd` is true and the cwd contains a
    `free-agent.yaml`, `free_agent_tools/`, or `free_agent_skills/`, those
    are copied into the new default workspace so existing setups keep
    working.
    """
    workspaces = list_workspaces()

    if not workspaces:
        WORKSPACES_ROOT.mkdir(parents=True, exist_ok=True)
        target = WORKSPACES_ROOT / DEFAULT_WORKSPACE
        target.mkdir()
        (target / "tools").mkdir()
        (target / "skills").mkdir()
        ws = Workspace(name=DEFAULT_WORKSPACE, root=target.resolve())
        if migrate_cwd:
            _migrate_cwd_into(ws)
        set_active(DEFAULT_WORKSPACE)
        return ws

    active_name = get_active_name()
    active = get_workspace(active_name) if active_name else None
    if active is None:
        # State pointed nowhere — pick the first workspace alphabetically.
        active = workspaces[0]
        set_active(active.name)
    return active


def _migrate_cwd_into(ws: Workspace) -> None:
    """Best-effort copy of any cwd-based agent assets into the workspace."""
    cwd = Path.cwd()

    src_yaml = cwd / PROFILE_FILENAME
    if src_yaml.is_file() and not ws.config_file.exists():
        try:
            shutil.copy2(src_yaml, ws.config_file)
            log.info("migrated %s → %s", src_yaml, ws.config_file)
        except OSError as exc:
            log.warning("could not migrate %s: %s", src_yaml, exc)

    src_tools = cwd / "free_agent_tools"
    if src_tools.is_dir():
        for py in src_tools.glob("*.py"):
            try:
                shutil.copy2(py, ws.tools_dir / py.name)
            except OSError as exc:
                log.warning("could not migrate %s: %s", py, exc)

    src_skills = cwd / "free_agent_skills"
    if src_skills.is_dir():
        for sub in src_skills.iterdir():
            if sub.is_dir():
                try:
                    shutil.copytree(sub, ws.skills_dir / sub.name, dirs_exist_ok=True)
                except OSError as exc:
                    log.warning("could not migrate %s: %s", sub, exc)


def validate_name(name: str) -> None:
    """Raise ValueError if `name` doesn't match the workspace naming rule."""
    if not _NAME_RE.match(name):
        raise ValueError(
            f"invalid workspace name {name!r}: use lowercase letters, digits, "
            "underscore or hyphen (1-41 chars), starting with a letter."
        )


# Back-compat alias.
_validate_name = validate_name


def rename_workspace(old_name: str, new_name: str) -> Workspace:
    """Rename a workspace directory. If it's active, the state pointer follows."""
    validate_name(new_name)
    src = WORKSPACES_ROOT / old_name
    dst = WORKSPACES_ROOT / new_name
    if not src.is_dir():
        raise ValueError(f"workspace {old_name!r} does not exist")
    if dst.exists():
        raise ValueError(f"workspace {new_name!r} already exists")
    src.rename(dst)
    if get_active_name() == old_name:
        set_active(new_name)
    new_ws = Workspace(name=new_name, root=dst.resolve())
    if _ACTIVE is not None and _ACTIVE.name == old_name:
        bind_active(new_ws)
    return new_ws


# ─── module-level pointer to the live active workspace ─────────────────────
#
# Loaders (tools registry, skills registry, profile loader) read from this
# pointer rather than re-resolving via the state file every call. The CLI
# updates it whenever the user switches workspace. Default value is None
# until `bind_active()` is called at boot.

_ACTIVE: Workspace | None = None


def bind_active(ws: Workspace) -> None:
    """Set the in-process active-workspace pointer used by the loaders."""
    global _ACTIVE
    _ACTIVE = ws


def active() -> Workspace | None:
    """Return the in-process active workspace, or None if not yet bound."""
    return _ACTIVE
