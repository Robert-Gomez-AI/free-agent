"""Skills registry — discovers SKILL.md folders in two scopes.

Skills are deepagents' progressive-disclosure playbooks: each skill is a
directory containing a `SKILL.md` (YAML frontmatter + markdown body). The
agent sees only descriptions until it decides a skill applies, then it
reads the body.

Discovery sources (later overrides earlier on name collision):
    1. global    — ~/.config/free-agent/skills/<name>/SKILL.md
    2. workspace — <active workspace>/skills/<name>/SKILL.md
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from free_agent.workspace import active as _active_workspace

log = logging.getLogger(__name__)

GLOBAL_SKILLS_PATH = Path.home() / ".config" / "free-agent" / "skills"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass(frozen=True)
class SkillInfo:
    name: str
    description: str
    path: Path           # absolute path to the SKILL.md file
    scope: str           # "workspace" or "global"


def project_skills_dir() -> Path:
    """Active workspace's skills directory.

    Renamed conceptually from "project" to "workspace" — the function name
    is kept for back-compat with /skill open & friends.
    """
    ws = _active_workspace()
    if ws is None:
        return GLOBAL_SKILLS_PATH.parent / "workspaces" / "_unbound" / "skills"
    return ws.skills_dir


def global_skills_dir() -> Path:
    return GLOBAL_SKILLS_PATH


def discover_skill_sources() -> list[str]:
    """Return absolute, trailing-slash paths to skill source dirs that exist.

    Order matters for deepagents: later sources override earlier ones on
    name collision, so we list global first, workspace second.
    """
    sources: list[str] = []
    for d in (global_skills_dir(), project_skills_dir()):
        if d.is_dir():
            sources.append(str(d.resolve()) + "/")
    return sources


def list_skills() -> list[SkillInfo]:
    """Scan both folders and return parsed metadata for every SKILL.md found."""
    out: list[SkillInfo] = []
    for scope, source_dir in (
        ("global", global_skills_dir()),
        ("workspace", project_skills_dir()),
    ):
        if not source_dir.is_dir():
            continue
        for child in sorted(source_dir.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.is_file():
                continue
            try:
                meta = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("could not parse %s: %s", skill_md, exc)
                continue
            out.append(
                SkillInfo(
                    name=str(meta.get("name") or child.name),
                    description=str(meta.get("description") or "").strip(),
                    path=skill_md.resolve(),
                    scope=scope,
                )
            )
    return out


def _parse_frontmatter(text: str) -> dict[str, str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    data = yaml.safe_load(m.group(1)) or {}
    if not isinstance(data, dict):
        return {}
    return data


SKILL_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,40}$")


def is_user_skill(skill_dir: Path) -> bool:
    """True if `skill_dir` is inside one of our managed scope directories."""
    try:
        skill_dir = skill_dir.resolve()
    except OSError:
        return False
    candidates: list[Path] = []
    proj = project_skills_dir()
    if proj.exists():
        candidates.append(proj.resolve())
    glob = global_skills_dir()
    if glob.exists():
        candidates.append(glob.resolve())
    for root in candidates:
        try:
            skill_dir.relative_to(root)
            return True
        except ValueError:
            continue
    return False
