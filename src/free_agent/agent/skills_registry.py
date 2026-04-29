"""Skills registry — discovers SKILL.md folders in two scopes.

Skills are deepagents' progressive-disclosure playbooks: each skill is a
directory containing a `SKILL.md` (YAML frontmatter + markdown body). The
agent sees only descriptions until it decides a skill applies, then it
reads the body.

Discovery mirrors the tools layout:
    1. ./free_agent_skills/<skill-name>/SKILL.md         (project, cwd)
    2. ~/.config/free-agent/skills/<skill-name>/SKILL.md (global)

Project takes precedence on name collision (deepagents handles this — sources
later in the list override earlier ones).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

PROJECT_SKILLS_DIRNAME = "free_agent_skills"
GLOBAL_SKILLS_PATH = Path.home() / ".config" / "free-agent" / "skills"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass(frozen=True)
class SkillInfo:
    name: str
    description: str
    path: Path           # absolute path to the SKILL.md file
    scope: str           # "project" or "global"


def project_skills_dir() -> Path:
    return Path.cwd() / PROJECT_SKILLS_DIRNAME


def global_skills_dir() -> Path:
    return GLOBAL_SKILLS_PATH


def discover_skill_sources() -> list[str]:
    """Return absolute, trailing-slash paths to skill source dirs that exist.

    Order matters for deepagents: later sources override earlier ones on
    name collision, so we list global first, project second.
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
        ("project", project_skills_dir()),
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
    for root in (project_skills_dir().resolve(), global_skills_dir().resolve() if global_skills_dir().exists() else None):
        if root is None:
            continue
        try:
            skill_dir.relative_to(root)
            return True
        except ValueError:
            continue
    return False
