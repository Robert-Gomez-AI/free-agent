from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from free_agent.agent.profile import AgentProfile

CONFIG_FILENAME = "free-agent.yaml"


def find_config(explicit: str | Path | None = None) -> Path | None:
    """Locate the agent config file.

    Order:
        1. `explicit` argument (raises if missing)
        2. `./free-agent.yaml` in the current working directory
        3. None — caller falls back to built-in defaults
    """
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"agent config not found: {p}")
        return p
    cwd = Path.cwd() / CONFIG_FILENAME
    return cwd if cwd.exists() else None


def load_profile(path: Path | None) -> AgentProfile:
    """Read and validate an agent profile from disk, or return defaults."""
    if path is None:
        return AgentProfile.default()

    raw_text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML syntax error in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be a mapping, got {type(data).__name__}")

    try:
        return AgentProfile.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"invalid agent profile at {path}:\n{exc}") from exc


def save_profile(path: Path | None, profile: AgentProfile) -> Path:
    """Serialize the profile back to YAML.

    Writes to `path`, defaulting to ./free-agent.yaml. If the target exists
    its contents are first copied to a `.bak` sibling. Comments in the
    original file are NOT preserved — the YAML is re-emitted from scratch.
    """
    target = path if path is not None else Path.cwd() / CONFIG_FILENAME

    # `exclude_none=True` drops the optional fields (system_prompt=None,
    # tools=None) so they round-trip cleanly: a subagent with tools=None
    # in memory becomes a yaml entry with no `tools:` key.
    data = profile.model_dump(exclude_none=True)

    if target.exists():
        backup = target.with_suffix(target.suffix + ".bak")
        backup.write_bytes(target.read_bytes())

    target.write_text(
        yaml.safe_dump(
            data, sort_keys=False, allow_unicode=True, default_flow_style=False
        ),
        encoding="utf-8",
    )
    return target
