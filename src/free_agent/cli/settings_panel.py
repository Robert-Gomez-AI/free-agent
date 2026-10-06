"""Settings panel — edit a YAML in your `$EDITOR`.

The numbered-menu version was clunky and the prompt_toolkit-dialog version
collided with the chat prompt. This implementation takes a different path:
write the current settings as commented YAML to a temp file, spawn the
user's editor (`$VISUAL` / `$EDITOR`, fallback to nano/vim/vi/code), and
on save parse + validate + apply. Quit without saving = cancel.

Why an editor:
  - works in any terminal (no TUI dance with the live prompt)
  - the user gets paste, undo, multi-line, syntax-highlight for free
  - all fields are visible at once and editable in any order
  - comments explain each field, no hidden state
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import SecretStr

from free_agent.config import PROVIDERS, normalize_fireworks_model

if TYPE_CHECKING:
    from rich.console import Console

    from free_agent.cli.context import SessionContext


# Curated catalog rendered into the YAML preamble so users can see the
# valid Anthropic model IDs at a glance and copy/paste into the field.
# Keep ordered newest → oldest within each tier.
ANTHROPIC_MODELS = (
    ("claude-opus-4-7",          "Opus 4.7      — strongest reasoning, slowest, priciest"),
    ("claude-sonnet-4-6",        "Sonnet 4.6    — balanced default"),
    ("claude-haiku-4-5-20251001","Haiku 4.5     — fast + cheap, weaker reasoning"),
)


# Curated Fireworks serverless models with tool calling. `/model list` shows
# the live catalog for your key; any `accounts/...` id works.
FIREWORKS_MODELS = (
    ("accounts/fireworks/models/kimi-k3",            "Kimi K3       — default, strong agentic coding, 1M ctx"),
    ("accounts/fireworks/models/glm-5p3",            "GLM 5.3       — strong reasoning + tool use, 1M ctx"),
    ("accounts/fireworks/models/deepseek-v4p1-flash","DeepSeek V4.1 Flash — fast, cheap, 1M ctx"),
    ("accounts/fireworks/models/minimax-m3",         "MiniMax M3    — long-context agent, 512k ctx"),
    ("accounts/fireworks/models/qwen3p8-max",        "Qwen 3.8 Max  — multilingual generalist"),
    ("accounts/fireworks/models/gpt-oss-120b",       "gpt-oss 120B  — open-weight OpenAI, 128k ctx"),
)


@dataclass
class _Snapshot:
    """Working copy of every field the panel can mutate."""

    workspace_name: str
    provider: str
    fireworks_model: str
    fireworks_api_key: str | None
    mode: str
    ollama_model: str
    anthropic_model: str
    anthropic_api_key: str | None  # plaintext; never logged, never written to YAML body
    temperature: float
    max_tokens: int
    writable_root: Path | None


# Sentinel rendered in the YAML when a key already exists. The user replaces
# it with a real key to overwrite, deletes the line / writes empty / "null"
# to clear, or leaves it untouched to keep the existing key.
_KEY_KEEP_SENTINEL = "<keep current>"
_KEY_PRESENT_HINT = "<keep current — set, masked>"
_KEY_ABSENT_HINT = "<unset — paste your sk-ant-… here to set>"


def _snapshot(ctx: "SessionContext") -> _Snapshot:
    key = ctx.settings.anthropic_api_key
    fw_key = ctx.settings.fireworks_api_key
    return _Snapshot(
        workspace_name=ctx.workspace.name,
        provider=ctx.settings.provider,
        fireworks_model=ctx.settings.fireworks_model,
        fireworks_api_key=fw_key.get_secret_value() if fw_key is not None else None,
        mode=ctx.settings.mode,
        ollama_model=ctx.settings.ollama_model,
        anthropic_model=ctx.settings.anthropic_model,
        anthropic_api_key=key.get_secret_value() if key is not None else None,
        temperature=ctx.settings.temperature,
        max_tokens=ctx.settings.max_tokens,
        writable_root=ctx.writable_root,
    )


def _mask(key: str | None) -> str:
    """Render a key for display: `sk-ant-…abcd` (last 4 chars), or `(unset)`."""
    if not key:
        return "(unset)"
    s = key.strip()
    if len(s) <= 8:
        return "****"
    return f"{s[:7]}…{s[-4:]}"


# ─── public entry point ─────────────────────────────────────────────────────


async def open_settings(ctx: "SessionContext", console: "Console") -> None:
    from free_agent.cli.console import render_error, render_info

    editor = _find_editor()
    if editor is None:
        render_error(
            console,
            "no editor found. set $EDITOR (e.g. `export EDITOR=nano`) or install "
            "one of: nano, vim, vi, code, notepad.",
        )
        return

    snap = _snapshot(ctx)
    initial_yaml = _render_yaml(snap, ctx)

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="free-agent-settings-",
        delete=False,
        encoding="utf-8",
    )
    tmp.write(initial_yaml)
    tmp.close()
    tmp_path = Path(tmp.name)

    try:
        render_info(
            console,
            f"opening [bold bright_cyan]{editor}[/] — save & quit to apply, "
            "leave unchanged or empty to cancel.",
        )

        try:
            rc = await asyncio.to_thread(_run_editor, editor, tmp_path)
        except FileNotFoundError as exc:
            render_error(console, f"could not launch editor: {exc}")
            return

        if rc != 0:
            render_error(console, f"editor exited with status {rc}.")
            return

        new_text = tmp_path.read_text(encoding="utf-8")
        if new_text.strip() == initial_yaml.strip():
            render_info(console, "no changes — settings unchanged.")
            return

        try:
            data = yaml.safe_load(new_text) or {}
            if not isinstance(data, dict):
                raise ValueError("top-level must be a YAML mapping")
            pending = _parse_yaml(data, snap)
        except (yaml.YAMLError, ValueError, TypeError) as exc:
            render_error(console, f"invalid yaml — settings not applied: {exc}")
            return

        await _commit(ctx, pending, snap, console)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


# ─── editor discovery + spawn ───────────────────────────────────────────────


def _find_editor() -> str | None:
    """Return the editor command (with args) or None if nothing is available."""
    for env in ("VISUAL", "EDITOR"):
        v = os.environ.get(env)
        if v and v.strip():
            return v.strip()
    candidates = ("nano", "vim", "vi", "micro", "code", "notepad")
    for candidate in candidates:
        if shutil.which(candidate):
            # `code` needs --wait so we block until the user closes the buffer.
            if candidate == "code":
                return "code --wait"
            return candidate
    return None


def _run_editor(editor: str, path: Path) -> int:
    """Block on the editor process. Stdin/stdout/stderr inherit from us."""
    cmd = editor.split() + [str(path)]
    return subprocess.call(cmd)  # noqa: S603 — args list, no shell


# ─── YAML render / parse ────────────────────────────────────────────────────


def _render_yaml(snap: _Snapshot, ctx: "SessionContext") -> str:
    """Serialize current settings as commented YAML for the editor buffer."""
    wr = "null" if snap.writable_root is None else str(snap.writable_root)
    key_placeholder = (
        _KEY_KEEP_SENTINEL if snap.anthropic_api_key else ""
    )
    key_hint = _KEY_PRESENT_HINT if snap.anthropic_api_key else _KEY_ABSENT_HINT
    key_status = (
        f"currently: {_mask(snap.anthropic_api_key)}"
        if snap.anthropic_api_key
        else "currently: (unset)"
    )

    catalog_lines = "\n".join(
        f"#   - {name:<30}  {note}" for name, note in ANTHROPIC_MODELS
    )
    fw_catalog = "\n".join(f"#   - {name:<46}  {note}" for name, note in FIREWORKS_MODELS)
    fw_placeholder = _KEY_KEEP_SENTINEL if snap.fireworks_api_key else ""
    fw_hint = _KEY_PRESENT_HINT if snap.fireworks_api_key else "<unset — paste your fw_… key here to set>"
    fw_status = f"currently: {_mask(snap.fireworks_api_key)}"

    from free_agent.modes import MODES

    mode_lines = "\n".join(f"#   - {m.name:<9} {m.title} — {m.description}" for m in MODES.values())

    return f"""\
# ── free-agent settings ────────────────────────────────────────────────────
# Edit values below. Save and quit your editor to apply.
# Quit without saving (or leave the file unchanged) to cancel.
# Lines starting with `#` are comments and ignored.
#
# Active workspace path: {ctx.workspace.root}
# Persisted to:
#   ~/.config/free-agent/settings.json   (non-secret prefs)
#   ~/.config/free-agent/secrets.json    (api keys, mode 0600)
# ───────────────────────────────────────────────────────────────────────────

# Workspace name. Renaming moves the workspace directory and updates the
# active-workspace pointer atomically. Allowed: a–z, 0–9, _, -, 1–41 chars,
# must start with a letter.
workspace: {snap.workspace_name}

# Preset mode — specialized system prompt + tool pack. `off` = plain agent.
{mode_lines}
mode: {snap.mode or "off"}

# Model provider: `fireworks` (standard — hosted open models, API key
# required), `ollama` (local) or `anthropic` (API key required).
provider: {snap.provider}

# Active Fireworks model. Short ids (`glm-5p3`) are expanded to
# `accounts/fireworks/models/<id>`. `/model list` shows the live catalog.
{fw_catalog}
fireworks_model: {snap.fireworks_model}

# Fireworks API key — used when provider == fireworks.
# {fw_status}
# Same editing rules as the Anthropic key below. A real FIREWORKS_API_KEY
# env var (or .env entry) still wins over the persisted value.
fireworks_api_key: {fw_placeholder}  # {fw_hint}

# Active Ollama model. Use `/model browse` from the chat to list pullable
# models, or `ollama pull <name>` from your shell.
ollama_model: {snap.ollama_model}

# Active Anthropic model. Curated catalog (only the ones below are tested):
{catalog_lines}
anthropic_model: {snap.anthropic_model}

# Anthropic API key — only used when provider == anthropic.
# {key_status}
#
# How to edit:
#   - Leave as `{_KEY_KEEP_SENTINEL}` (or unchanged) to keep the existing key.
#   - Paste a new key (e.g. `sk-ant-api03-…`) to overwrite.
#   - Set to empty / `null` / `""` to clear the persisted key.
#
# The key is written to ~/.config/free-agent/secrets.json with mode 0600.
# A real ANTHROPIC_API_KEY env var still wins over the persisted value.
anthropic_api_key: {key_placeholder}  # {key_hint}

# Sampling temperature in [0.0, 2.0]. 0 = deterministic, 1 = balanced, 2 = wild.
temperature: {snap.temperature}

# Hard cap on response token count. Positive integer.
max_tokens: {snap.max_tokens}

# Real-filesystem mode. Controls read_file / write_file / ls / edit_file.
#
#   `null`              → DISABLED. Tools operate on an in-memory virtual fs.
#                         Writes are sandboxed, nothing touches your disk.
#   any directory path  → ENABLED. Tools read/write the real disk under that
#                         root. Paths cannot escape via .. / ~ / absolute
#                         outside-root.
#
# Examples (uncomment ONE of the formats below to enable, or leave `null`):
#   writable_root: /home/you/projects/my-app
#   writable_root: ~/projects/my-app          # ~ is expanded
#   writable_root: {ctx.workspace.root.parent}        # any abs path works
#
# Note: the `shell` tool always runs on the real machine regardless of this
# setting — it's the escape hatch for full terminal access.
writable_root: {wr}
"""


def _parse_yaml(data: dict, snap: _Snapshot) -> _Snapshot:
    """Validate a parsed yaml dict and produce an updated _Snapshot."""
    from free_agent.workspace import validate_name

    pending = replace(snap)

    if "workspace" in data and data["workspace"] is not None:
        name = str(data["workspace"]).strip()
        if name:
            validate_name(name)
            pending.workspace_name = name

    if "provider" in data and data["provider"] is not None:
        p = str(data["provider"]).strip().lower()
        if p not in PROVIDERS:
            raise ValueError(f"unknown provider: {p!r} (use {', '.join(PROVIDERS)})")
        pending.provider = p

    if "mode" in data:
        from free_agent.modes import get_mode

        raw_mode = data["mode"]
        # YAML 1.1 parses a bare `off` / `no` as boolean False.
        m = get_mode(None if raw_mode in (None, False) else str(raw_mode))
        pending.mode = m.name if m else ""

    if "fireworks_model" in data and data["fireworks_model"] is not None:
        pending.fireworks_model = normalize_fireworks_model(str(data["fireworks_model"]))

    if "fireworks_api_key" in data:
        pending.fireworks_api_key = _parse_key(data["fireworks_api_key"], pending.fireworks_api_key)

    if "ollama_model" in data and data["ollama_model"] is not None:
        pending.ollama_model = str(data["ollama_model"]).strip()

    if "anthropic_model" in data and data["anthropic_model"] is not None:
        pending.anthropic_model = str(data["anthropic_model"]).strip()

    if "anthropic_api_key" in data:
        pending.anthropic_api_key = _parse_key(data["anthropic_api_key"], pending.anthropic_api_key)

    if "temperature" in data and data["temperature"] is not None:
        try:
            v = float(data["temperature"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"temperature must be a number: {data['temperature']!r}") from exc
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"temperature {v} out of range [0.0, 2.0]")
        pending.temperature = v

    if "max_tokens" in data and data["max_tokens"] is not None:
        try:
            v = int(data["max_tokens"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"max_tokens must be an integer: {data['max_tokens']!r}") from exc
        if v <= 0:
            raise ValueError(f"max_tokens must be positive, got {v}")
        pending.max_tokens = v

    if "writable_root" in data:
        wr = data["writable_root"]
        if wr is None or (isinstance(wr, str) and wr.strip().lower() in ("", "null", "none", "off")):
            pending.writable_root = None
        else:
            p = Path(str(wr).strip()).expanduser().resolve()
            if not p.is_dir():
                raise ValueError(f"writable_root is not an existing directory: {p}")
            pending.writable_root = p

    if pending.provider == "fireworks" and not (pending.fireworks_api_key or "").strip():
        raise ValueError(
            "provider set to 'fireworks' but no API key is configured. "
            "Paste a key into the `fireworks_api_key` field."
        )

    # Refuse the obvious foot-gun: switching to anthropic with no key set.
    if pending.provider == "anthropic" and not (pending.anthropic_api_key or "").strip():
        raise ValueError(
            "provider set to 'anthropic' but no API key is configured. "
            "Paste a key into the `anthropic_api_key` field, or keep provider as 'ollama'."
        )

    return pending


def _parse_key(raw: object, current: str | None) -> str | None:
    """API key field — three states: keep (sentinel/unchanged) / set new / clear."""
    if raw is None:
        return None  # explicit clear via `null`
    text = str(raw).strip()
    if text == _KEY_KEEP_SENTINEL or text in (_KEY_PRESENT_HINT, _KEY_ABSENT_HINT):
        return current
    # Strip wrapping quotes the user might paste from a secrets manager.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text or None


# ─── commit ─────────────────────────────────────────────────────────────────


def _restore(ctx: "SessionContext", snap: _Snapshot) -> None:
    ctx.settings.provider = snap.provider  # type: ignore[assignment]
    ctx.settings.fireworks_model = snap.fireworks_model
    ctx.settings.fireworks_api_key = (
        SecretStr(snap.fireworks_api_key) if snap.fireworks_api_key else None
    )
    ctx.settings.mode = snap.mode
    ctx.settings.ollama_model = snap.ollama_model
    ctx.settings.anthropic_model = snap.anthropic_model
    ctx.settings.anthropic_api_key = (
        SecretStr(snap.anthropic_api_key) if snap.anthropic_api_key else None
    )
    ctx.settings.temperature = snap.temperature
    ctx.settings.max_tokens = snap.max_tokens
    ctx.settings.writable = snap.writable_root is not None
    ctx.writable_root = snap.writable_root


async def _commit(
    ctx: "SessionContext",
    pending: _Snapshot,
    original: _Snapshot,
    console: "Console",
) -> None:
    from free_agent.agent.builder import make_chat_model
    from free_agent.cli.console import render_error, render_info
    from free_agent.config import save_secret_api_key, save_user_settings
    from free_agent.workspace import rename_workspace

    # 1) Workspace rename — first, so the rebuild downstream reads
    #    profile/tools/skills from the new directory.
    if pending.workspace_name != original.workspace_name:
        try:
            new_ws = rename_workspace(original.workspace_name, pending.workspace_name)
            ctx.workspace = new_ws
            ctx.config_path = (
                new_ws.config_file if new_ws.config_file.is_file() else None
            )
        except (ValueError, OSError) as exc:
            render_error(console, f"rename failed: {exc} (other settings not applied)")
            return

    # 2) Scalar settings (in-memory).
    ctx.settings.provider = pending.provider  # type: ignore[assignment]
    ctx.settings.fireworks_model = pending.fireworks_model
    ctx.settings.fireworks_api_key = (
        SecretStr(pending.fireworks_api_key) if pending.fireworks_api_key else None
    )
    ctx.settings.mode = pending.mode
    ctx.settings.ollama_model = pending.ollama_model
    ctx.settings.anthropic_model = pending.anthropic_model
    ctx.settings.anthropic_api_key = (
        SecretStr(pending.anthropic_api_key) if pending.anthropic_api_key else None
    )
    ctx.settings.temperature = pending.temperature
    ctx.settings.max_tokens = pending.max_tokens
    ctx.settings.writable = pending.writable_root is not None
    ctx.writable_root = pending.writable_root

    needs_model_rebuild = (
        pending.provider != original.provider
        or pending.fireworks_model != original.fireworks_model
        or pending.fireworks_api_key != original.fireworks_api_key
        or pending.ollama_model != original.ollama_model
        or pending.anthropic_model != original.anthropic_model
        or pending.anthropic_api_key != original.anthropic_api_key
        or pending.temperature != original.temperature
        or pending.max_tokens != original.max_tokens
    )

    try:
        if needs_model_rebuild:
            ctx.chat_model = make_chat_model(ctx.settings)
        ctx.rebuild_agent()
    except Exception as exc:
        _restore(ctx, original)
        try:
            ctx.rebuild_agent()
        except Exception:
            pass
        render_error(console, f"settings rebuild failed; reverted scalar values. ({exc})")
        return

    # 3) Persist to disk only AFTER the rebuild succeeds — never write a
    #    config the running session itself rejected.
    try:
        save_user_settings(ctx.settings)
        if pending.anthropic_api_key != original.anthropic_api_key:
            save_secret_api_key(pending.anthropic_api_key)
        if pending.fireworks_api_key != original.fireworks_api_key:
            save_secret_api_key(pending.fireworks_api_key, "fireworks_api_key")
    except OSError as exc:
        render_error(
            console,
            f"settings applied in-session but could not be persisted: {exc}",
        )
        return

    _render_diff(console, original, pending)


def _render_diff(console: "Console", before: _Snapshot, after: _Snapshot) -> None:
    """Print a one-line-per-changed-field summary so the user sees the impact."""
    from free_agent.cli.console import render_info

    fields = [
        ("workspace",        before.workspace_name,    after.workspace_name),
        ("mode",             before.mode or "off",     after.mode or "off"),
        ("provider",         before.provider,          after.provider),
        ("fireworks_model",  before.fireworks_model,   after.fireworks_model),
        ("fireworks_api_key", _mask(before.fireworks_api_key), _mask(after.fireworks_api_key)),
        ("ollama_model",     before.ollama_model,      after.ollama_model),
        ("anthropic_model",  before.anthropic_model,   after.anthropic_model),
        ("anthropic_api_key", _mask(before.anthropic_api_key), _mask(after.anthropic_api_key)),
        ("temperature",      before.temperature,       after.temperature),
        ("max_tokens",       before.max_tokens,        after.max_tokens),
        ("writable_root",    before.writable_root,     after.writable_root),
    ]
    changed = [(k, b, a) for k, b, a in fields if b != a]
    if not changed:
        render_info(console, "settings applied — no changes detected.")
        return

    lines = ["[bold bright_green]settings applied[/] —"]
    for k, b, a in changed:
        lines.append(
            f"  · [bold bright_cyan]{k}[/]: [grey50]{b}[/] → [bold]{a}[/]"
        )
    render_info(console, "\n".join(lines))
