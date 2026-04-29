"""Interactive flows that drive prompt_toolkit + the LLM together.

Currently houses the subagent-creation wizard (/sub new).
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.panel import Panel

from pathlib import Path

from rich.syntax import Syntax

from free_agent.agent.profile import SubAgentProfile
from free_agent.cli.console import render_error, render_info, stream_token
from free_agent.cli.context import SessionContext
from free_agent.tools import TOOLS, reload_tools, user_tools_dir

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,40}$")
_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
_ARG_RE = re.compile(r"^([a-z_][a-z0-9_]*)(?::\s*([a-zA-Z_][a-zA-Z0-9_]*))?$")

_META_PROMPT_TEMPLATE = """\
Write a system prompt for a specialist AI subagent.

Name: {name}
Public description (used for routing): {description}
User's intent for this agent: {goal}

Constraints for the prompt you produce:
- Use second person ("You are ...").
- 3 to 6 sentences total.
- Cover the agent's role, scope, input handling, and output expectations.
- Be self-contained — the subagent will only see a task description, never the\
 chat history.
- Do NOT mention specific tools by name; tool routing is configured separately.
- No preamble, no markdown fences, no commentary. Output ONLY the prompt itself.
"""


async def create_subagent_wizard(ctx: SessionContext, console: Console) -> SubAgentProfile | None:
    """Walk the user through creating a new subagent. Returns the spec or None."""
    console.print()
    render_info(
        console,
        "subagent wizard — empty input on any step cancels.",
    )

    # ── name ──────────────────────────────────────────────────────────────
    name = await _prompt(ctx, "name")
    if not name:
        return _cancel(console)
    if not _NAME_RE.match(name):
        render_error(
            console,
            f"invalid name {name!r}. use letters/digits/_/- starting with a letter.",
        )
        return None
    if any(sa.name == name for sa in ctx.profile.subagents):
        render_error(console, f"a subagent named {name!r} already exists.")
        return None

    # ── description ───────────────────────────────────────────────────────
    description = await _prompt(
        ctx, "description (one line — main agent uses this for routing)"
    )
    if not description:
        return _cancel(console)

    # ── goal for prompt synthesis ─────────────────────────────────────────
    console.print()
    render_info(
        console,
        "goal — describe the agent's purpose. the LLM will draft a system prompt:",
    )
    goal = await _prompt(ctx, "goal")
    if not goal:
        return _cancel(console)

    # ── LLM-drafted system prompt (with regenerate loop) ──────────────────
    system_prompt = await _draft_prompt_loop(ctx, console, name, description, goal)
    if system_prompt is None:
        return _cancel(console)

    # ── tool selection ────────────────────────────────────────────────────
    tools = await _ask_tools(ctx, console)
    if tools is _SENTINEL_CANCEL:
        return _cancel(console)

    return SubAgentProfile(
        name=name,
        description=description,
        system_prompt=system_prompt,
        tools=tools,  # type: ignore[arg-type]
    )


# ─── prompt synthesis ──────────────────────────────────────────────────────


async def _draft_prompt_loop(
    ctx: SessionContext, console: Console, name: str, description: str, goal: str
) -> str | None:
    while True:
        draft = await _stream_draft(ctx.chat_model, console, name, description, goal)
        if not draft.strip():
            render_error(console, "model returned an empty draft — try again.")
            return None

        choice = (
            await _prompt(
                ctx, "accept this draft? [Y]es / [r]egenerate / [n]o"
            )
        ).lower()
        if choice in ("", "y", "yes"):
            return draft
        if choice in ("r", "re", "regen", "regenerate"):
            console.print()
            render_info(console, "regenerating...")
            continue
        if choice in ("n", "no", "cancel", "q"):
            return None
        render_info(console, f"didn't recognize {choice!r} — treating as cancel.")
        return None


async def _stream_draft(
    model: BaseChatModel, console: Console, name: str, description: str, goal: str
) -> str:
    meta = _META_PROMPT_TEMPLATE.format(name=name, description=description, goal=goal)
    console.print()
    console.print(
        Panel.fit(
            "[grey50]drafting system prompt...[/]",
            border_style="bright_magenta",
            title="[tag] DRAFT [/]",
            title_align="left",
            padding=(0, 1),
        )
    )
    console.print()

    chunks: list[str] = []
    try:
        async for chunk in model.astream(meta):
            text = _extract_text(chunk)
            if text:
                stream_token(console, text)
                chunks.append(text)
    except Exception as exc:
        console.print()
        render_error(console, f"prompt generation failed: {exc}")
        return ""
    console.print()
    return "".join(chunks).strip()


def _extract_text(chunk: Any) -> str:
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


# ─── tool selection ─────────────────────────────────────────────────────────

_SENTINEL_CANCEL = object()


async def _ask_tools(ctx: SessionContext, console: Console) -> list[str] | None | object:
    available = sorted(t.name for t in TOOLS)
    console.print()
    render_info(
        console,
        f"available user tools: [bold yellow1]{', '.join(available) or '(none)'}[/]",
    )
    render_info(
        console,
        "  comma-separated names · 'inherit' to share main's tools · empty for none",
    )
    raw = await _prompt(ctx, "tools")
    raw = raw.strip()

    if raw.lower() == "inherit":
        return None  # → don't write a `tools:` key, subagent inherits
    if not raw:
        return []  # → explicitly no user tools (deepagents builtins still apply)

    requested = [t.strip() for t in raw.split(",") if t.strip()]
    unknown = [t for t in requested if t not in available]
    if unknown:
        render_error(
            console,
            f"unknown tool(s): {unknown}. available: {available}",
        )
        return _SENTINEL_CANCEL
    return requested


# ─── small input helpers ────────────────────────────────────────────────────


async def _prompt(ctx: SessionContext, label: str) -> str:
    fragment = HTML(f'<tag>▓▒░</tag> <label>{label}</label> <arrow>▶</arrow> ')
    raw = await ctx.prompt_session.prompt_async(fragment)
    return raw.strip()


def _cancel(console: Console) -> None:
    render_info(console, "wizard cancelled.")
    return None


# ─── tool wizard ────────────────────────────────────────────────────────────


_TOOL_META_PROMPT = """\
Write a Python source file that implements ONE tool for our agent system.

Strict constraints:
- Use the `@tool` decorator from `langchain_core.tools` (import it).
- Function name: `{name}`
- Function signature: `def {name}({signature}) -> str:`
- The function MUST return a string (the agent reads it as text).
- Use ONLY the Python standard library — no third-party imports.
- Pure-ish: avoid filesystem writes, network calls, or subprocess spawns unless\
 the user's intent explicitly requires them.
- The first line of the docstring is what the agent sees to decide when to call\
 the tool — make it crisp, self-contained, and action-oriented.
- Catch foreseeable errors and return a human-readable string (never raise to the agent).

Tool description (one-liner — used as docstring summary): {description}
Implementation goal (in plain English): {goal}

Output ONLY the contents of the .py file. No markdown fences, no commentary,\
 no preamble. Start with the import line.
"""


async def create_tool_wizard(ctx: SessionContext, console: Console) -> str | None:
    """Walk the user through creating a new tool. Returns the tool name or None."""
    console.print()
    render_info(console, "tool wizard — empty input on any step cancels.")
    render_info(
        console,
        "[yellow1]heads up: the LLM writes Python that runs in this process. "
        "review the draft before saving.[/]",
    )

    # ── name ──────────────────────────────────────────────────────────────
    name = await _prompt(ctx, "tool name (snake_case, must be a valid Python identifier)")
    if not name:
        return _cancel(console)
    if not _TOOL_NAME_RE.match(name):
        render_error(
            console,
            f"invalid name {name!r}. use lowercase letters / digits / underscore, "
            "starting with a letter.",
        )
        return None
    if any(t.name == name for t in TOOLS):
        render_error(console, f"a tool named {name!r} already exists.")
        return None

    # ── description ───────────────────────────────────────────────────────
    description = await _prompt(
        ctx, "description (one line — agent uses this to decide when to call it)"
    )
    if not description:
        return _cancel(console)

    # ── arguments ─────────────────────────────────────────────────────────
    console.print()
    render_info(
        console,
        "arguments: comma-separated `name:type` pairs (types are stdlib hints).",
    )
    render_info(
        console,
        "  examples: [grey70]url:str, timeout:int[/]   ·   [grey70]country:str[/]"
        "   ·   empty for no args",
    )
    args_raw = await _prompt(ctx, "arguments")
    args = _parse_args(args_raw)
    if args is None:
        render_error(
            console,
            f"could not parse arguments: {args_raw!r}. expected `name:type, ...`.",
        )
        return None

    # ── goal ──────────────────────────────────────────────────────────────
    console.print()
    render_info(
        console, "implementation goal — describe what the function should do:"
    )
    goal = await _prompt(ctx, "goal")
    if not goal:
        return _cancel(console)

    # ── LLM-drafted source (with regenerate loop) ─────────────────────────
    signature = ", ".join(_render_arg(a) for a in args) if args else ""
    source = await _draft_tool_loop(ctx, console, name, description, goal, signature)
    if source is None:
        return _cancel(console)

    if not _looks_like_tool_source(source, name):
        render_error(
            console,
            "draft doesn't look like a valid tool (missing @tool / def / docstring). "
            "regenerate or cancel.",
        )
        return None

    # ── scope: project vs global ─────────────────────────────────────────
    target_dir = await _ask_scope(ctx, console)
    if target_dir is None:
        return _cancel(console)

    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{name}.py"
    if target.exists():
        render_error(console, f"file already exists: {target}")
        return None

    target.write_text(source, encoding="utf-8")

    try:
        reload_tools()
    except Exception as exc:
        target.unlink(missing_ok=True)
        render_error(console, f"reload failed; tool file removed: {exc}")
        return None

    if not any(t.name == name for t in TOOLS):
        target.unlink(missing_ok=True)
        render_error(
            console,
            f"the file ran but no @tool named {name!r} was found in it. "
            "the file has been removed.",
        )
        return None

    render_info(
        console,
        f"tool [bold yellow1]{name}[/] registered → "
        f"[grey50]{target.relative_to(Path.cwd())}[/]",
    )
    return name


# ─── tool wizard helpers ────────────────────────────────────────────────────


async def _ask_scope(ctx: SessionContext, console: Console) -> Path | None:
    project = user_tools_dir()
    global_dir = global_tools_dir()
    console.print()
    render_info(console, "where should this tool live?")
    render_info(
        console,
        f"  [bold]P[/]roject  → [grey70]{project}[/]  "
        "[grey50](only this directory)[/]",
    )
    render_info(
        console,
        f"  [bold]G[/]lobal   → [grey70]{global_dir}[/]  "
        "[grey50](every directory)[/]",
    )
    raw = await _prompt(ctx, "scope [P/g]")
    choice = raw.lower()
    if choice in ("", "p", "project"):
        return project
    if choice in ("g", "global"):
        return global_dir
    render_error(console, f"didn't recognize {raw!r}.")
    return None


def _parse_args(raw: str) -> list[tuple[str, str]] | None:
    raw = raw.strip()
    if not raw:
        return []
    out: list[tuple[str, str]] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        m = _ARG_RE.match(piece)
        if not m:
            return None
        arg_name, arg_type = m.group(1), (m.group(2) or "str")
        out.append((arg_name, arg_type))
    return out


def _render_arg(arg: tuple[str, str]) -> str:
    name, t = arg
    return f"{name}: {t}"


async def _draft_tool_loop(
    ctx: SessionContext,
    console: Console,
    name: str,
    description: str,
    goal: str,
    signature: str,
) -> str | None:
    while True:
        source = await _stream_tool_source(
            ctx.chat_model, console, name, description, goal, signature
        )
        if not source:
            return None
        choice = (
            await _prompt(ctx, "accept this draft? [Y]es / [r]egenerate / [n]o")
        ).lower()
        if choice in ("", "y", "yes"):
            return source
        if choice in ("r", "regen", "regenerate"):
            console.print()
            render_info(console, "regenerating...")
            continue
        return None


async def _stream_tool_source(
    model,
    console: Console,
    name: str,
    description: str,
    goal: str,
    signature: str,
) -> str:
    meta = _TOOL_META_PROMPT.format(
        name=name, description=description, goal=goal, signature=signature
    )

    chunks: list[str] = []
    try:
        async for chunk in model.astream(meta):
            text = _extract_text(chunk)
            if text:
                chunks.append(text)
    except Exception as exc:
        render_error(console, f"codegen failed: {exc}")
        return ""

    source = _strip_fences("".join(chunks).strip())
    console.print()
    syntax = Syntax(
        source or "(empty draft)",
        "python",
        theme="ansi_dark",
        line_numbers=True,
        background_color="default",
    )
    from rich.panel import Panel

    console.print(
        Panel(
            syntax,
            border_style="bright_magenta",
            title="[tag] DRAFT // tool source [/]",
            title_align="left",
            padding=(0, 1),
        )
    )
    return source


def _strip_fences(text: str) -> str:
    """Remove ```python ... ``` fences if the LLM emitted them despite instructions."""
    t = text.strip()
    if t.startswith("```"):
        # Drop opening fence (and optional language tag)
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _looks_like_tool_source(source: str, name: str) -> bool:
    has_decorator = "@tool" in source
    has_def = re.search(rf"\bdef\s+{re.escape(name)}\s*\(", source) is not None
    return has_decorator and has_def


# ─── skill wizard ───────────────────────────────────────────────────────────


_SKILL_META_PROMPT = """\
Write a SKILL.md file for an agent system following Anthropic's Agent Skills format.

A skill is a progressive-disclosure playbook: the agent sees only the YAML
description initially; when it determines the skill is relevant, it reads the
full body for instructions.

Required structure:
  ---
  name: {name}
  description: <one-line description, max 1024 chars>
  ---

  # <Skill Title>

  ## When to Use
  - <bullet list of triggers / situations>

  ## Steps
  1. <numbered procedure>

  ## Output Format
  <how the agent should format its response>

You MAY add other sections (Examples, Caveats, References) if helpful.

Skill name (must match the YAML `name`): {name}
One-line description for the YAML frontmatter: {description}
User's intent for this skill: {goal}

Output ONLY the SKILL.md content (starting with `---` for the frontmatter).
No preamble, no markdown fences around the whole thing, no commentary.
"""


async def create_skill_wizard(ctx: SessionContext, console: Console) -> str | None:
    """Interactive flow for creating a new skill (SKILL.md). Returns name or None."""
    console.print()
    render_info(console, "skill wizard — empty input on any step cancels.")

    # ── name ──────────────────────────────────────────────────────────────
    name = await _prompt(ctx, "skill name (kebab-case, e.g. web-research)")
    if not name:
        return _cancel(console)
    if not SKILL_NAME_RE.match(name):
        render_error(
            console,
            f"invalid name {name!r}. use lowercase letters/digits/dash, starting with a letter.",
        )
        return None
    existing = {s.name for s in list_skills()}
    if name in existing:
        render_error(console, f"a skill named {name!r} already exists.")
        return None

    # ── description ───────────────────────────────────────────────────────
    description = await _prompt(
        ctx, "description (one line — agent uses this to decide when to apply)"
    )
    if not description:
        return _cancel(console)

    # ── goal ──────────────────────────────────────────────────────────────
    console.print()
    render_info(
        console,
        "describe the skill's intent — what should the agent do when it matches?",
    )
    goal = await _prompt(ctx, "goal")
    if not goal:
        return _cancel(console)

    # ── LLM-drafted SKILL.md (with regenerate loop) ──────────────────────
    skill_md = await _draft_skill_loop(ctx, console, name, description, goal)
    if skill_md is None:
        return _cancel(console)

    if not _looks_like_skill_md(skill_md, name):
        render_error(
            console,
            "draft doesn't look like a valid SKILL.md (missing frontmatter or wrong name).",
        )
        return None

    # ── scope: project vs global ─────────────────────────────────────────
    scope_dir = await _ask_skill_scope(ctx, console)
    if scope_dir is None:
        return _cancel(console)

    skill_dir = scope_dir / name
    if skill_dir.exists():
        render_error(console, f"directory already exists: {skill_dir}")
        return None

    skill_dir.mkdir(parents=True, exist_ok=False)
    skill_md_path = skill_dir / "SKILL.md"
    skill_md_path.write_text(skill_md, encoding="utf-8")

    render_info(
        console,
        f"skill [bold yellow1]{name}[/] written → "
        f"[grey50]{skill_md_path.relative_to(Path.cwd()) if skill_md_path.is_relative_to(Path.cwd()) else skill_md_path}[/]",
    )
    return name


async def _ask_skill_scope(ctx: SessionContext, console: Console) -> Path | None:
    project = project_skills_dir()
    global_dir = global_skills_dir()
    console.print()
    render_info(console, "where should this skill live?")
    render_info(
        console,
        f"  [bold]P[/]roject  → [grey70]{project}[/]  [grey50](only this directory)[/]",
    )
    render_info(
        console,
        f"  [bold]G[/]lobal   → [grey70]{global_dir}[/]  [grey50](every directory)[/]",
    )
    raw = await _prompt(ctx, "scope [P/g]")
    choice = raw.lower()
    if choice in ("", "p", "project"):
        return project
    if choice in ("g", "global"):
        return global_dir
    render_error(console, f"didn't recognize {raw!r}.")
    return None


async def _draft_skill_loop(
    ctx: SessionContext, console: Console, name: str, description: str, goal: str
) -> str | None:
    while True:
        draft = await _stream_skill_draft(ctx.chat_model, console, name, description, goal)
        if not draft.strip():
            render_error(console, "model returned an empty draft — try again.")
            return None

        choice = (
            await _prompt(ctx, "accept this draft? [Y]es / [r]egenerate / [n]o")
        ).lower()
        if choice in ("", "y", "yes"):
            return draft
        if choice in ("r", "regen", "regenerate"):
            console.print()
            render_info(console, "regenerating...")
            continue
        return None


async def _stream_skill_draft(
    model, console: Console, name: str, description: str, goal: str
) -> str:
    meta = _SKILL_META_PROMPT.format(name=name, description=description, goal=goal)

    console.print()
    from rich.panel import Panel as _Panel

    console.print(
        _Panel.fit(
            "[grey50]drafting SKILL.md...[/]",
            border_style="bright_magenta",
            title="[tag] DRAFT // skill [/]",
            title_align="left",
            padding=(0, 1),
        )
    )
    console.print()

    chunks: list[str] = []
    try:
        async for chunk in model.astream(meta):
            text = _extract_text(chunk)
            if text:
                stream_token(console, text)
                chunks.append(text)
    except Exception as exc:
        console.print()
        render_error(console, f"draft failed: {exc}")
        return ""
    console.print()
    return _strip_fences("".join(chunks).strip())


def _looks_like_skill_md(text: str, name: str) -> bool:
    """Sanity-check: starts with frontmatter and the YAML name matches."""
    if not text.lstrip().startswith("---"):
        return False
    # Find name: line in the frontmatter
    fm_match = re.match(r"\s*---\s*\n(.*?)\n---", text, re.DOTALL)
    if not fm_match:
        return False
    name_match = re.search(r"^\s*name\s*:\s*(\S+)", fm_match.group(1), re.MULTILINE)
    return name_match is not None and name_match.group(1).strip().strip("'\"") == name
