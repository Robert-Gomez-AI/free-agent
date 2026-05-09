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
from free_agent.agent.skills_registry import (
    SKILL_NAME_RE,
    global_skills_dir,
    list_skills,
    project_skills_dir,
)
from free_agent.cli.console import render_error, render_info, stream_token
from free_agent.cli.context import SessionContext
from free_agent.tools import (
    TOOLS,
    global_tools_dir,
    last_load_error,
    reload_tools,
    user_tools_dir,
)

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

    text, _count, _samples = await _stream_with_fallback(model, meta, console)
    console.print()
    return text.strip()


async def _stream_with_fallback(
    model: BaseChatModel,
    prompt: str,
    console: Console,
    *,
    show_streaming_status: bool = True,
) -> tuple[str, int, list[str]]:
    """Stream a model response, with backend-specific paths.

    Returns ``(full_text, chunk_count, sample_chunk_reprs)``.

    For ChatOllama models we go straight to the Ollama HTTP API with
    ``think=False`` so reasoning-model wizards don't burn the entire token
    budget on hidden ``<think>`` tokens. The langchain-ollama wrapper
    silently drops thinking content, which left earlier versions emitting
    thousands of empty chunks.

    For other backends (Anthropic, etc.) we use the standard langchain
    streaming with a non-streaming fallback and an extended-budget retry.
    """
    # ── Ollama: native API path, think disabled from the start ──────────
    if _is_ollama(model):
        return await _ollama_pipeline(model, prompt, console, show_streaming_status)

    chunks: list[str] = []
    raw_chunk_count = 0
    sample_chunks: list[str] = []

    try:
        async for chunk in model.astream(prompt):
            raw_chunk_count += 1
            if len(sample_chunks) < 2:
                try:
                    sample_chunks.append(repr(chunk)[:400])
                except Exception:
                    sample_chunks.append("<repr-failed>")
            text = _extract_text(chunk)
            if text:
                stream_token(console, text)
                chunks.append(text)
    except Exception as exc:
        console.print()
        render_error(console, f"streaming failed: {exc}")
        return "", raw_chunk_count, sample_chunks

    raw = "".join(chunks)
    if raw.strip():
        return raw, raw_chunk_count, sample_chunks

    # ── Fallback 2: non-streaming ainvoke ─────────────────────────────────
    if show_streaming_status:
        render_info(
            console,
            f"streaming returned empty across {raw_chunk_count} chunk(s) — "
            "retrying without streaming...",
        )
    try:
        result = await model.ainvoke(prompt)
    except Exception as exc:
        render_error(console, f"non-streaming fallback failed: {exc}")
        result = None

    if result is not None:
        full = _extract_text(result)
        if full:
            console.print()
            stream_token(console, full)
            console.print()
            return full, raw_chunk_count, sample_chunks

    # ── Fallback 3: extended budget, deterministic ────────────────────────
    # Reasoning / thinking models often blow the default num_predict on
    # internal `<think>` tokens, leaving zero budget for the visible answer.
    # We can't use .bind(num_predict=...) — `langchain-ollama` reads it from
    # the constructor, not per-call kwargs. So mutate-and-restore on the
    # live model object instead. Works for both Ollama (num_predict) and
    # Anthropic (max_tokens).
    if show_streaming_status:
        render_info(
            console,
            "still empty — retrying once more with extended token budget "
            "(fix for reasoning models)...",
        )

    saved: dict[str, Any] = {}
    for attr in ("num_predict", "max_tokens", "temperature"):
        if hasattr(model, attr):
            try:
                saved[attr] = getattr(model, attr)
            except Exception:
                pass

    def _safe_set(obj: Any, attr: str, value: Any) -> None:
        try:
            setattr(obj, attr, value)
        except Exception:
            pass

    _safe_set(model, "num_predict", 16384)
    _safe_set(model, "max_tokens", 16384)
    _safe_set(model, "temperature", 0.0)

    try:
        result2 = await model.ainvoke(prompt)
    except Exception as exc:
        render_error(console, f"extended-budget fallback failed: {exc}")
        result2 = None
    finally:
        for k, v in saved.items():
            _safe_set(model, k, v)

    if result2 is not None:
        full2 = _extract_text(result2)
        if full2:
            console.print()
            stream_token(console, full2)
            console.print()
            return full2, raw_chunk_count, sample_chunks

    return "", raw_chunk_count, sample_chunks


# ─── Ollama-specific path (think=False from the start) ──────────────────────


def _is_ollama(model: Any) -> bool:
    try:
        from langchain_ollama import ChatOllama  # type: ignore
    except ImportError:
        return False
    return isinstance(model, ChatOllama)


async def _ollama_pipeline(
    model: Any, prompt: str, console: Console, verbose: bool
) -> tuple[str, int, list[str]]:
    """End-to-end Ollama path that bypasses langchain-ollama.

    Three attempts, in order:
      1. Streaming with ``think=False`` — live tokens, no hidden reasoning.
         For thinking models (qwen3, deepseek-r1, phi4-reasoning) this is
         usually the only step that runs.
      2. Non-streaming with ``think=False`` — same plan, in case the stream
         hiccupped.
      3. Non-streaming with ``think=True`` — last resort that captures the
         ``thinking`` field separately so we can at least *show* the user
         what the model reasoned about, even if no final answer came out.

    All three pass an extended ``num_predict`` so reasoning models still
    have enough room when ``think=False`` isn't honored by the model.
    """
    import ollama as _ollama

    model_name = getattr(model, "model", None)
    base_url = getattr(model, "base_url", None) or "http://localhost:11434"
    if not model_name:
        return "", 0, []

    client = _ollama.AsyncClient(host=base_url)
    options = {"num_predict": 16384, "temperature": getattr(model, "temperature", 0.7)}
    messages = [{"role": "user", "content": prompt}]

    # ── Attempt 1: streaming, think=False ──
    chunks: list[str] = []
    raw_chunk_count = 0
    try:
        async for chunk in await _client_chat_stream(
            client, model_name, messages, options, think=False
        ):
            raw_chunk_count += 1
            content, _thinking = _split_message(chunk)
            if content:
                stream_token(console, content)
                chunks.append(content)
    except Exception as exc:
        render_error(console, f"ollama stream (think=False) failed: {exc}")

    if "".join(chunks).strip():
        console.print()
        return "".join(chunks), raw_chunk_count, []

    # ── Attempt 2: non-streaming, think=False ──
    if verbose:
        render_info(
            console,
            f"streaming returned empty across {raw_chunk_count} chunk(s) — "
            "retrying without streaming...",
        )
    resp = None
    try:
        resp = await _client_chat(client, model_name, messages, options, think=False)
    except Exception as exc:
        render_error(console, f"ollama non-stream (think=False) failed: {exc}")
    content, thinking = _split_message(resp)
    if content.strip():
        console.print()
        stream_token(console, content)
        console.print()
        return content, raw_chunk_count, []

    # ── Attempt 3: think=True so we can dump the reasoning ──
    if verbose:
        render_info(
            console,
            "still empty — retrying with think=True to capture reasoning...",
        )
    resp2 = None
    try:
        resp2 = await _client_chat(client, model_name, messages, options, think=True)
    except Exception as exc:
        render_error(console, f"ollama (think=True) failed: {exc}")
    content2, thinking2 = _split_message(resp2)
    if content2.strip():
        console.print()
        stream_token(console, content2)
        console.print()
        return content2, raw_chunk_count, []

    # No usable content. Dump thinking for transparency.
    final_thinking = thinking2 or thinking
    if final_thinking:
        from rich.panel import Panel as _Panel

        excerpt = final_thinking
        if len(excerpt) > 8000:
            excerpt = excerpt[:8000] + f"\n\n[…truncated, {len(final_thinking) - 8000} more chars]"
        console.print()
        console.print(
            _Panel(
                excerpt,
                title="[bold yellow1] MODEL THINKING (no final answer produced) [/]",
                border_style="yellow1",
                padding=(1, 2),
            )
        )
        render_info(
            console,
            "the model only produced thinking — no answer to use as code. "
            "switch to a non-reasoning model with [bold]/model use[/].",
        )

    return "", raw_chunk_count, []


async def _client_chat(client, model_name, messages, options, *, think):
    """Non-streaming chat. Falls back if the ``think`` kwarg isn't supported."""
    try:
        return await client.chat(
            model=model_name,
            messages=messages,
            options=options,
            stream=False,
            think=think,
        )
    except TypeError:
        # Older ollama python lib (<0.4) doesn't accept `think`.
        return await client.chat(
            model=model_name,
            messages=messages,
            options=options,
            stream=False,
        )


async def _client_chat_stream(client, model_name, messages, options, *, think):
    """Streaming chat. Returns an async iterator. Falls back without ``think``."""
    try:
        return await client.chat(
            model=model_name,
            messages=messages,
            options=options,
            stream=True,
            think=think,
        )
    except TypeError:
        return await client.chat(
            model=model_name,
            messages=messages,
            options=options,
            stream=True,
        )


def _split_message(resp: Any) -> tuple[str, str]:
    """Pull (content, thinking) out of an ollama chat response, dict or object."""
    if resp is None:
        return "", ""
    msg = resp.get("message") if isinstance(resp, dict) else getattr(resp, "message", None)
    if msg is None:
        return "", ""
    if isinstance(msg, dict):
        return str(msg.get("content") or ""), str(msg.get("thinking") or "")
    return (
        str(getattr(msg, "content", "") or ""),
        str(getattr(msg, "thinking", "") or ""),
    )


def _extract_text(chunk: Any) -> str:
    """Pull text out of an LLM chunk regardless of which shape the backend uses.

    Order of attempts:
      1. `chunk.content` as a plain string (OpenAI / Ollama default).
      2. `chunk.content` as a list of blocks (Anthropic / multimodal); take
         every block with a `text` string field, regardless of `type`.
      3. `chunk.tool_call_chunks` — tool-tuned Ollama models stream the
         response as a fake tool call when no tools are bound; the assistant's
         intended output ends up in the `args` string. Salvage it.
      4. `chunk.additional_kwargs[text|content|completion]`.
      5. `chunk.text` (older langchain shapes).
    """
    if chunk is None:
        return ""

    content = getattr(chunk, "content", None)
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        if parts:
            return "".join(parts)

    # Tool-call chunks — emitted by tool-tuned models even when no tool is bound.
    tcc = getattr(chunk, "tool_call_chunks", None)
    if tcc:
        parts: list[str] = []
        for c in tcc:
            if isinstance(c, dict):
                args = c.get("args")
            else:
                args = getattr(c, "args", None)
            if isinstance(args, str) and args:
                parts.append(args)
        if parts:
            return "".join(parts)

    extra = getattr(chunk, "additional_kwargs", None)
    if isinstance(extra, dict):
        for key in ("text", "content", "completion"):
            val = extra.get(key)
            if isinstance(val, str) and val:
                return val

    text = getattr(chunk, "text", None)
    if isinstance(text, str) and text:
        return text

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

This is a plain-text task, NOT a tool call. Do not invoke any function.
Output the .py file content as ordinary text — your message body is what we read.

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

Output ONLY the contents of the .py file as text. Start with the import line.
You MAY wrap the code in a ```python fence; the wrapper will strip it.
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
        # Either the file failed to import (broken syntax / missing import / …)
        # or it imported fine but didn't define a @tool with the expected name.
        # Surface the actual import error if there was one — much more useful
        # than the generic "no @tool found" message.
        load_err = last_load_error(target)
        target.unlink(missing_ok=True)
        if load_err:
            render_error(
                console,
                f"could not import [bold]{target.name}[/] — file removed.\n\n"
                f"[grey50]{load_err.strip()}[/]\n\n"
                "fix: regenerate (the LLM made a syntax/import mistake) or "
                "switch to a stronger code model with [bold]/model use[/].",
            )
        else:
            render_error(
                console,
                f"the file imported cleanly but no @tool named {name!r} was "
                "found in it. the file has been removed. likely the LLM gave "
                "the function a different name or skipped the @tool decorator.",
            )
        return None

    render_info(
        console,
        f"tool [bold yellow1]{name}[/] registered → "
        f"[grey50]{_pretty_path(target)}[/]\n"
        f"[grey50]registry now holds {len(TOOLS)} tool(s) total.[/]",
    )
    return name


def _pretty_path(p: Path) -> Path | str:
    """Render `p` as cwd-relative when possible, else as an absolute path."""
    try:
        return p.relative_to(Path.cwd())
    except ValueError:
        return p


# ─── tool wizard helpers ────────────────────────────────────────────────────


async def _ask_scope(ctx: SessionContext, console: Console) -> Path | None:
    workspace = user_tools_dir()
    global_dir = global_tools_dir()
    console.print()
    render_info(console, "where should this tool live?")
    render_info(
        console,
        f"  [bold]W[/]orkspace → [grey70]{workspace}[/]  "
        "[grey50](only the active workspace)[/]",
    )
    render_info(
        console,
        f"  [bold]G[/]lobal    → [grey70]{global_dir}[/]  "
        "[grey50](every workspace)[/]",
    )
    raw = await _prompt(ctx, "scope [W/g]")
    choice = raw.lower()
    if choice in ("", "w", "workspace", "p", "project"):
        return workspace
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
            # Empty draft → don't auto-cancel. Most weak Ollama models hiccup
            # on the first try; one retry is usually enough.
            choice = (
                await _prompt(
                    ctx,
                    "empty draft — [r]etry / [n]o (cancel) / [s]witch model",
                )
            ).lower()
            if choice in ("", "r", "retry", "regen", "regenerate", "y", "yes"):
                console.print()
                render_info(console, "retrying...")
                continue
            if choice in ("s", "switch"):
                render_info(
                    console,
                    "use [bold]/model use <name>[/] to switch, then run "
                    "[bold]/tool new[/] again.",
                )
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
    from rich.panel import Panel as _Panel

    meta = _TOOL_META_PROMPT.format(
        name=name, description=description, goal=goal, signature=signature
    )

    # Status panel BEFORE streaming, so the user sees something while the model thinks.
    console.print()
    console.print(
        _Panel.fit(
            "[grey50]drafting tool source...[/]",
            border_style="bright_magenta",
            title="[tag] DRAFT // tool source [/]",
            title_align="left",
            padding=(0, 1),
        )
    )
    console.print()

    raw, raw_chunk_count, sample_chunks = await _stream_with_fallback(
        model, meta, console
    )
    console.print()
    source = _strip_fences(raw.strip())

    if not source:
        debug = ""
        if sample_chunks:
            debug = f"\n\n[grey50]first chunk repr:[/]\n[grey70]{sample_chunks[0]}[/]"
        render_error(
            console,
            "model produced no usable text after 3 fallback strategies "
            f"({raw_chunk_count} stream chunks, {len(raw)} chars).\n"
            "\n[bold]most common cause:[/] you're on a [bold]reasoning model[/] "
            "(qwen3-thinking, deepseek-r1, phi4-reasoning, …) that burned the "
            "token budget on hidden `<think>` tokens and never produced a visible "
            "answer.\n"
            "\n[bold]try:[/]\n"
            "  · [bold]/settings[/] → bump [bold]max_tokens[/] to 16384+ "
            "(reasoning needs lots of room)\n"
            "  · [bold]/model use <non-reasoning>[/] — qwen2.5:7b, llama3.1:8b, "
            "mistral, codellama work well for codegen\n"
            "  · check `ollama logs` while it runs — you'll see if it's just "
            f"thinking{debug}",
        )
        return ""

    # Re-render the final source as syntax-highlighted code.
    console.print()
    syntax = Syntax(
        source,
        "python",
        theme="ansi_dark",
        line_numbers=True,
        background_color="default",
    )
    console.print(
        _Panel(
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
        f"[grey50]{_pretty_path(skill_md_path)}[/]",
    )
    return name


async def _ask_skill_scope(ctx: SessionContext, console: Console) -> Path | None:
    workspace = project_skills_dir()
    global_dir = global_skills_dir()
    console.print()
    render_info(console, "where should this skill live?")
    render_info(
        console,
        f"  [bold]W[/]orkspace → [grey70]{workspace}[/]  [grey50](only the active workspace)[/]",
    )
    render_info(
        console,
        f"  [bold]G[/]lobal    → [grey70]{global_dir}[/]  [grey50](every workspace)[/]",
    )
    raw = await _prompt(ctx, "scope [W/g]")
    choice = raw.lower()
    if choice in ("", "w", "workspace", "p", "project"):
        return workspace
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

    text, _count, _samples = await _stream_with_fallback(model, meta, console)
    console.print()
    return _strip_fences(text.strip())


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
