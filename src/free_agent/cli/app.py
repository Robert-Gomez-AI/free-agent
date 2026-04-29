from __future__ import annotations

import asyncio
import logging
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.console import Console

from free_agent.agent.builder import build_session
from free_agent.agent.loader import find_config, load_profile
from free_agent.cli.commands import SlashResult, handle_slash_command
from free_agent.cli.console import (
    boot_progress,
    end_stream,
    make_console,
    render_aborted,
    render_assistant_prefix,
    render_banner_glitch,
    render_disconnect,
    render_error,
    render_info,
    render_separator,
    render_todos,
    render_tool_call,
    render_tool_result,
    stream_token,
    type_boot_line,
)
from free_agent.cli.context import SessionContext
from free_agent.cli.slash_registry import SlashCompleter
from free_agent.config import Settings
from free_agent.session.history import Conversation

log = logging.getLogger(__name__)

# prompt_toolkit input style — neon cyan prompt, white input, magenta completion menu.
PROMPT_STYLE = Style.from_dict(
    {
        "tag": "fg:#ff00ff bold",
        "arrow": "fg:#00ffff bold",
        "label": "fg:#00ffff bold",
        "": "fg:#e0e0e0",
        # Completion menu — keep readable on both light and dark terminals.
        "completion-menu":                          "bg:#1a0033",
        "completion-menu.completion":               "bg:#1a0033 fg:#e0e0e0",
        "completion-menu.completion.current":       "bg:#ff00ff fg:#000000 bold",
        "completion-menu.meta.completion":          "bg:#1a0033 fg:#888888",
        "completion-menu.meta.completion.current":  "bg:#ff00ff fg:#000000",
        "completion-menu.multi-column-meta":        "bg:#1a0033 fg:#888888",
        "scrollbar.background":                     "bg:#1a0033",
        "scrollbar.button":                         "bg:#ff00ff",
    }
)

PROMPT_FRAGMENT = HTML(
    '<tag>▓▒░</tag> <label>you</label> <arrow>▶</arrow> '
)


async def run(settings: Settings, *, config_override: "Path | None" = None) -> int:
    from pathlib import Path

    console = make_console()

    # Glitch decode of the banner.
    await render_banner_glitch(console, settings.active_model, settings.provider)
    console.print()

    # Resolve writable mode — CLI flag already merged into settings; here we just
    # turn the flag into a concrete root directory the agent is allowed to touch.
    writable_root: Path | None = Path.cwd().resolve() if settings.writable else None

    # Load the (optional) agent profile from free-agent.yaml.
    try:
        config_path = find_config(config_override)
        profile = load_profile(config_path)
    except (FileNotFoundError, ValueError) as exc:
        await type_boot_line(console, " HALT ", "profile load failed.", ok=False)
        render_error(console, str(exc))
        return 1

    if config_path is not None:
        await type_boot_line(
            console,
            " CONF ",
            f"profile loaded → {config_path.name} "
            f"(subagents: {len(profile.subagents)})",
        )

    # Type-in boot line, then animated noise while model + agent are built in a thread.
    await type_boot_line(console, " BOOT ", "spinning up neural shell...")
    build_task: asyncio.Task[tuple[Any, Any]] = asyncio.create_task(
        asyncio.to_thread(build_session, settings, profile, writable_root=writable_root)
    )
    try:
        await boot_progress(console, "establishing neural uplink", build_task)
        chat_model, agent = build_task.result()
    except Exception as exc:
        await type_boot_line(console, " HALT ", "boot failed.", ok=False)
        render_error(console, str(exc))
        return 1
    await type_boot_line(
        console,
        "  OK  ",
        f"agent online → {settings.active_model} @ {settings.provider}",
    )
    if writable_root is not None:
        await type_boot_line(
            console,
            " WRT  ",
            f"WRITE MODE — agent can modify files under {writable_root}",
            ok=False,  # render in red/warning style
        )
    console.print()

    # Box that holds the SessionContext so the completer (built before the
    # context exists) can resolve it lazily for dynamic argument completion.
    ctx_box: list[SessionContext | None] = [None]

    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(settings.history_file)),
        style=PROMPT_STYLE,
        completer=SlashCompleter(ctx_provider=lambda: ctx_box[0]),
        complete_while_typing=True,
    )
    ctx = SessionContext(
        conversation=Conversation(),
        settings=settings,
        profile=profile,
        chat_model=chat_model,
        agent=agent,
        prompt_session=session,
        config_path=config_path,
        writable_root=writable_root,
    )
    ctx_box[0] = ctx

    while True:
        try:
            with patch_stdout():
                user_input = await session.prompt_async(PROMPT_FRAGMENT)
        except (EOFError, KeyboardInterrupt):
            console.print()
            render_disconnect(console)
            return 0

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.startswith("/"):
            result = await handle_slash_command(user_input, ctx, console)
            if result is SlashResult.QUIT:
                render_disconnect(console)
                return 0
            if result is SlashResult.RETRY:
                last_user = ctx.conversation.last_user_message()
                if last_user is None:
                    render_info(console, "no previous turn to retry.")
                    continue
                await _attempt_turn(ctx.agent, ctx.conversation, console)
            elif result is SlashResult.RUN:
                # The slash handler already appended a user message; just run.
                await _attempt_turn(ctx.agent, ctx.conversation, console)
            elif result is SlashResult.RUN_PLAN:
                await _run_planning_loop(ctx, console)
            continue

        ctx.conversation.append_user(user_input)
        await _attempt_turn(ctx.agent, ctx.conversation, console)


async def _attempt_turn(
    agent: Any,
    conversation: Conversation,
    console: Console,
) -> list[dict[str, Any]] | None:
    """Stream one assistant turn. Returns the latest plan snapshot (or None).

    On Ctrl+C, cancels and rolls back the user message; returns None.
    """
    task = asyncio.create_task(_stream_turn(agent, conversation, console))
    try:
        assistant_text, latest_todos = await task
    except (asyncio.CancelledError, KeyboardInterrupt):
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, KeyboardInterrupt):
                pass
        console.print()
        render_aborted(console)
        if conversation.messages and conversation.messages[-1]["role"] == "user":
            conversation.pop_last()
        return None
    except Exception as exc:
        log.exception("turn failed")
        console.print()
        render_error(console, str(exc))
        if conversation.messages and conversation.messages[-1]["role"] == "user":
            conversation.pop_last()
        return None

    if assistant_text:
        conversation.append_assistant(assistant_text)
    render_separator(console)
    return latest_todos


_PLAN_MAX_STEPS = 4  # initial turn + up to 3 auto-continues


async def _run_planning_loop(ctx: SessionContext, console: Console) -> None:
    """Run an initial /plan turn, then auto-continue while pending todos remain."""
    from free_agent.cli.commands import CONTINUE_DIRECTIVE

    previous_todos: list[dict[str, Any]] | None = None
    for step in range(_PLAN_MAX_STEPS):
        latest = await _attempt_turn(ctx.agent, ctx.conversation, console)

        # Bail conditions: turn cancelled / errored, no plan to drive, or finished.
        if latest is None:
            return
        unfinished = [t for t in latest if t.get("status") != "completed"]
        if not unfinished:
            render_info(
                console,
                "[bold bright_green]plan complete[/] — all todos marked completed.",
            )
            return

        # Stall detection: same plan two turns in a row → model isn't progressing.
        if previous_todos is not None and _todos_equal(previous_todos, latest):
            render_info(
                console,
                "agent appears stalled (plan unchanged) — stopping auto-continue. "
                "type your own message to nudge it.",
            )
            return
        previous_todos = latest

        if step + 1 >= _PLAN_MAX_STEPS:
            render_info(
                console,
                f"hit auto-continue limit ({_PLAN_MAX_STEPS} turns). "
                f"{len(unfinished)} todo(s) still pending — say 'continue' to resume.",
            )
            return

        # Inject the continue directive and loop.
        render_info(
            console,
            f"[bold bright_magenta]auto-continue[/] "
            f"({step + 1}/{_PLAN_MAX_STEPS - 1})  ·  "
            f"{len(unfinished)} todo(s) still open",
        )
        ctx.conversation.append_user(CONTINUE_DIRECTIVE)


def _todos_equal(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x.get("content") != y.get("content") or x.get("status") != y.get("status"):
            return False
    return True


async def _stream_turn(
    agent: Any, conversation: Conversation, console: Console
) -> tuple[str, list[dict[str, Any]] | None]:
    """Stream one assistant turn.

    A turn can interleave:
      - assistant text chunks  → rendered live as Markdown via rich.live.Live
      - tool calls / results   → rendered as inline lines outside the Live region
      - todo writes            → rendered as a plan checklist

    Each contiguous span of assistant text gets its own Live section. When a
    tool call interrupts, we close the current Live (committing the rendered
    Markdown to scrollback) and the next stream-chunk opens a new Live below.

    Fallback: weaker models sometimes hallucinate a JSON `{"todos": [...]}`
    object in their reply instead of invoking write_todos. After each section
    closes, we scan the buffer and render any inline todo JSON as a styled
    checklist (so the user gets the same UX either way).

    Returns `(assistant_text, latest_todos)` where `latest_todos` is the most
    recent plan snapshot seen during the turn (from a real `write_todos` call
    or from inline JSON), or None if the agent never produced one.
    """
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.padding import Padding

    payload = conversation.to_payload()
    chunks: list[str] = []

    state: dict[str, Any] = {"live": None, "buffer": [], "latest_todos": None}

    def _renderable(text: str) -> Padding:
        return Padding(Markdown(text or " "), (0, 0, 0, 2))

    def _begin_section() -> None:
        if state["live"] is not None:
            return
        # Prefix goes on its own line above the Live region.
        render_assistant_prefix(console)
        console.print()
        state["buffer"] = []
        live = Live(
            _renderable(""),
            console=console,
            refresh_per_second=10,
            transient=False,
            vertical_overflow="visible",
        )
        live.__enter__()
        state["live"] = live

    def _end_section() -> None:
        live = state["live"]
        if live is None:
            return
        text = "".join(state["buffer"]).strip()
        # Fallback: detect inline JSON the model wrote instead of tool-calling.
        todos: list[dict[str, Any]] | None = None
        cleaned = text
        if text:
            todos, cleaned = _extract_inline_todos(text)
        # Replace the Live's content with the prose-only version (JSON stripped).
        live.update(_renderable(cleaned or " "))
        live.__exit__(None, None, None)
        state["live"] = None
        if todos:
            render_todos(console, todos)
            state["latest_todos"] = todos

    try:
        async for event in agent.astream_events(payload, version="v2"):
            kind = event.get("event")

            if kind == "on_chat_model_stream":
                text = _extract_text(event["data"].get("chunk"))
                if not text:
                    continue
                if state["live"] is None:
                    _begin_section()
                state["buffer"].append(text)
                chunks.append(text)
                state["live"].update(_renderable("".join(state["buffer"])))

            elif kind == "on_tool_start":
                _end_section()
                name = event.get("name", "tool")
                inputs = event["data"].get("input")
                if isinstance(inputs, dict) and "input" in inputs and len(inputs) == 1:
                    inputs = inputs["input"]
                if name == "write_todos" and isinstance(inputs, dict):
                    todos = inputs.get("todos") or []
                    render_todos(console, todos)
                    # Track the latest plan snapshot from real tool calls.
                    state["latest_todos"] = [
                        {
                            "content": t.get("content", ""),
                            "status": t.get("status", "pending"),
                        }
                        for t in todos
                        if isinstance(t, dict)
                    ]
                else:
                    console.print()
                    render_tool_call(
                        console, name, inputs if isinstance(inputs, dict) else None
                    )

            elif kind == "on_tool_end":
                name = event.get("name", "tool")
                if name != "write_todos":
                    output = event["data"].get("output")
                    render_tool_result(console, name, output)
                # The next on_chat_model_stream will open a fresh section.
    finally:
        _end_section()

    return "".join(chunks).strip(), state["latest_todos"]


def _extract_inline_todos(
    text: str,
) -> tuple[list[dict[str, Any]] | None, str]:
    """Detect a `{"todos": [...]}` JSON object inside assistant text.

    Open-source models with weak tool-calling sometimes emit the plan as a
    JSON code block in their reply instead of calling `write_todos`. We
    salvage it: locate the JSON object, parse it, normalize the schema
    (different models name the content field differently — `content`,
    `description`, `text`, `task`), and return the same shape `render_todos`
    expects.

    Returns `(todos_or_None, cleaned_text)` where `cleaned_text` has the
    JSON block (and any surrounding ```json fence) stripped on success, so
    callers can render the prose body without the redundant JSON dump.
    """
    import json as _json
    import re as _re

    if '"todos"' not in text:
        return None, text

    for start in (i for i in range(len(text)) if text[i] == "{"):
        if text.find('"todos"', start) < 0:
            continue
        # Find the balanced closing brace, ignoring braces inside strings.
        depth = 0
        in_str = False
        escape = False
        end = -1
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end < 0:
            continue

        candidate = text[start:end]
        try:
            data = _json.loads(candidate)
        except _json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        raw_todos = data.get("todos")
        if not isinstance(raw_todos, list) or not raw_todos:
            continue

        normalized: list[dict[str, Any]] = []
        for t in raw_todos:
            if not isinstance(t, dict):
                continue
            content = (
                t.get("content")
                or t.get("description")
                or t.get("text")
                or t.get("task")
                or ""
            )
            status = str(t.get("status", "pending")).lower()
            if status not in ("pending", "in_progress", "completed"):
                status = {
                    "done": "completed",
                    "doing": "in_progress",
                    "todo": "pending",
                    "wip": "in_progress",
                    "in-progress": "in_progress",
                }.get(status, "pending")
            normalized.append({"content": str(content).strip(), "status": status})

        if not normalized:
            continue

        # Strip the JSON span — and any wrapping ```json / ``` / ~~~ fence —
        # so the prose body can re-render without the duplicated JSON dump.
        strip_start, strip_end = start, end
        before = text[:strip_start]
        m = _re.search(r"(?:^|\n)\s*(?:```|~~~)[a-zA-Z0-9_-]*\s*\n?\s*$", before)
        if m:
            strip_start = m.start() + (1 if m.group(0).startswith("\n") else 0)
        after = text[strip_end:]
        m = _re.match(r"\s*\n?\s*(?:```|~~~)\s*", after)
        if m:
            strip_end = strip_end + m.end()

        cleaned = (text[:strip_start] + text[strip_end:]).strip()
        return normalized, cleaned

    return None, text


def _extract_text(chunk: Any) -> str:
    """Pull text out of a LangChain message chunk regardless of content shape."""
    if chunk is None:
        return ""
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""
