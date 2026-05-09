from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path

from rich.console import Console

from free_agent.agent.loader import save_profile
from free_agent.agent.ollama_admin import (
    delete_model as _ollama_delete,
    list_models as _ollama_list,
    pull_model as _ollama_pull,
)
from free_agent.agent.ollama_catalog import filter_catalog as _filter_curated
from free_agent.agent.ollama_library import (
    LibraryEntry,
    cached_library,
    filter_library,
)
from free_agent.cli.console import (
    render_agent_profile,
    render_error,
    render_info,
    render_markdown_block,
    render_model_library,
    render_model_list,
    render_tools_inventory,
)
from free_agent.cli.context import SessionContext
from free_agent.agent.skills_registry import (
    global_skills_dir,
    list_skills,
    project_skills_dir,
)
from free_agent.cli.settings_panel import open_settings
from free_agent.cli.wizard import (
    create_skill_wizard,
    create_subagent_wizard,
    create_tool_wizard,
)
from free_agent.tools import (
    DEEPAGENTS_BUILTINS,
    TOOLS,
    global_tools_dir,
    is_user_tool,
    origin_of,
    reload_tools,
    user_tools_dir,
)
from free_agent.workspace import (
    create_workspace,
    delete_workspace,
    get_workspace,
    list_workspaces,
)


class SlashResult(Enum):
    HANDLED = "handled"
    QUIT = "quit"
    RETRY = "retry"
    RUN = "run"          # main loop runs one turn
    RUN_PLAN = "run_plan"  # main loop runs the planning loop (turn + auto-continue)


PLAN_DIRECTIVE = """\
[planning mode — plan AND execute]

Protocol for this turn:

1. INVOKE the `write_todos` function (a real tool call, NOT text). Argument: \
`todos`, a list of `{content, status}` objects. Initial status is "pending". \
Write at least 3 concrete todos for the task. Do NOT write JSON in your \
reply — invoke the function.

2. EXECUTE the plan. The plan is NOT your answer. Pick the first todo and \
do the actual work — research, write, call other tools, whatever it requires.

3. UPDATE statuses as you progress: call `write_todos` again with the same \
list but the relevant todo's status moved (pending → in_progress → completed).

4. Continue through every todo. End the turn ONLY when every todo is \
"completed" AND you have answered the original task with prose.

Do NOT stop after step 1. Writing the plan and giving a brief acknowledgment \
is HALF the job. The user wants you to execute the plan you wrote.

Task:
"""


# Sent automatically by the planning loop when the previous turn left todos
# in pending / in_progress state. Keeps the agent moving forward without the
# user having to nudge it. Formatted with the current plan snapshot so the
# model has explicit context (the JSON it wrote was stripped from history
# to avoid same-plan regurgitation).
CONTINUE_DIRECTIVE_TEMPLATE = """\
[plan execution — continue]

Current plan state:
{plan_snapshot}

Pick the FIRST pending or in_progress todo above and actually answer it now, \
in prose. Most of these are knowledge tasks — just write the answer directly. \
Do NOT write fake shell commands like `ls /foo` or `cat /bar`; those are not \
tool calls, they're dead text. Only call a tool if the registered tool list \
actually contains one that fits.

After answering, re-emit the plan (call `write_todos`, or write the same \
JSON object) with that todo's status moved to "completed". Then move to the \
next one. End ONLY when every todo is "completed" and the original task has \
a real prose answer.
"""


def format_continue_directive(todos: list[dict]) -> str:
    """Render CONTINUE_DIRECTIVE_TEMPLATE with the current plan snapshot."""
    if not todos:
        snapshot = "(no plan recorded)"
    else:
        snapshot = "\n".join(
            f"- [{t.get('status', 'pending')}] {t.get('content', '')}"
            for t in todos
        )
    return CONTINUE_DIRECTIVE_TEMPLATE.format(plan_snapshot=snapshot)


HELP_TEXT = """\
| cmd | effect |
|---|---|
| `/help`    | show this panel |
| `/plan <task>` | force the agent to write_todos before acting on the task |
| `/tools`   | list every tool the agent can call |
| `/agent`   | show the loaded agent profile (main + subagents) |
| `/sub new`         | wizard — create a subagent (LLM drafts the system prompt) |
| `/sub rm <name>`   | remove a subagent by name |
| `/sub list`        | alias for `/agent` |
| `/tool new`        | wizard — generate a new tool (LLM writes the Python) |
| `/tool rm <name>`  | delete a user-created tool file |
| `/tool reload`     | re-scan tool folders without restart |
| `/tool open [scope]` | open the tools folder in your file manager (`project` / `global` / `both`) |
| `/tool dir`        | print the tool folder paths without opening |
| `/skill list`      | list every SKILL.md loaded |
| `/skill new`       | wizard — generate a SKILL.md (LLM drafts the body) |
| `/skill rm <name>` | delete a skill folder |
| `/skill reload`    | rebuild the agent so it sees skill folder changes |
| `/skill open [s]`  | open the skills folder in your file manager |
| `/skill dir`       | print skill folder paths |
| `/model list`      | local Ollama models (size + active marker) |
| `/model browse [q]` | curated catalog of pullable models (with [q] substring filter) |
| `/model pull <name>` | download a model with live progress |
| `/model rm <name>` | remove a local Ollama model |
| `/model use [name]` | switch the active model — no name opens an interactive picker (Ollama pulled + Anthropic catalog) |
| `/writable [on\\|off\\|<path>]` | toggle real-disk mode (off → virtual fs); no arg shows state |
| `/settings` | open the full-screen settings panel (provider · model · writable · …) |
| `/ws list`        | list workspaces — active marker, paths |
| `/ws current`     | show the active workspace details |
| `/ws new <name>`  | create an empty workspace |
| `/ws clone <src> <dst>` | duplicate a workspace's profile + tools + skills |
| `/ws use <name>`  | switch active workspace (reloads profile, tools, skills) |
| `/ws rm <name>`   | delete a workspace (refuses active or last) |
| `/ws open`        | open the active workspace folder in your file manager |
| `/clear`   | wipe the current session buffer |
| `/history` | dump the session as markdown |
| `/save`    | export session to `session-YYYYMMDD-HHMMSS.md` |
| `/retry`   | drop last AI turn and re-fire the previous prompt |
| `/exit`    | sever the link (Ctrl+D works too) |

**Ctrl+C** aborts the current turn without quitting.
**Customize:** edit `free-agent.yaml` and `./free_agent_tools/*.py`, or build them via `/sub new` and `/tool new`.
"""


async def handle_slash_command(
    line: str,
    ctx: SessionContext,
    console: Console,
) -> SlashResult:
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    conversation = ctx.conversation

    if cmd in ("/exit", "/quit"):
        return SlashResult.QUIT

    if cmd == "/help":
        render_markdown_block(console, HELP_TEXT, title="▓▒░ COMMANDS ░▒▓")
        return SlashResult.HANDLED

    if cmd == "/plan":
        if not rest:
            render_error(
                console,
                "usage: /plan <task>  — forces the agent to plan AND execute.",
            )
            return SlashResult.HANDLED
        ctx.conversation.append_user(PLAN_DIRECTIVE + rest)
        render_info(
            console,
            "[bold bright_magenta]planning mode[/] — agent will plan, then execute. "
            "auto-continues if it stops mid-plan.",
        )
        return SlashResult.RUN_PLAN

    if cmd == "/tools":
        render_tools_inventory(console, TOOLS, DEEPAGENTS_BUILTINS)
        return SlashResult.HANDLED

    if cmd == "/agent":
        _render_profile(ctx, console)
        return SlashResult.HANDLED

    if cmd in ("/sub", "/subagent"):
        return await _handle_sub(rest, ctx, console)

    if cmd == "/tool":
        return await _handle_tool(rest, ctx, console)

    if cmd == "/skill":
        return await _handle_skill(rest, ctx, console)

    if cmd == "/model":
        return await _handle_model(rest, ctx, console)

    if cmd == "/writable":
        return _handle_writable(rest, ctx, console)

    if cmd == "/settings":
        await open_settings(ctx, console)
        return SlashResult.HANDLED

    if cmd in ("/ws", "/workspace"):
        return await _handle_workspace(rest, ctx, console)

    if cmd == "/clear":
        conversation.clear()
        render_info(console, "buffer wiped — context purged.")
        return SlashResult.HANDLED

    if cmd == "/history":
        if not conversation.messages:
            render_info(console, "(buffer empty)")
        else:
            render_markdown_block(
                console,
                conversation.to_markdown(),
                title=f"▓▒░ TRACE :: {len(conversation.messages)} msgs ░▒▓",
            )
        return SlashResult.HANDLED

    if cmd == "/save":
        path = Path(rest) if rest else Path(f"session-{datetime.now():%Y%m%d-%H%M%S}.md")
        try:
            path.write_text(conversation.to_markdown(), encoding="utf-8")
            render_info(console, f"trace dumped → [bold bright_cyan]{path}[/]")
        except OSError as e:
            render_error(console, f"could not save: {e}")
        return SlashResult.HANDLED

    if cmd == "/retry":
        if conversation.messages and conversation.messages[-1]["role"] == "assistant":
            conversation.pop_last()
            return SlashResult.RETRY
        render_info(console, "nothing to retry.")
        return SlashResult.HANDLED

    render_error(console, f"unknown command: {cmd} — try /help.")
    return SlashResult.HANDLED


# ─── /ws (workspace) dispatcher ─────────────────────────────────────────────


async def _handle_workspace(
    rest: str, ctx: SessionContext, console: Console
) -> SlashResult:
    parts = rest.split(maxsplit=2)
    sub = parts[0].lower() if parts else ""

    if sub in ("", "list", "ls"):
        return _ws_list(ctx, console)
    if sub in ("current", "show", "info"):
        return _ws_current(ctx, console)
    if sub in ("use", "switch", "activate"):
        return await _ws_use(parts[1] if len(parts) > 1 else "", ctx, console)
    if sub in ("new", "create", "add"):
        return await _ws_new(parts[1] if len(parts) > 1 else "", ctx, console)
    if sub == "clone":
        src = parts[1] if len(parts) > 1 else ""
        dst = parts[2] if len(parts) > 2 else ""
        return await _ws_clone(src, dst, ctx, console)
    if sub in ("rm", "remove", "del", "delete"):
        return await _ws_remove(parts[1] if len(parts) > 1 else "", ctx, console)
    if sub in ("open", "edit", "explore"):
        _open_in_file_manager(ctx.workspace.root, console)
        return SlashResult.HANDLED
    if sub in ("dir", "where", "path"):
        render_info(
            console,
            f"workspace [bold bright_magenta]{ctx.workspace.name}[/] → "
            f"[bold bright_cyan]{ctx.workspace.root}[/]",
        )
        return SlashResult.HANDLED

    render_error(
        console,
        f"unknown /ws action: {sub!r}. try: list · current · use · new · clone · rm · open · dir",
    )
    return SlashResult.HANDLED


def _ws_list(ctx: SessionContext, console: Console) -> SlashResult:
    workspaces = list_workspaces()
    if not workspaces:
        render_info(console, "no workspaces yet. create one with [bold]/ws new <name>[/].")
        return SlashResult.HANDLED

    lines = ["| active | name | path | tools | skills |", "|---|---|---|---|---|"]
    for ws in workspaces:
        marker = "●" if ws.name == ctx.workspace.name else " "
        n_tools = (
            sum(1 for _ in ws.tools_dir.glob("*.py"))
            if ws.tools_dir.is_dir()
            else 0
        )
        n_skills = (
            sum(1 for c in ws.skills_dir.iterdir() if c.is_dir())
            if ws.skills_dir.is_dir()
            else 0
        )
        lines.append(
            f"| {marker} | **{ws.name}** | `{ws.root}` | {n_tools} | {n_skills} |"
        )
    render_markdown_block(console, "\n".join(lines), title="▓▒░ WORKSPACES ░▒▓")
    return SlashResult.HANDLED


def _ws_current(ctx: SessionContext, console: Console) -> SlashResult:
    ws = ctx.workspace
    n_tools = (
        sum(1 for _ in ws.tools_dir.glob("*.py")) if ws.tools_dir.is_dir() else 0
    )
    n_skills = (
        sum(1 for c in ws.skills_dir.iterdir() if c.is_dir())
        if ws.skills_dir.is_dir()
        else 0
    )
    has_yaml = "yes" if ws.config_file.is_file() else "no (using defaults)"
    body = (
        f"- **name**:    `{ws.name}`\n"
        f"- **root**:    `{ws.root}`\n"
        f"- **profile**: {has_yaml}\n"
        f"- **tools**:   {n_tools} python file(s) under `{ws.tools_dir}`\n"
        f"- **skills**:  {n_skills} skill folder(s) under `{ws.skills_dir}`\n"
    )
    render_markdown_block(console, body, title="▓▒░ ACTIVE WORKSPACE ░▒▓")
    return SlashResult.HANDLED


async def _ws_use(name: str, ctx: SessionContext, console: Console) -> SlashResult:
    if not name:
        render_error(console, "usage: /ws use <name>")
        return SlashResult.HANDLED
    target = get_workspace(name)
    if target is None:
        render_error(
            console,
            f"no workspace named {name!r}. existing: "
            f"{[w.name for w in list_workspaces()]}",
        )
        return SlashResult.HANDLED
    if target.name == ctx.workspace.name:
        render_info(console, f"already on [bold bright_magenta]{name}[/].")
        return SlashResult.HANDLED

    try:
        ctx.switch_workspace(target)
    except Exception as exc:
        render_error(console, f"switch failed: {exc}")
        return SlashResult.HANDLED

    render_info(
        console,
        f"workspace → [bold bright_magenta]{target.name}[/]  "
        f"[grey50]({target.root})[/]",
    )
    return SlashResult.HANDLED


async def _ws_new(name: str, ctx: SessionContext, console: Console) -> SlashResult:
    if not name:
        render_error(console, "usage: /ws new <name>")
        return SlashResult.HANDLED
    try:
        ws = create_workspace(name)
    except ValueError as exc:
        render_error(console, str(exc))
        return SlashResult.HANDLED

    render_info(
        console,
        f"workspace [bold bright_magenta]{ws.name}[/] created at "
        f"[bold bright_cyan]{ws.root}[/].",
    )
    try:
        ans = await ctx.prompt_session.prompt_async(
            f"  switch to [bold bright_magenta]{ws.name}[/] now? [Y/n] "
        )
    except (KeyboardInterrupt, EOFError):
        ans = "n"
    if ans.strip().lower() in ("", "y", "yes"):
        try:
            ctx.switch_workspace(ws)
            render_info(console, f"now on [bold bright_magenta]{ws.name}[/].")
        except Exception as exc:
            render_error(console, f"switch failed: {exc}")
    return SlashResult.HANDLED


async def _ws_clone(
    src_name: str, dst_name: str, ctx: SessionContext, console: Console
) -> SlashResult:
    if not src_name or not dst_name:
        render_error(console, "usage: /ws clone <src> <dst>")
        return SlashResult.HANDLED
    src = get_workspace(src_name)
    if src is None:
        render_error(console, f"no workspace named {src_name!r}.")
        return SlashResult.HANDLED
    try:
        ws = create_workspace(dst_name, seed_from=src)
    except ValueError as exc:
        render_error(console, str(exc))
        return SlashResult.HANDLED
    render_info(
        console,
        f"cloned [bold bright_magenta]{src.name}[/] → "
        f"[bold bright_magenta]{ws.name}[/]  [grey50]({ws.root})[/]",
    )
    return SlashResult.HANDLED


async def _ws_remove(
    name: str, ctx: SessionContext, console: Console
) -> SlashResult:
    if not name:
        render_error(console, "usage: /ws rm <name>")
        return SlashResult.HANDLED
    target = get_workspace(name)
    if target is None:
        render_error(console, f"no workspace named {name!r}.")
        return SlashResult.HANDLED
    try:
        ans = await ctx.prompt_session.prompt_async(
            f"  delete workspace [bold bright_magenta]{name}[/] and ALL its "
            f"contents? this cannot be undone. [y/N] "
        )
    except (KeyboardInterrupt, EOFError):
        ans = "n"
    if ans.strip().lower() not in ("y", "yes"):
        render_info(console, "kept.")
        return SlashResult.HANDLED
    try:
        delete_workspace(name)
    except ValueError as exc:
        render_error(console, str(exc))
        return SlashResult.HANDLED
    render_info(console, f"workspace [bold bright_magenta]{name}[/] removed.")
    return SlashResult.HANDLED


# ─── /writable dispatcher ───────────────────────────────────────────────────


def _handle_writable(arg: str, ctx: SessionContext, console: Console) -> SlashResult:
    """Toggle real-filesystem mode at runtime.

    Usage:
      /writable             show current state
      /writable on          enable, scoped to the current working directory
      /writable off         disable (back to the in-memory virtual filesystem)
      /writable <path>      enable, scoped to the given path
    """
    arg = arg.strip()

    if not arg:
        if ctx.writable_root is None:
            render_info(
                console,
                "writable mode is [bold]OFF[/] — agent uses the in-memory virtual "
                f"filesystem. enable with [bold bright_cyan]/writable on[/] (scope = "
                f"[bold bright_cyan]{Path.cwd()}[/]) or [bold bright_cyan]/writable <path>[/].",
            )
        else:
            render_info(
                console,
                "writable mode is [bold red]ON[/] — agent reads/writes under "
                f"[bold bright_cyan]{ctx.writable_root}[/]. disable with "
                "[bold bright_cyan]/writable off[/].",
            )
        return SlashResult.HANDLED

    low = arg.lower()
    if low == "off":
        if ctx.writable_root is None:
            render_info(console, "already off.")
            return SlashResult.HANDLED
        old_root = ctx.writable_root
        old_flag = ctx.settings.writable
        ctx.writable_root = None
        ctx.settings.writable = False
        try:
            ctx.rebuild_agent()
        except Exception as exc:
            ctx.writable_root = old_root
            ctx.settings.writable = old_flag
            render_error(console, f"rebuild failed; writable mode unchanged: {exc}")
            return SlashResult.HANDLED
        render_info(
            console,
            "writable mode [bold]OFF[/] — virtual filesystem active. "
            "(the [bold yellow1]shell[/] tool still works on the real machine.)",
        )
        return SlashResult.HANDLED

    if low == "on":
        new_root = Path.cwd().resolve()
    else:
        new_root = Path(arg).expanduser().resolve()

    if not new_root.is_dir():
        render_error(console, f"not a directory: {new_root}")
        return SlashResult.HANDLED

    old_root = ctx.writable_root
    old_flag = ctx.settings.writable
    ctx.writable_root = new_root
    ctx.settings.writable = True
    try:
        ctx.rebuild_agent()
    except Exception as exc:
        ctx.writable_root = old_root
        ctx.settings.writable = old_flag
        render_error(console, f"rebuild failed; writable mode unchanged: {exc}")
        return SlashResult.HANDLED
    render_info(
        console,
        f"writable mode [bold red]ON[/] — agent can read/write under "
        f"[bold bright_cyan]{new_root}[/]. paths cannot escape via .. / ~ / absolute outside-root.",
    )
    return SlashResult.HANDLED


# ─── /sub dispatcher ────────────────────────────────────────────────────────


async def _handle_sub(rest: str, ctx: SessionContext, console: Console) -> SlashResult:
    parts = rest.split(maxsplit=1)
    sub = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    if sub in ("", "list", "ls"):
        _render_profile(ctx, console)
        return SlashResult.HANDLED

    if sub in ("new", "create", "add"):
        return await _sub_new(ctx, console)

    if sub in ("rm", "remove", "del", "delete"):
        return await _sub_remove(arg, ctx, console)

    render_error(
        console,
        f"unknown /sub action: {sub!r}. try: /sub new · /sub rm <name> · /sub list",
    )
    return SlashResult.HANDLED


async def _sub_new(ctx: SessionContext, console: Console) -> SlashResult:
    try:
        new = await create_subagent_wizard(ctx, console)
    except (KeyboardInterrupt, EOFError):
        console.print()
        render_info(console, "wizard cancelled.")
        return SlashResult.HANDLED

    if new is None:
        return SlashResult.HANDLED

    # Add to profile + rebuild agent. Roll back on failure.
    ctx.profile.subagents.append(new)
    try:
        ctx.rebuild_agent()
    except Exception as exc:
        ctx.profile.subagents.pop()
        render_error(console, f"rebuild failed; subagent rolled back: {exc}")
        return SlashResult.HANDLED

    console.print()
    render_info(
        console,
        f"subagent [bold bright_magenta]{new.name}[/] online "
        f"({len(ctx.profile.subagents)} total).",
    )

    # Offer to persist.
    try:
        choice = await ctx.prompt_session.prompt_async(
            f"  save to {ctx.config_path or 'free-agent.yaml'}? [Y/n] "
            "(rewrites file; comments lost) "
        )
    except (KeyboardInterrupt, EOFError):
        choice = "n"
    if choice.strip().lower() in ("", "y", "yes"):
        try:
            written = save_profile(ctx.config_path, ctx.profile)
            ctx.config_path = written
            render_info(console, f"persisted → [bold bright_cyan]{written}[/]")
        except Exception as exc:
            render_error(console, f"could not save: {exc}")
    else:
        render_info(console, "kept in memory only — will vanish on exit.")

    return SlashResult.HANDLED


# ─── /tool dispatcher ───────────────────────────────────────────────────────


async def _handle_tool(rest: str, ctx: SessionContext, console: Console) -> SlashResult:
    parts = rest.split(maxsplit=1)
    sub = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    if sub in ("new", "create", "add"):
        return await _tool_new(ctx, console)
    if sub in ("rm", "remove", "del", "delete"):
        return await _tool_remove(arg, ctx, console)
    if sub in ("reload", "refresh"):
        return _tool_reload(ctx, console)
    if sub in ("open", "edit", "explore"):
        return _tool_open(arg, console)
    if sub in ("dir", "where", "path"):
        return _tool_dir(console)
    if sub in ("", "list", "ls"):
        from free_agent.cli.console import render_tools_inventory

        render_tools_inventory(console, TOOLS, DEEPAGENTS_BUILTINS)
        return SlashResult.HANDLED

    render_error(
        console,
        f"unknown /tool action: {sub!r}. try: new · rm <name> · reload · open · dir · list",
    )
    return SlashResult.HANDLED


async def _tool_new(ctx: SessionContext, console: Console) -> SlashResult:
    try:
        name = await create_tool_wizard(ctx, console)
    except (KeyboardInterrupt, EOFError):
        console.print()
        render_info(console, "wizard cancelled.")
        return SlashResult.HANDLED

    if name is None:
        return SlashResult.HANDLED

    # If the main agent has an explicit tool list (not None=all), offer to add the new tool to it.
    if ctx.profile.tools is not None and name not in ctx.profile.tools:
        try:
            ans = await ctx.prompt_session.prompt_async(
                f"  add [bold yellow1]{name}[/] to the main agent's tools? [Y/n] "
            )
        except (KeyboardInterrupt, EOFError):
            ans = "n"
        if ans.strip().lower() in ("", "y", "yes"):
            ctx.profile.tools.append(name)

    # Rebuild so the agent sees the new tool.
    try:
        ctx.rebuild_agent()
    except Exception as exc:
        render_error(console, f"rebuild failed: {exc}")
        return SlashResult.HANDLED

    # Show what the main agent actually has bound, so the user can verify
    # the new tool reached it (vs. just sitting in the registry).
    if ctx.profile.tools is None:
        agent_tools = [t.name for t in TOOLS]
        scope_note = "all registered tools"
    else:
        agent_tools = list(ctx.profile.tools)
        scope_note = "main agent's explicit list"
    in_agent = name in agent_tools
    marker = "[bold bright_green]✓[/]" if in_agent else "[bold red1]✗[/]"
    render_info(
        console,
        f"tool online — [bold yellow1]{name}[/] {marker} reachable by the agent.\n"
        f"[grey50]agent has {len(agent_tools)} tool(s) ({scope_note}): "
        f"{', '.join(agent_tools[:8])}"
        + (f" … (+{len(agent_tools) - 8} more)" if len(agent_tools) > 8 else "")
        + "[/]",
    )
    if not in_agent:
        render_info(
            console,
            "[bold]heads-up:[/] the tool is registered but [bold red1]not in "
            "the agent's tools list[/]. run [bold]/agent[/] to inspect, or edit "
            "your workspace's [bold]free-agent.yaml[/] to add it.",
        )
    return SlashResult.HANDLED


async def _tool_remove(name: str, ctx: SessionContext, console: Console) -> SlashResult:
    if not name:
        render_error(console, "usage: /tool rm <name>")
        return SlashResult.HANDLED

    if not is_user_tool(name):
        render_error(
            console,
            f"{name!r} is not a user-created tool — refusing to remove. "
            "(built-in tools live in the package; edit code to change them.)",
        )
        return SlashResult.HANDLED

    path = origin_of(name)
    if path is None or not path.exists():
        render_error(console, f"could not locate the file for {name!r}.")
        return SlashResult.HANDLED

    try:
        ans = await ctx.prompt_session.prompt_async(
            f"  delete {path.name} and unregister {name!r}? [y/N] "
        )
    except (KeyboardInterrupt, EOFError):
        ans = "n"
    if ans.strip().lower() not in ("y", "yes"):
        render_info(console, "kept.")
        return SlashResult.HANDLED

    path.unlink()
    reload_tools()

    # If main agent profile referenced it explicitly, drop it from the list.
    if ctx.profile.tools is not None and name in ctx.profile.tools:
        ctx.profile.tools = [t for t in ctx.profile.tools if t != name]

    # Same for any subagent that referenced it.
    for sa in ctx.profile.subagents:
        if sa.tools and name in sa.tools:
            sa.tools = [t for t in sa.tools if t != name]

    try:
        ctx.rebuild_agent()
    except Exception as exc:
        render_error(console, f"rebuild failed after removal: {exc}")
        return SlashResult.HANDLED

    render_info(console, f"tool [bold yellow1]{name}[/] removed.")
    return SlashResult.HANDLED


def _tool_open(scope: str, console: Console) -> SlashResult:
    """Open the user-tools folder in the OS file manager."""
    s = scope.lower().strip() if scope else "workspace"
    if s in ("p", "project", "local", "ws", "workspace"):
        targets = [user_tools_dir()]
    elif s in ("g", "global"):
        targets = [global_tools_dir()]
    elif s in ("both", "all", "*"):
        targets = [user_tools_dir(), global_tools_dir()]
    else:
        render_error(
            console,
            f"unknown scope: {scope!r}. use workspace · global · both.",
        )
        return SlashResult.HANDLED

    for target in targets:
        _open_in_file_manager(target, console)
    return SlashResult.HANDLED


def _tool_dir(console: Console) -> SlashResult:
    """Print where the tools folders live without opening anything."""
    project = user_tools_dir()
    global_ = global_tools_dir()
    proj_state = "exists" if project.is_dir() else "not yet created"
    glob_state = "exists" if global_.is_dir() else "not yet created"
    render_info(console, f"workspace tools: [bold bright_cyan]{project}[/]  [grey50]({proj_state})[/]")
    render_info(console, f"global    tools: [bold bright_cyan]{global_}[/]  [grey50]({glob_state})[/]")
    return SlashResult.HANDLED


def _open_in_file_manager(path: Path, console: Console) -> None:
    """Open `path` in the OS file manager. Creates the directory if missing."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        render_error(console, f"could not create {path}: {exc}")
        return

    resolved = path.resolve()
    cmd = _file_manager_command()
    if cmd is None:
        render_info(
            console,
            f"path: [bold bright_cyan]{resolved}[/]  "
            "[grey50](no GUI file manager available — open it manually)[/]",
        )
        return

    try:
        subprocess.Popen(  # noqa: S603 — args list, no shell
            [*cmd, str(resolved)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        render_info(console, f"opening [bold bright_cyan]{resolved}[/] ...")
    except FileNotFoundError:
        render_info(
            console,
            f"could not run [bold]{cmd[0]}[/] — open manually: "
            f"[bold bright_cyan]{resolved}[/]",
        )
    except OSError as exc:
        render_error(console, f"failed to open {resolved}: {exc}")


def _file_manager_command() -> list[str] | None:
    if sys.platform == "darwin":
        return ["open"]
    if sys.platform == "win32":
        return ["explorer"]
    if shutil.which("xdg-open"):
        return ["xdg-open"]
    return None


def _tool_reload(ctx: SessionContext, console: Console) -> SlashResult:
    try:
        loaded = reload_tools()
    except Exception as exc:
        render_error(console, f"reload failed: {exc}")
        return SlashResult.HANDLED
    try:
        ctx.rebuild_agent()
    except Exception as exc:
        render_error(console, f"rebuild failed: {exc}")
        return SlashResult.HANDLED
    render_info(
        console,
        f"reloaded — {len(loaded)} user tool(s): "
        f"{', '.join(loaded) if loaded else '(none)'}",
    )
    return SlashResult.HANDLED


async def _sub_remove(name: str, ctx: SessionContext, console: Console) -> SlashResult:
    if not name:
        render_error(console, "usage: /sub rm <name>")
        return SlashResult.HANDLED

    before = len(ctx.profile.subagents)
    ctx.profile.subagents = [sa for sa in ctx.profile.subagents if sa.name != name]
    if len(ctx.profile.subagents) == before:
        render_error(
            console,
            f"no subagent named {name!r}. existing: "
            f"{[sa.name for sa in ctx.profile.subagents]}",
        )
        return SlashResult.HANDLED

    try:
        ctx.rebuild_agent()
    except Exception as exc:
        render_error(console, f"rebuild failed: {exc}")
        return SlashResult.HANDLED

    render_info(console, f"subagent [bold bright_magenta]{name}[/] removed.")
    return SlashResult.HANDLED


# ─── /skill dispatcher ──────────────────────────────────────────────────────


async def _handle_skill(rest: str, ctx: SessionContext, console: Console) -> SlashResult:
    parts = rest.split(maxsplit=1)
    sub = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    if sub in ("new", "create", "add"):
        return await _skill_new(ctx, console)
    if sub in ("rm", "remove", "del", "delete"):
        return await _skill_remove(arg, ctx, console)
    if sub in ("reload", "refresh"):
        return _skill_reload(ctx, console)
    if sub in ("open", "edit", "explore"):
        return _skill_open(arg, console)
    if sub in ("dir", "where", "path"):
        return _skill_dir(console)
    if sub in ("", "list", "ls"):
        return _skill_list(console)

    render_error(
        console,
        f"unknown /skill action: {sub!r}. try: new · rm <name> · reload · open · dir · list",
    )
    return SlashResult.HANDLED


async def _skill_new(ctx: SessionContext, console: Console) -> SlashResult:
    try:
        name = await create_skill_wizard(ctx, console)
    except (KeyboardInterrupt, EOFError):
        console.print()
        render_info(console, "wizard cancelled.")
        return SlashResult.HANDLED

    if name is None:
        return SlashResult.HANDLED

    try:
        ctx.rebuild_agent()
    except Exception as exc:
        render_error(console, f"rebuild failed: {exc}")
        return SlashResult.HANDLED
    render_info(console, f"skill [bold yellow1]{name}[/] online.")
    return SlashResult.HANDLED


async def _skill_remove(name: str, ctx: SessionContext, console: Console) -> SlashResult:
    if not name:
        render_error(console, "usage: /skill rm <name>")
        return SlashResult.HANDLED

    matches = [s for s in list_skills() if s.name == name]
    if not matches:
        render_error(
            console,
            f"no skill named {name!r}. existing: {[s.name for s in list_skills()]}",
        )
        return SlashResult.HANDLED

    skill = matches[0]
    skill_dir = skill.path.parent
    try:
        ans = await ctx.prompt_session.prompt_async(
            f"  delete {skill_dir} (and all files inside)? [y/N] "
        )
    except (KeyboardInterrupt, EOFError):
        ans = "n"
    if ans.strip().lower() not in ("y", "yes"):
        render_info(console, "kept.")
        return SlashResult.HANDLED

    import shutil

    try:
        shutil.rmtree(skill_dir)
    except OSError as exc:
        render_error(console, f"could not remove: {exc}")
        return SlashResult.HANDLED

    try:
        ctx.rebuild_agent()
    except Exception as exc:
        render_error(console, f"rebuild failed after removal: {exc}")
        return SlashResult.HANDLED

    render_info(console, f"skill [bold yellow1]{name}[/] removed.")
    return SlashResult.HANDLED


def _skill_reload(ctx: SessionContext, console: Console) -> SlashResult:
    try:
        ctx.rebuild_agent()
    except Exception as exc:
        render_error(console, f"rebuild failed: {exc}")
        return SlashResult.HANDLED
    skills = list_skills()
    render_info(
        console,
        f"reloaded — {len(skills)} skill(s): "
        f"{', '.join(s.name for s in skills) if skills else '(none)'}",
    )
    return SlashResult.HANDLED


def _skill_open(scope: str, console: Console) -> SlashResult:
    s = scope.lower().strip() if scope else "workspace"
    if s in ("p", "project", "local", "ws", "workspace"):
        targets = [project_skills_dir()]
    elif s in ("g", "global"):
        targets = [global_skills_dir()]
    elif s in ("both", "all", "*"):
        targets = [project_skills_dir(), global_skills_dir()]
    else:
        render_error(console, f"unknown scope: {scope!r}. use workspace · global · both.")
        return SlashResult.HANDLED
    for target in targets:
        _open_in_file_manager(target, console)
    return SlashResult.HANDLED


def _skill_dir(console: Console) -> SlashResult:
    project = project_skills_dir()
    global_ = global_skills_dir()
    proj_state = "exists" if project.is_dir() else "not yet created"
    glob_state = "exists" if global_.is_dir() else "not yet created"
    render_info(console, f"workspace skills: [bold bright_cyan]{project}[/]  [grey50]({proj_state})[/]")
    render_info(console, f"global    skills: [bold bright_cyan]{global_}[/]  [grey50]({glob_state})[/]")
    return SlashResult.HANDLED


def _skill_list(console: Console) -> SlashResult:
    from free_agent.cli.console import render_skills_inventory

    render_skills_inventory(console, list_skills())
    return SlashResult.HANDLED


# ─── /model dispatcher ──────────────────────────────────────────────────────


async def _handle_model(rest: str, ctx: SessionContext, console: Console) -> SlashResult:
    parts = rest.split(maxsplit=1)
    sub = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    if sub in ("", "list", "ls"):
        return _model_list(ctx, console)
    if sub in ("browse", "catalog", "search", "available"):
        return _model_browse(arg, ctx, console)
    if sub == "pull":
        return await _model_pull(arg, ctx, console)
    if sub in ("rm", "remove", "del", "delete"):
        return await _model_remove(arg, ctx, console)
    if sub in ("use", "switch", "set"):
        return await _model_use(arg, ctx, console)

    render_error(
        console,
        f"unknown /model action: {sub!r}. try: list · pull <name> · rm <name> · use <name>",
    )
    return SlashResult.HANDLED


def _require_ollama(ctx: SessionContext, console: Console, action: str) -> bool:
    if ctx.settings.provider != "ollama":
        render_error(
            console,
            f"/model {action} only applies to provider=ollama "
            f"(active provider: {ctx.settings.provider}).",
        )
        return False
    return True


def _model_list(ctx: SessionContext, console: Console) -> SlashResult:
    if not _require_ollama(ctx, console, "list"):
        return SlashResult.HANDLED
    try:
        models = _ollama_list(ctx.settings.ollama_base_url)
    except RuntimeError as exc:
        render_error(console, str(exc))
        return SlashResult.HANDLED
    render_model_list(
        console,
        models,
        current=ctx.settings.ollama_model,
        base_url=ctx.settings.ollama_base_url,
    )
    return SlashResult.HANDLED


def _model_browse(query: str, ctx: SessionContext, console: Console) -> SlashResult:
    if not _require_ollama(ctx, console, "browse"):
        return SlashResult.HANDLED

    # Argument grammar:
    #   /model browse              → tool-capable only
    #   /model browse all          → include non-tool-capable
    #   /model browse refresh      → bypass cache, re-scrape
    #   /model browse <substring>  → filter (works alongside the above)
    show_all = False
    force_refresh = False
    tokens = query.split()
    leftover: list[str] = []
    for t in tokens:
        if t.lower() == "all":
            show_all = True
        elif t.lower() in ("refresh", "fresh", "--no-cache"):
            force_refresh = True
        else:
            leftover.append(t)
    text_query = " ".join(leftover)

    # Fetch the library (with cache).
    try:
        with console.status(
            "[bright_magenta]» fetching ollama.com/library...[/]", spinner="dots"
        ):
            entries, source = cached_library(force_refresh=force_refresh)
    except RuntimeError as exc:
        render_error(
            console,
            f"could not load library ({exc}). showing curated fallback.",
        )
        return _model_browse_curated(text_query, ctx, console)

    pulled: set[str] = set()
    try:
        pulled = {m["name"] for m in _ollama_list(ctx.settings.ollama_base_url)}
    except RuntimeError:
        pass
    pulled_bases = {p.split(":", 1)[0] for p in pulled}

    # Apply text filter, then capability filter.
    filtered = filter_library(entries, text_query)
    hidden_count = 0
    if not show_all:
        hidden_count = sum(1 for e in filtered if not e.supports_tools)
        filtered = [e for e in filtered if e.supports_tools]

    # Sort: active → pulled → tool-capable → others, alphabetical within each.
    active_base = ctx.settings.ollama_model.split(":", 1)[0]

    def _sort_key(e: LibraryEntry) -> tuple[int, str]:
        if e.name == active_base:
            return (0, e.name)
        if e.name in pulled_bases:
            return (1, e.name)
        if e.supports_tools:
            return (2, e.name)
        return (3, e.name)

    filtered.sort(key=_sort_key)

    render_model_library(
        console,
        filtered,
        pulled=pulled,
        pulled_bases=pulled_bases,
        active=ctx.settings.ollama_model,
        query=text_query,
        source=source,
        show_all=show_all,
        hidden_count=hidden_count,
    )
    return SlashResult.HANDLED


def _model_browse_curated(
    query: str, ctx: SessionContext, console: Console
) -> SlashResult:
    """Offline fallback when ollama.com/library can't be reached."""
    pulled: set[str] = set()
    try:
        pulled = {m["name"] for m in _ollama_list(ctx.settings.ollama_base_url)}
    except RuntimeError:
        pass
    pulled_bases = {p.split(":", 1)[0] for p in pulled}

    # Convert curated CatalogEntry → LibraryEntry shape so we reuse the renderer.
    fake_entries = [
        LibraryEntry(
            name=e.name.split(":", 1)[0],
            description=e.blurb,
            capabilities=("tools",),
            sizes=(e.name.split(":", 1)[1],) if ":" in e.name else (),
        )
        for e in _filter_curated(query)
    ]
    render_model_library(
        console,
        fake_entries,
        pulled=pulled,
        pulled_bases=pulled_bases,
        active=ctx.settings.ollama_model,
        query=query,
        source="offline / curated",
        show_all=True,
        hidden_count=0,
    )
    return SlashResult.HANDLED


async def _model_pull(name: str, ctx: SessionContext, console: Console) -> SlashResult:
    if not _require_ollama(ctx, console, "pull"):
        return SlashResult.HANDLED
    if not name:
        render_error(console, "usage: /model pull <name>  (e.g. qwen3.5:9b)")
        return SlashResult.HANDLED

    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    render_info(console, f"pulling [bold yellow1]{name}[/] from ollama...")

    progress = Progress(
        TextColumn("[bold bright_magenta]{task.description}[/]"),
        BarColumn(bar_width=40, complete_style="bright_magenta", finished_style="bright_green"),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    task_ids: dict[str, int] = {}
    last_status = ""

    try:
        with progress:
            async for ev in _ollama_pull(ctx.settings.ollama_base_url, name):
                status = str(ev.get("status", ""))
                digest = ev.get("digest")
                total = ev.get("total")
                completed = ev.get("completed", 0) or 0

                key = digest or status
                if total and key not in task_ids:
                    task_ids[key] = progress.add_task(status[:30], total=total)

                if total and key in task_ids:
                    progress.update(task_ids[key], completed=completed, description=status[:30])
                elif status and status != last_status:
                    progress.console.print(f"  [grey50]» {status}[/]")
                    last_status = status
    except RuntimeError as exc:
        render_error(console, str(exc))
        return SlashResult.HANDLED

    render_info(console, f"pulled [bold yellow1]{name}[/].")

    # Offer to switch.
    try:
        ans = await ctx.prompt_session.prompt_async(
            f"  switch active model to [bold yellow1]{name}[/]? [y/N] "
        )
    except (KeyboardInterrupt, EOFError):
        ans = "n"
    if ans.strip().lower() in ("y", "yes"):
        try:
            await asyncio.to_thread(ctx.switch_model, name)
        except Exception as exc:
            render_error(console, f"switch failed: {exc}")
            return SlashResult.HANDLED
        render_info(console, f"now using [bold yellow1]{name}[/].")

    return SlashResult.HANDLED


async def _model_remove(name: str, ctx: SessionContext, console: Console) -> SlashResult:
    if not _require_ollama(ctx, console, "rm"):
        return SlashResult.HANDLED
    if not name:
        render_error(console, "usage: /model rm <name>")
        return SlashResult.HANDLED

    if name == ctx.settings.ollama_model:
        render_error(
            console,
            f"{name!r} is the active model. switch first with /model use <other>.",
        )
        return SlashResult.HANDLED

    try:
        ans = await ctx.prompt_session.prompt_async(
            f"  remove [bold yellow1]{name}[/] from disk? [y/N] "
        )
    except (KeyboardInterrupt, EOFError):
        ans = "n"
    if ans.strip().lower() not in ("y", "yes"):
        render_info(console, "kept.")
        return SlashResult.HANDLED

    try:
        await asyncio.to_thread(_ollama_delete, ctx.settings.ollama_base_url, name)
    except RuntimeError as exc:
        render_error(console, str(exc))
        return SlashResult.HANDLED
    render_info(console, f"removed [bold yellow1]{name}[/].")
    return SlashResult.HANDLED


def _infer_provider(name: str) -> str:
    """Guess provider from a model name. Anthropic models all start with
    `claude-`; everything else is treated as Ollama (where any string can
    be a valid tag, e.g. `qwen3.5:9b`, `mistral:latest`, `llama3:8b`)."""
    return "anthropic" if name.lower().startswith("claude-") else "ollama"


def _collect_pickable_models(ctx: SessionContext) -> list[dict]:
    """Build the rows for the interactive picker.

    Order: locally-pulled Ollama models first (alphabetical), then the
    curated Anthropic catalog. Each row is enabled iff it can actually
    be used right now — Anthropic rows get disabled when no API key is
    configured, with a clear note.
    """
    from free_agent.cli.settings_panel import ANTHROPIC_MODELS

    rows: list[dict] = []

    # Ollama — only models actually pulled (no point offering ones that
    # would trigger a fresh download from a "use" picker; for that the
    # user has /model pull or /model browse).
    try:
        ollama_models = _ollama_list(ctx.settings.ollama_base_url)
    except RuntimeError:
        ollama_models = []

    for m in sorted(ollama_models, key=lambda x: x["name"]):
        size = m.get("size_bytes") or 0
        units = ["B", "KB", "MB", "GB", "TB"]
        val = float(size)
        i = 0
        while val >= 1024 and i < len(units) - 1:
            val /= 1024
            i += 1
        size_label = f"{val:.1f} {units[i]}" if size else "—"
        rows.append({
            "name": m["name"],
            "provider": "ollama",
            "note": size_label,
            "enabled": True,
        })

    # Anthropic — always show the curated catalog; mark disabled when
    # no key is present so the user understands why a pick would fail.
    key = ctx.settings.anthropic_api_key
    has_key = key is not None and key.get_secret_value().strip() != ""
    for name, blurb in ANTHROPIC_MODELS:
        rows.append({
            "name": name,
            "provider": "anthropic",
            "note": blurb if has_key else f"{blurb}  ·  needs api key (/settings)",
            "enabled": has_key,
        })

    # Number them 1-based.
    for idx, row in enumerate(rows, start=1):
        row["index"] = idx
    return rows


async def _model_use_interactive(
    ctx: SessionContext, console: Console
) -> SlashResult:
    """Show the picker and dispatch to the same switch path as `_model_use`."""
    from free_agent.cli.console import render_model_picker

    rows = _collect_pickable_models(ctx)
    if not rows:
        render_error(
            console,
            "no models available — pull one with /model pull <name> or "
            "configure an Anthropic key via /settings.",
        )
        return SlashResult.HANDLED

    render_model_picker(
        console,
        rows,
        active_name=ctx.settings.active_model,
        active_provider=ctx.settings.provider,
    )

    try:
        ans = await ctx.prompt_session.prompt_async(
            "  pick number ▶ "
        )
    except (KeyboardInterrupt, EOFError):
        render_info(console, "cancelled.")
        return SlashResult.HANDLED

    ans = ans.strip()
    if not ans:
        render_info(console, "cancelled.")
        return SlashResult.HANDLED

    try:
        idx = int(ans)
    except ValueError:
        # Friendly fallback: let the user type the model name too.
        return await _model_use(ans, ctx, console)

    if not 1 <= idx <= len(rows):
        render_error(console, f"invalid pick {idx} — choose 1..{len(rows)}.")
        return SlashResult.HANDLED

    pick = rows[idx - 1]
    if not pick["enabled"]:
        render_error(
            console,
            f"{pick['name']!r} is not selectable: {pick['note']}",
        )
        return SlashResult.HANDLED

    return await _model_use(pick["name"], ctx, console)


async def _model_use(name: str, ctx: SessionContext, console: Console) -> SlashResult:
    if not name:
        return await _model_use_interactive(ctx, console)

    target_provider = _infer_provider(name)

    if (
        name == ctx.settings.active_model
        and target_provider == ctx.settings.provider
    ):
        render_info(console, f"already using {name}.")
        return SlashResult.HANDLED

    # Cross-provider switch: validate prerequisites BEFORE we touch state.
    if target_provider != ctx.settings.provider:
        if target_provider == "anthropic":
            key = ctx.settings.anthropic_api_key
            if key is None or not key.get_secret_value().strip():
                render_error(
                    console,
                    f"{name!r} is an Anthropic model but no API key is set. "
                    "configure it via [bold bright_cyan]/settings[/] (it gets "
                    "saved to ~/.config/free-agent/secrets.json), or "
                    "`export ANTHROPIC_API_KEY=…`.",
                )
                return SlashResult.HANDLED
            render_info(
                console,
                f"switching provider [grey50]{ctx.settings.provider}[/] → "
                f"[bold]anthropic[/] for this model.",
            )

    # Ollama-target: if not pulled, offer to pull it. Skip when we're
    # crossing INTO Ollama from Anthropic — let the preflight inside
    # switch_model surface the missing-tag error itself, since prompting
    # to pull mid-switch is more complexity than it's worth.
    if (
        target_provider == "ollama"
        and ctx.settings.provider == "ollama"
    ):
        try:
            available = {m["name"] for m in _ollama_list(ctx.settings.ollama_base_url)}
        except RuntimeError as exc:
            render_error(console, str(exc))
            return SlashResult.HANDLED
        if name not in available:
            try:
                ans = await ctx.prompt_session.prompt_async(
                    f"  {name!r} is not pulled. pull it now? [Y/n] "
                )
            except (KeyboardInterrupt, EOFError):
                ans = "n"
            if ans.strip().lower() not in ("", "y", "yes"):
                render_info(console, "cancelled.")
                return SlashResult.HANDLED
            result = await _model_pull(name, ctx, console)
            # _model_pull already offered the switch; if user said yes there, we're done.
            if ctx.settings.active_model == name:
                return result

    try:
        await asyncio.to_thread(
            ctx.switch_model, name, provider=target_provider
        )
    except Exception as exc:
        render_error(console, f"switch failed: {exc}")
        return SlashResult.HANDLED
    render_info(
        console,
        f"now using [bold yellow1]{name}[/] @ {ctx.settings.provider}.",
    )
    return SlashResult.HANDLED


# ─── shared rendering ───────────────────────────────────────────────────────


def _render_profile(ctx: SessionContext, console: Console) -> None:
    from free_agent.agent.prompts import SYSTEM_PROMPT

    profile = ctx.profile
    available_tool_names = [t.name for t in TOOLS]
    main_tools = (
        list(profile.tools) if profile.tools is not None else available_tool_names
    )

    subagents = []
    for sa in profile.subagents:
        subagents.append(
            {
                "name": sa.name,
                "description": sa.description,
                "system_prompt": sa.system_prompt,
                "tools": list(sa.tools) if sa.tools is not None else None,
            }
        )

    render_agent_profile(
        console,
        main_model=ctx.settings.active_model,
        main_provider=ctx.settings.provider,
        main_system_prompt=profile.system_prompt or SYSTEM_PROMPT,
        main_tools=main_tools,
        subagents=subagents,
        config_path=str(ctx.config_path) if ctx.config_path else None,
        writable_root=str(ctx.writable_root) if ctx.writable_root else None,
    )
