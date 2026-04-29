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
# user having to nudge it.
CONTINUE_DIRECTIVE = """\
[plan execution — continue]

You wrote a plan but haven't finished executing it. Continue now: take action \
on the next pending or in-progress todo, then call `write_todos` again to \
update its status. Don't write a summary — actually do the next step. End \
only when every todo is "completed" AND the original task is answered.
"""


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
| `/model use <name>` | switch the active model in-session (rebuilds the agent) |
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

    render_info(
        console,
        f"tool online — [bold yellow1]{name}[/] available to the agent now.",
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
    s = scope.lower().strip() if scope else "project"
    if s in ("p", "project", "local"):
        targets = [user_tools_dir()]
    elif s in ("g", "global"):
        targets = [global_tools_dir()]
    elif s in ("both", "all", "*"):
        targets = [user_tools_dir(), global_tools_dir()]
    else:
        render_error(
            console,
            f"unknown scope: {scope!r}. use project · global · both.",
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
    render_info(console, f"project tools: [bold bright_cyan]{project}[/]  [grey50]({proj_state})[/]")
    render_info(console, f"global  tools: [bold bright_cyan]{global_}[/]  [grey50]({glob_state})[/]")
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
    s = scope.lower().strip() if scope else "project"
    if s in ("p", "project", "local"):
        targets = [project_skills_dir()]
    elif s in ("g", "global"):
        targets = [global_skills_dir()]
    elif s in ("both", "all", "*"):
        targets = [project_skills_dir(), global_skills_dir()]
    else:
        render_error(console, f"unknown scope: {scope!r}. use project · global · both.")
        return SlashResult.HANDLED
    for target in targets:
        _open_in_file_manager(target, console)
    return SlashResult.HANDLED


def _skill_dir(console: Console) -> SlashResult:
    project = project_skills_dir()
    global_ = global_skills_dir()
    proj_state = "exists" if project.is_dir() else "not yet created"
    glob_state = "exists" if global_.is_dir() else "not yet created"
    render_info(console, f"project skills: [bold bright_cyan]{project}[/]  [grey50]({proj_state})[/]")
    render_info(console, f"global  skills: [bold bright_cyan]{global_}[/]  [grey50]({glob_state})[/]")
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


async def _model_use(name: str, ctx: SessionContext, console: Console) -> SlashResult:
    if not name:
        render_error(console, "usage: /model use <name>")
        return SlashResult.HANDLED

    if name == ctx.settings.active_model:
        render_info(console, f"already using {name}.")
        return SlashResult.HANDLED

    # If Ollama and model not pulled, offer to pull it.
    if ctx.settings.provider == "ollama":
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
        await asyncio.to_thread(ctx.switch_model, name)
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
