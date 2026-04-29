"""Cyberpunk-flavored rendering for the terminal chat UI.

Palette: neon magenta + cyan + acid yellow + matrix green on black.
Glyphs:  ▰ ▱ ◆ ▶ ◢ ◣ ░ ▒ ▓ ╔═╗║╚╝ ━━ ▔
"""
from __future__ import annotations

import asyncio
import random
from typing import Any

from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

# ─── theme ──────────────────────────────────────────────────────────────────

NEON = Theme(
    {
        "neon.magenta": "bold bright_magenta",
        "neon.cyan": "bold bright_cyan",
        "neon.green": "bold bright_green",
        "neon.yellow": "bold yellow1",
        "neon.red": "bold red1",
        "neon.purple": "bold magenta",
        "agent.prefix": "bold bright_magenta",
        "user.prefix": "bold bright_cyan",
        "tool.exec": "yellow1",
        "tool.recv": "bright_green",
        "subagent": "bright_magenta",
        "border": "bright_magenta",
        "border.alt": "bright_cyan",
        "info": "cyan",
        "ghost": "grey50",
        "fault": "bold red1 on grey3",
        "boot": "bold bright_green",
        "tag": "bold black on bright_magenta",
        "tag.cyan": "bold black on bright_cyan",
        "tag.yellow": "bold black on yellow1",
        "tag.green": "bold black on bright_green",
        "tag.red": "bold white on red1",
    }
)


def make_console() -> Console:
    return Console(theme=NEON, highlight=False)


# ─── banner ─────────────────────────────────────────────────────────────────

_ASCII = r"""
█▀▀ █▀█ █▀▀ █▀▀   ▄▀█ █▀▀ █▀▀ █▄ █ ▀█▀
█▀  █▀▄ ██▄ ██▄   █▀█ █▄█ ██▄ █ ▀█  █
"""

# Pool of glyphs we substitute in during the glitch decode.
_GLITCH_GLYPHS = "░▒▓█▖▗▘▙▚▛▜▝▞▟▤▥▦▧▨▩◆◇◈◉◊@#%&*+-=<>"

# Corruption schedule for the boot decode — eased, with a deliberate "glitch back" near the end.
_CORRUPTION_SCHEDULE = [
    0.95, 0.92, 0.85, 0.75, 0.62, 0.48, 0.55, 0.32, 0.18, 0.06, 0.0,
]


def _corrupt_text(text: str, ratio: float) -> str:
    if ratio <= 0:
        return text
    out = []
    for ch in text:
        if ch in (" ", "\n"):
            out.append(ch)
        elif random.random() < ratio:
            out.append(random.choice(_GLITCH_GLYPHS))
        else:
            out.append(ch)
    return "".join(out)


def _build_banner_panel(model: str, provider: str, *, art_corruption: float = 0.0) -> Panel:
    lines = _ASCII.strip("\n").splitlines()
    art = Text()
    for i, line in enumerate(lines):
        rendered = _corrupt_text(line, art_corruption) if art_corruption > 0 else line
        style = "bright_magenta" if i == 0 else "bright_cyan"
        art.append(rendered + "\n", style=f"bold {style}")

    tagline = Text()
    tagline.append("░▒▓ ", style="bright_magenta")
    tagline.append("neural shell", style="bold bright_magenta")
    tagline.append(" :: ", style="grey50")
    tagline.append("deepagents runtime", style="bold bright_cyan")
    tagline.append(" :: ", style="grey50")
    tagline.append("zero leash", style="bold bright_green")
    tagline.append(" ▓▒░", style="bright_magenta")

    status = Text()
    status.append(" NET ", style="tag.green")
    status.append(" online   ", style="bright_green")
    status.append(" MODEL ", style="tag.cyan")
    status.append(f" {model}   ", style="bold bright_cyan")
    status.append(" LINK ", style="tag")
    status.append(f" {provider} ", style="bold bright_magenta")

    hint = Text(
        "type /help · Ctrl+C aborts a turn · Ctrl+D severs the link",
        style="grey50",
    )

    body = Text()
    body.append(art)
    body.append("\n")
    body.append(tagline)
    body.append("\n\n")
    body.append(status)
    body.append("\n")
    body.append(hint)

    return Panel(
        Align.left(body),
        border_style="bright_magenta",
        padding=(1, 2),
        title="[bold bright_magenta]▓▒░ FREE-AGENT ░▒▓[/]",
        subtitle="[grey50]v0.1.0 // open source[/]",
        title_align="left",
        subtitle_align="right",
        expand=False,
    )


def render_banner(console: Console, model: str, provider: str) -> None:
    """Static banner — used as a fallback when the terminal isn't interactive."""
    console.print(_build_banner_panel(model, provider))


async def render_banner_glitch(
    console: Console,
    model: str,
    provider: str,
    *,
    frame_delay: float = 0.055,
) -> None:
    """Animate the banner decoding from random glitch noise into the clean form."""
    if not console.is_terminal:
        render_banner(console, model, provider)
        return

    with Live(console=console, refresh_per_second=30, transient=False) as live:
        for corruption in _CORRUPTION_SCHEDULE:
            live.update(_build_banner_panel(model, provider, art_corruption=corruption))
            await asyncio.sleep(frame_delay)
        live.update(_build_banner_panel(model, provider, art_corruption=0.0))


# ─── chat turns ─────────────────────────────────────────────────────────────


def render_user_prefix(console: Console) -> None:
    pass  # input prompt handles this side


def render_assistant_prefix(console: Console) -> None:
    console.print("[agent.prefix]◆ ai[/] [grey50]▶[/] ", end="")


def stream_token(console: Console, text: str) -> None:
    """Print a streamed token verbatim. No markdown render mid-stream — avoids flicker."""
    console.out(text, end="", highlight=False, style=None)


def end_stream(console: Console) -> None:
    console.print()


# ─── tool / subagent traces ─────────────────────────────────────────────────


def render_tool_call(console: Console, name: str, inputs: dict[str, Any] | None = None) -> None:
    args = _summarize_inputs(inputs)
    console.print(
        f"  [tool.exec]▰[/] [tag.yellow] EXEC [/] "
        f"[tool.exec]{name}[/][grey50]{args}[/]"
    )


def render_tool_result(console: Console, name: str, output: Any) -> None:
    summary = _summarize_output(output)
    if not summary:
        return
    console.print(
        f"  [tool.recv]▱[/] [tag.green] RECV [/] "
        f"[grey50]{name} ←[/] [tool.recv]{summary}[/]"
    )


def render_subagent(console: Console, name: str) -> None:
    console.print(f"  [subagent]☰[/] [tag] LINK [/] [subagent]{name}[/] [grey50]engaged[/]")


def render_todos(console: Console, todos: list[dict[str, Any]]) -> None:
    """Render the current plan (deepagents `write_todos` payload) as a checklist."""
    if not todos:
        return
    completed = sum(1 for t in todos if (t.get("status") or "") == "completed")
    in_progress = sum(1 for t in todos if (t.get("status") or "") == "in_progress")
    total = len(todos)
    console.print()
    console.print(
        f"  [tag] ▦ PLAN [/] [grey50]{completed}/{total} done"
        + (f"  ·  {in_progress} in progress" if in_progress else "")
        + "[/]"
    )
    for t in todos:
        content = (t.get("content") or "").strip()
        status = (t.get("status") or "pending").lower()
        if status == "completed":
            console.print(
                f"    [bright_green]✓[/] [strike grey50]{content}[/]"
            )
        elif status == "in_progress":
            console.print(
                f"    [bold yellow1]▶[/] [bold yellow1]{content}[/]"
            )
        else:
            console.print(f"    [grey50]○[/] [grey70]{content}[/]")


def render_model_list(
    console: Console,
    models: list[dict[str, Any]],
    *,
    current: str,
    base_url: str,
) -> None:
    """Render the list of locally-pulled Ollama models, marking the active one."""
    from rich.console import Group
    from rich.table import Table

    sections: list[Any] = []
    sections.append(Text())
    header = Text()
    header.append(" ▰ MODELS ", style="tag.cyan")
    header.append(f"  ollama @ {base_url}", style="grey50")
    sections.append(header)

    if not models:
        sections.append(
            Text(
                "  (none pulled — try /model pull qwen3.5:9b)",
                style="grey50",
            )
        )
    else:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold bright_cyan", no_wrap=True)  # marker
        grid.add_column(style="bold yellow1", no_wrap=True)  # name
        grid.add_column(style="grey70", no_wrap=True)  # size
        grid.add_column(style="grey50")  # modified
        for m in sorted(models, key=lambda x: x["name"]):
            marker = "▶" if m["name"] == current else " "
            grid.add_row(
                f"  {marker}",
                m["name"],
                _human_size(m["size_bytes"]),
                _human_modified(m.get("modified")),
            )
        sections.append(grid)

    sections.append(Text())
    sections.append(
        Text(
            "  /model pull <name> · /model rm <name> · /model use <name>",
            style="grey50",
        )
    )

    console.print(
        Panel(
            Group(*sections),
            border_style="bright_cyan",
            title="[tag.cyan] ▓▒░ MODEL CACHE ░▒▓ [/]",
            title_align="left",
            padding=(0, 2),
        )
    )


def _human_size(n: int) -> str:
    if not n:
        return "—"
    units = ["B", "KB", "MB", "GB", "TB"]
    val = float(n)
    i = 0
    while val >= 1024 and i < len(units) - 1:
        val /= 1024
        i += 1
    return f"{val:.1f} {units[i]}"


def _human_modified(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    # Trim noisy fractional seconds + timezone if present
    if "T" in s:
        s = s.split(".", 1)[0].replace("T", " ")
    return s


def render_model_library(
    console: Console,
    entries: list[Any],
    *,
    pulled: set[str],
    pulled_bases: set[str],
    active: str,
    query: str = "",
    source: str = "",
    show_all: bool = False,
    hidden_count: int = 0,
) -> None:
    """Render scraped ollama.com/library entries with status markers."""
    from rich.console import Group
    from rich.table import Table

    sections: list[Any] = []
    sections.append(Text())
    header = Text()
    header.append(" ▰ LIBRARY ", style="tag.yellow")
    bits = ["ollama.com/library"]
    if source:
        bits.append(source)
    if query:
        bits.append(f"filter='{query}'")
    header.append("  " + "  ·  ".join(bits), style="grey50")
    sections.append(header)

    if not entries:
        sections.append(
            Text(
                f"  no entries match '{query}'. try /model browse without args.",
                style="grey50",
            )
        )
    else:
        # Compute name column width from the longest visible name (bounded).
        name_w = min(max((len(e.name) for e in entries), default=12), 22)

        for e in entries:
            base = e.name
            line = Text()
            line.append("  ")
            if active.split(":", 1)[0] == base:
                line.append("▶ ", style="bold bright_magenta")
            elif base in pulled_bases:
                line.append("✓ ", style="bold bright_green")
            else:
                line.append("○ ", style="grey50")

            display_name = base if len(base) <= name_w else base[: name_w - 1] + "…"
            line.append(display_name.ljust(name_w + 2), style="bold yellow1")
            line.append(_caps_text(e.capabilities))
            if e.sizes:
                line.append("  ")
                line.append(" · ".join(e.sizes), style="grey70")
            sections.append(line)

        # Legend
        legend = Text("  ")
        legend.append("▶", style="bold bright_magenta")
        legend.append(" active   ", style="grey50")
        legend.append("✓", style="bold bright_green")
        legend.append(" pulled   ", style="grey50")
        legend.append("○", style="grey50")
        legend.append(" available", style="grey50")
        sections.append(Text())
        sections.append(legend)

    if hidden_count:
        sections.append(Text())
        sections.append(
            Text(
                f"  {hidden_count} non-tool-calling model(s) hidden — "
                "use /model browse all to include them",
                style="grey50",
            )
        )

    sections.append(Text())
    sections.append(
        Text(
            "  /model pull <name>[:<size>]   ·   /model browse all   ·   "
            "/model browse refresh",
            style="grey50",
        )
    )

    console.print(
        Panel(
            Group(*sections),
            border_style="bright_cyan",
            title="[tag.cyan] ▓▒░ OLLAMA LIBRARY ░▒▓ [/]",
            title_align="left",
            padding=(0, 2),
        )
    )


def _caps_text(caps: tuple[str, ...]) -> Text:
    """Render capability badges as a Text object (correct width measurement)."""
    out = Text()
    if not caps:
        return out
    for i, c in enumerate(caps):
        if i:
            out.append(" ")
        if c == "tools":
            out.append(" tools ", style="bold black on yellow1")
        elif c == "thinking":
            out.append(" think ", style="bold black on bright_cyan")
        elif c == "vision":
            out.append(" vis ", style="bold black on bright_magenta")
        elif c == "embedding":
            out.append(" emb ", style="bold black on grey70")
        else:
            out.append(f"[{c}]", style="grey50")
    return out


def render_skills_inventory(console: Console, skills: list[Any]) -> None:
    """Render the loaded SKILL.md folders, grouped by scope."""
    from rich.console import Group
    from rich.table import Table

    sections: list[Any] = []

    project = [s for s in skills if s.scope == "project"]
    global_ = [s for s in skills if s.scope == "global"]

    def _block(label_style: str, label: str, hint: str, items: list[Any]) -> None:
        sections.append(Text())
        header = Text()
        header.append(f" {label} ", style=label_style)
        header.append(f"  {hint}", style="grey50")
        sections.append(header)
        if items:
            grid = Table.grid(padding=(0, 2))
            grid.add_column(style="bold yellow1", no_wrap=True)
            grid.add_column(style="grey70")
            for s in items:
                grid.add_row(f"  {s.name}", _short_desc(s.description))
            sections.append(grid)
        else:
            sections.append(Text("  (none)", style="grey50"))

    _block(
        "tag.cyan",
        "▰ PROJECT",
        "loaded from ./free_agent_skills/",
        project,
    )
    _block(
        "tag.yellow",
        "▰ GLOBAL ",
        "loaded from ~/.config/free-agent/skills/",
        global_,
    )

    sections.append(Text())
    sections.append(
        Text(
            f"  total: {len(skills)} skill(s)  ·  /skill new to create one",
            style="grey50",
        )
    )

    console.print(
        Panel(
            Group(*sections),
            border_style="bright_cyan",
            title="[tag.cyan] ▓▒░ SKILL INVENTORY ░▒▓ [/]",
            title_align="left",
            padding=(0, 2),
        )
    )


def render_tools_inventory(
    console: Console,
    user_tools: list[Any],
    builtin_tools: list[tuple[str, str]],
) -> None:
    """Render a styled inventory of every tool the agent can call."""
    from rich.console import Group
    from rich.table import Table

    from free_agent.tools import is_user_tool, origin_of

    sections: list[Any] = []

    # ── tools registered in the package ───────────────────────────────────
    pkg_tools = [t for t in user_tools if not is_user_tool(t.name)]
    file_tools = [t for t in user_tools if is_user_tool(t.name)]

    sections.append(Text())
    header = Text()
    header.append(" ▰ PKG ", style="tag.yellow")
    header.append("  shipped in tools/basic.py", style="grey50")
    sections.append(header)
    if pkg_tools:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold yellow1", no_wrap=True)
        grid.add_column(style="grey70")
        for tool in pkg_tools:
            sig = _format_signature(tool)
            desc = _short_desc(getattr(tool, "description", ""))
            grid.add_row(f"  {sig}", desc)
        sections.append(grid)
    else:
        sections.append(Text("  (none)", style="grey50"))

    # ── tools loaded from disk (project or global) ────────────────────────
    from free_agent.tools import global_tools_dir, user_tools_dir

    project_dir = user_tools_dir()
    global_dir = global_tools_dir()

    def _scope_label(path: Any) -> str:
        if path is None:
            return ""
        try:
            if path.is_relative_to(project_dir):
                return "project"
            if path.is_relative_to(global_dir):
                return "global"
        except (ValueError, AttributeError):
            pass
        return "user"

    sections.append(Text())
    header = Text()
    header.append(" ▰ USER ", style="tag.cyan")
    header.append(
        f"  project={project_dir}  ·  global={global_dir}",
        style="grey50",
    )
    sections.append(header)
    if file_tools:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold bright_cyan", no_wrap=True)
        grid.add_column(style="grey70")
        grid.add_column(style="bright_magenta", no_wrap=True)
        for tool in file_tools:
            sig = _format_signature(tool)
            desc = _short_desc(getattr(tool, "description", ""))
            origin = origin_of(tool.name)
            scope = _scope_label(origin)
            grid.add_row(f"  {sig}", desc, f"[{scope}]" if scope else "")
        sections.append(grid)
    else:
        sections.append(
            Text(
                "  (none — create one with /tool new)",
                style="grey50",
            )
        )

    # ── deepagents builtins ───────────────────────────────────────────────
    sections.append(Text())
    header = Text()
    header.append(" ▱ CORE ", style="tag")
    header.append("  provided by deepagents middleware", style="grey50")
    sections.append(header)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold bright_magenta", no_wrap=True)
    grid.add_column(style="grey70")
    for name, desc in builtin_tools:
        grid.add_row(f"  {name}", desc)
    sections.append(grid)

    # ── footer ────────────────────────────────────────────────────────────
    sections.append(Text())
    footer = Text(
        f"total: {len(user_tools) + len(builtin_tools)} tools  ·  "
        "edit tools/basic.py to add your own",
        style="grey50",
    )
    sections.append(footer)

    console.print(
        Panel(
            Group(*sections),
            border_style="bright_cyan",
            title="[tag.cyan] ▓▒░ TOOL INVENTORY ░▒▓ [/]",
            title_align="left",
            padding=(0, 2),
        )
    )


def render_agent_profile(
    console: Console,
    *,
    main_model: str,
    main_provider: str,
    main_system_prompt: str,
    main_tools: list[str],
    subagents: list[dict[str, Any]],
    config_path: str | None,
    writable_root: str | None = None,
) -> None:
    """Render the loaded agent profile (main agent + subagents)."""
    from rich.console import Group
    from rich.table import Table

    sections: list[Any] = []

    # ── main agent ────────────────────────────────────────────────────────
    sections.append(Text())
    header = Text()
    header.append(" ▰ MAIN ", style="tag.yellow")
    header.append("  orchestrator", style="grey50")
    sections.append(header)

    main_grid = Table.grid(padding=(0, 2))
    main_grid.add_column(style="grey50", no_wrap=True)
    main_grid.add_column(style="bright_cyan")
    main_grid.add_row("  model", f"{main_model} [grey50]@[/] {main_provider}")
    main_grid.add_row("  prompt", _wrap_excerpt(main_system_prompt))
    main_grid.add_row(
        "  tools",
        ", ".join(main_tools) if main_tools else "[grey50](none)[/]",
    )
    if writable_root:
        main_grid.add_row(
            "  filesystem",
            f"[bold red1]WRITABLE[/] [grey50]→[/] [bold]{writable_root}[/]",
        )
    else:
        main_grid.add_row("  filesystem", "[grey50]virtual (in-memory state)[/]")
    sections.append(main_grid)

    # ── subagents ─────────────────────────────────────────────────────────
    sections.append(Text())
    header = Text()
    header.append(" ▱ SUBAGENTS ", style="tag")
    header.append(f"  {len(subagents)} configured", style="grey50")
    sections.append(header)

    if not subagents:
        sections.append(
            Text(
                "  (none — declare them under `subagents:` in free-agent.yaml)",
                style="grey50",
            )
        )
    else:
        for sa in subagents:
            sa_grid = Table.grid(padding=(0, 2))
            sa_grid.add_column(style="grey50", no_wrap=True)
            sa_grid.add_column(style="bright_magenta")
            sa_grid.add_row(
                f"  ◆ [bold bright_magenta]{sa['name']}[/]",
                f"[grey70]{sa.get('description', '')}[/]",
            )
            sa_grid.add_row(
                "    prompt",
                f"[grey70]{_wrap_excerpt(sa.get('system_prompt', ''))}[/]",
            )
            tools = sa.get("tools")
            if tools is None:
                tools_repr = "[grey50](inherits from main)[/]"
            elif not tools:
                tools_repr = "[grey50](none — pure reasoning)[/]"
            else:
                tools_repr = "[bright_magenta]" + ", ".join(tools) + "[/]"
            sa_grid.add_row("    tools", tools_repr)
            sections.append(sa_grid)
            sections.append(Text())

    # ── footer ────────────────────────────────────────────────────────────
    footer = Text()
    if config_path:
        footer.append("config: ", style="grey50")
        footer.append(config_path, style="bold bright_cyan")
    else:
        footer.append(
            "config: (defaults — create free-agent.yaml to customize)",
            style="grey50",
        )
    sections.append(footer)

    console.print(
        Panel(
            Group(*sections),
            border_style="bright_cyan",
            title="[tag.cyan] ▓▒░ AGENT PROFILE ░▒▓ [/]",
            title_align="left",
            padding=(0, 2),
        )
    )


def _wrap_excerpt(text: str, *, limit: int = 90) -> str:
    text = (text or "").strip()
    if not text:
        return "[grey50](none)[/]"
    first = text.split("\n", 1)[0].strip()
    if len(first) > limit:
        first = first[: limit - 1] + "…"
    return first


def _format_signature(tool: Any) -> str:
    """Build a `name(arg1: type = default, arg2: type)` string from a LangChain tool."""
    name = getattr(tool, "name", "tool")
    args = getattr(tool, "args", None) or {}
    if not isinstance(args, dict):
        return f"{name}()"
    parts = []
    for arg_name, schema in args.items():
        if not isinstance(schema, dict):
            parts.append(arg_name)
            continue
        t = schema.get("type") or ""
        default = schema.get("default", _MISSING)
        piece = f"{arg_name}: {t}" if t else arg_name
        if default is not _MISSING:
            piece += f" = {default!r}"
        parts.append(piece)
    return f"{name}({', '.join(parts)})"


def _short_desc(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "(no description)"
    first = text.split("\n", 1)[0].strip()
    return first.rstrip(".")


_MISSING = object()


# ─── system messages ────────────────────────────────────────────────────────


def render_error(console: Console, message: str) -> None:
    panel = Panel(
        f"[fault]{message}[/]",
        border_style="red1",
        title="[tag.red] ▣ FAULT [/]",
        title_align="left",
        padding=(0, 1),
    )
    console.print(panel)


def render_separator(console: Console) -> None:
    console.print(Rule(style="bright_magenta", characters="━"))


def render_info(console: Console, message: str) -> None:
    console.print(f"[grey50]»[/] [info]{message}[/]")


def render_boot_line(console: Console, tag: str, message: str, *, ok: bool = True) -> None:
    style = "tag.green" if ok else "tag.red"
    color = "bright_green" if ok else "bright_red"
    console.print(f"[{style}] {tag} [/] [{color}]{message}[/]")


async def type_boot_line(
    console: Console,
    tag: str,
    message: str,
    *,
    ok: bool = True,
    char_delay: float = 0.012,
) -> None:
    """Render a boot line with a per-character type-in effect."""
    if not console.is_terminal:
        render_boot_line(console, tag, message, ok=ok)
        return

    style = "tag.green" if ok else "tag.red"
    color = "bright_green" if ok else "bright_red"
    console.print(f"[{style}] {tag} [/] ", end="")
    for ch in message:
        console.print(ch, end="", style=color, highlight=False)
        await asyncio.sleep(char_delay)
    console.print()


async def boot_progress(console: Console, message: str, task: asyncio.Task[Any]) -> None:
    """Show shifting glitch noise + spinner while `task` runs. Vanishes when it completes."""
    if not console.is_terminal:
        await task
        return

    spinner = "▖▘▝▗"
    glitch_pool = "▒░▓█▤▥▦▧▨▩◆◇◈◉◊"

    with Live(console=console, refresh_per_second=18, transient=True) as live:
        i = 0
        while not task.done():
            noise = "".join(random.choice(glitch_pool) for _ in range(28))
            scan = "".join(random.choice("─━┄┅") for _ in range(12))
            line = Text()
            line.append(f"  {spinner[i % len(spinner)]} ", style="bold bright_magenta")
            line.append(f"{message}  ", style="bright_cyan")
            line.append(noise, style="grey50")
            line.append("  ")
            line.append(scan, style="bright_magenta")
            live.update(line)
            i += 1
            await asyncio.sleep(0.06)

    # Surface any exception from the task; no-op on success.
    await task


def render_aborted(console: Console) -> None:
    console.print("[tag.red] ▣ ABORTED [/] [grey50]turn cancelled by user[/]")


def render_disconnect(console: Console) -> None:
    console.print("[tag] ▣ LINK SEVERED [/] [grey50]see you on the grid.[/]")


# ─── markdown helper for /history, /help ────────────────────────────────────


def render_markdown_block(console: Console, text: str, title: str = "") -> None:
    from rich.markdown import Markdown

    panel = Panel(
        Markdown(text),
        border_style="bright_cyan",
        title=f"[tag.cyan] {title} [/]" if title else None,
        title_align="left",
        padding=(0, 1),
    )
    console.print(panel)


# ─── helpers ────────────────────────────────────────────────────────────────


def _summarize_inputs(inputs: dict[str, Any] | None) -> str:
    if not inputs:
        return "()"
    parts = []
    for k, v in inputs.items():
        s = repr(v)
        if len(s) > 40:
            s = s[:37] + "..."
        parts.append(f"{k}={s}")
    return "(" + ", ".join(parts) + ")"


def _summarize_output(output: Any) -> str:
    s = str(output) if output is not None else ""
    s = s.replace("\n", " ⏎ ")
    if len(s) > 90:
        s = s[:87] + "..."
    return s
