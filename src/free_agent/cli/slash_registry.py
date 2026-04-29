"""Single source of truth for slash commands + the prompt_toolkit completer.

Listing here is what the completion menu shows when the user types `/`.
Keep entries in roughly the same order as the `/help` panel.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

if TYPE_CHECKING:
    from free_agent.cli.context import SessionContext

# (command, description) — description shows as `display_meta` in the menu.
SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help",         "show all commands"),
    ("/plan",         "force planning: /plan <task>  (agent will write_todos first)"),
    ("/tools",        "tool inventory (PKG vs USER)"),
    ("/agent",        "current agent profile (main + subagents)"),

    ("/sub new",      "wizard — create a subagent (LLM drafts the prompt)"),
    ("/sub rm",       "remove a subagent: /sub rm <name>"),
    ("/sub list",     "alias for /agent"),

    ("/tool new",     "wizard — create a tool (LLM writes the Python)"),
    ("/tool rm",      "delete a user tool: /tool rm <name>"),
    ("/tool reload",  "re-scan tool folders without restart"),
    ("/tool open",    "open tools folder: [project | global | both]"),
    ("/tool dir",     "print tool folder paths"),

    ("/skill list",   "list every loaded skill (SKILL.md folders)"),
    ("/skill new",    "wizard — create a skill (LLM drafts the SKILL.md)"),
    ("/skill rm",     "delete a skill folder: /skill rm <name>"),
    ("/skill reload", "rebuild the agent so it sees skill changes"),
    ("/skill open",   "open skills folder: [project | global | both]"),
    ("/skill dir",    "print skill folder paths"),

    ("/model list",   "list local Ollama models"),
    ("/model browse", "browse ollama.com/library [query | all | refresh]"),
    ("/model pull",   "download a model: /model pull <name>"),
    ("/model use",    "switch active model: /model use <name>"),
    ("/model rm",     "remove a local model: /model rm <name>"),

    ("/clear",        "wipe the conversation buffer"),
    ("/history",      "dump the session as markdown"),
    ("/save",         "export session: /save [path]"),
    ("/retry",        "drop last AI turn and re-fire previous prompt"),
    ("/exit",         "quit (Ctrl+D works too)"),
]


# Per-prefix dynamic value providers — keys are matched as `text.startswith(...)`.
# Each provider returns an iterable of (value, meta) tuples. They run on every
# keystroke, so they MUST be cheap (in-memory only — no network).
_DynamicProvider = Callable[["SessionContext"], Iterable[tuple[str, str]]]


def _subagent_names(ctx: "SessionContext") -> Iterable[tuple[str, str]]:
    return ((sa.name, _truncate(sa.description, 60)) for sa in ctx.profile.subagents)


def _user_tool_names(ctx: "SessionContext") -> Iterable[tuple[str, str]]:
    from free_agent.tools import TOOLS, is_user_tool, origin_of

    out: list[tuple[str, str]] = []
    for t in TOOLS:
        if not is_user_tool(t.name):
            continue
        origin = origin_of(t.name)
        meta = origin.name if origin else "user tool"
        out.append((t.name, meta))
    return out


def _skill_names(ctx: "SessionContext") -> Iterable[tuple[str, str]]:
    from free_agent.agent.skills_registry import list_skills

    return ((s.name, f"{s.scope} · {_truncate(s.description, 50)}") for s in list_skills())


_DYNAMIC: dict[str, _DynamicProvider] = {
    "/sub rm ":      _subagent_names,
    "/sub remove ":  _subagent_names,
    "/tool rm ":     _user_tool_names,
    "/tool remove ": _user_tool_names,
    "/skill rm ":    _skill_names,
    "/skill remove ": _skill_names,
}


class SlashCompleter(Completer):
    """Yield slash command completions when the line starts with `/`.

    `ctx_provider` is a zero-arg callable that returns the current
    SessionContext (or None if it isn't built yet). Used for dynamic
    completion of arguments to commands like `/sub rm <name>`.
    """

    def __init__(
        self,
        commands: list[tuple[str, str]] | None = None,
        ctx_provider: Callable[[], "SessionContext | None"] | None = None,
    ) -> None:
        self.commands = commands or SLASH_COMMANDS
        self.ctx_provider = ctx_provider

    def get_completions(self, document: Document, complete_event: object):  # type: ignore[override]
        text = document.text
        if not text.startswith("/"):
            return

        # Dynamic argument completion (after the command + space).
        for prefix, provider in _DYNAMIC.items():
            if text.startswith(prefix):
                ctx = self.ctx_provider() if self.ctx_provider else None
                if ctx is None:
                    return
                partial = text[len(prefix):]
                for value, meta in provider(ctx):
                    if value.startswith(partial):
                        yield Completion(
                            value,
                            start_position=-len(partial),
                            display=value,
                            display_meta=meta,
                        )
                return  # do not also show the command list

        # Static: every command whose full path starts with what the user typed.
        for cmd, desc in self.commands:
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=cmd,
                    display_meta=desc,
                )


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"
