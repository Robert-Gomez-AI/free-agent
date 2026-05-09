from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import ollama
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.skills import SkillsMiddleware
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama

from free_agent.agent.profile import AgentProfile, SubAgentProfile
from free_agent.agent.prompts import SYSTEM_PROMPT
from free_agent.agent.skills_registry import discover_skill_sources, list_skills
from free_agent.config import Settings
from free_agent.tools import TOOLS


def make_chat_model(settings: Settings) -> BaseChatModel:
    """Build a fresh chat model. Performs the Ollama preflight (network)."""
    if settings.provider == "ollama":
        _preflight_ollama(settings.ollama_base_url, settings.ollama_model)
        chat = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            num_predict=settings.max_tokens,
            # gpt-oss / qwen3 / deepseek-r1 use harmony / channel-based output
            # where reasoning lives in a separate channel from the final answer
            # and tool-calls in yet another. With `reasoning=True`, the langchain
            # wrapper routes those into `additional_kwargs.reasoning_content`
            # and `tool_calls` instead of dumping them into `content` as raw
            # text. Safe for non-reasoning models — they just don't emit those
            # channels and `content` works as before.
            reasoning=True,
            # qwen3.5:9b's bundled renderer occasionally fails to emit EOS at
            # turn end and starts hallucinating fake `User said: ...` /
            # `Assistant: ...` exchanges as raw text inside the response. That
            # garbage gets persisted into the conversation history and poisons
            # subsequent turns (the model "remembers" its own hallucinated
            # turns instead of the real ones). These stop tokens cut the
            # generation as soon as a role-label leak appears. Harmless for
            # well-behaved models — they never emit these strings.
            stop=[
                "<|im_start|>",
                "<|im_end|>",
                "\nUser said:",
                "\nuser said:",
                "\nUser:",
                "\nuser:",
                "\nHuman:",
                "\nhuman:",
            ],
        )
        # gpt-oss / harmony-template models choke on tool JSON schemas that
        # include `anyOf` (the langchain-default representation of `Optional[X]`
        # parameters). Wrap the outgoing client so we strip those at the wire.
        # Harmless for non-fragile templates — they accept the simpler form too.
        _patch_ollama_client_for_oss(chat)
        return chat
    if settings.provider == "anthropic":
        assert settings.anthropic_api_key is not None  # validated in Settings
        return ChatAnthropic(
            model=settings.anthropic_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            api_key=settings.anthropic_api_key.get_secret_value(),
        )
    raise ValueError(f"Unknown provider: {settings.provider!r}")


_MEMORY_REMINDER = """\

# ─── Final reminder — read carefully ───
The conversation history of this session is already loaded into your context \
as previous turns visible above this final system block. Scroll up and read \
the prior user/assistant turns directly — they are already in front of you.

CRITICAL distinction: the filesystem tools (`ls`, `read_file`, `glob`, \
`grep`, `write_file`, `edit_file`) operate on a VIRTUAL filesystem that has \
NOTHING to do with the conversation history. Do NOT call any tool to "fetch" \
or "search for" prior turns — those tools cannot see chat messages. The chat \
history lives in the message turns above this system prompt; just look at \
them with your own attention.

When the user says "antes", "what did I say", "ese código", or asks anything \
that depends on earlier turns, look at the visible message turns above and \
answer from them. Never claim to lack memory of past turns when those turns \
appear above, and never use a filesystem tool to look for them.\
"""


def _patch_ollama_client_for_oss(chat: Any) -> None:
    """Wrap the chat instance's async client.chat with two fixes:

    1. Sanitize `tools` schemas — flatten `anyOf` to a single concrete type.
       Required for gpt-oss / harmony-template models whose Go templating
       fails on properties with empty `Type` arrays.

    2. Append a memory-anchor block to the trailing system message. The
       deepagents framework appends a large "Deep Agent" system prompt that
       biases the model into stateless-task mode, making it deny access to
       the `messages` history even when the history is right there. Sticking
       the reminder at the END (after deepagents' prose) is the only place
       it reliably lands in the model's recency buffer.
    """
    async_client = getattr(chat, "_async_client", None)
    if async_client is None:
        return
    original_chat = getattr(async_client, "chat", None)
    if original_chat is None or getattr(original_chat, "_oss_patched", False):
        return

    async def patched_chat(*args: Any, **kwargs: Any) -> Any:
        tools = kwargs.get("tools")
        if tools:
            kwargs["tools"] = [_sanitize_tool_anyof(t) for t in tools]
        messages = kwargs.get("messages")
        if isinstance(messages, list) and messages:
            msgs = _append_memory_reminder(messages)
            msgs = _inject_user_hint(msgs)
            kwargs["messages"] = msgs
        return await original_chat(*args, **kwargs)

    patched_chat._oss_patched = True  # type: ignore[attr-defined]
    async_client.chat = patched_chat


def _append_memory_reminder(messages: list[Any]) -> list[Any]:
    """Append the memory reminder to the trailing system message, in place.

    Returns a new list (shallow-copied) with the last system message's
    content extended. If there's no system message (rare), prepends one.
    Idempotent: skips if the reminder is already present.
    """
    out = list(messages)
    # Find the last system message.
    sys_idx = -1
    for i in range(len(out) - 1, -1, -1):
        m = out[i]
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "system":
            sys_idx = i
            break

    if sys_idx == -1:
        out.insert(0, {"role": "system", "content": _MEMORY_REMINDER.strip()})
        return out

    msg = out[sys_idx]
    is_dict = isinstance(msg, dict)
    content = msg.get("content") if is_dict else getattr(msg, "content", "")
    content_str = content if isinstance(content, str) else str(content)
    if "─── Final reminder" in content_str:
        return out  # already appended

    new_content = content_str + _MEMORY_REMINDER
    if is_dict:
        out[sys_idx] = {**msg, "content": new_content}
    else:
        # Best-effort for non-dict shapes — avoid mutating in place.
        try:
            out[sys_idx] = type(msg)(role="system", content=new_content)
        except Exception:
            out[sys_idx] = {"role": "system", "content": new_content}
    return out


_USER_HINT_MARKER = "[turno "


def _inject_user_hint(messages: list[Any]) -> list[Any]:
    """Prepend a continuity hint to the last user message (Ollama path only).

    The system-message reminder lands in a low-attention zone for smaller
    Ollama models — deepagents' large "Deep Agent" prose biases them toward
    treating each turn as an isolated task, even with the reminder appended.
    The last user message, by contrast, is always in the model's recency
    buffer with maximum attention.

    On turns 2+, prepend a one-block tag that names the turn number and
    explicitly anchors back-references ("antes", "vuelve a hacerlo",
    "ese código"). On turn 1 there's no continuity to anchor — skip.

    Idempotent: skips when the marker is already at the head of the message
    (re-entrant calls leave the payload unchanged).

    Returns a shallow copy with the last user message rewritten.
    """
    out = list(messages)
    user_count = 0
    last_user_idx = -1
    for i, m in enumerate(out):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "user":
            user_count += 1
            last_user_idx = i

    if user_count <= 1 or last_user_idx < 0:
        return out

    msg = out[last_user_idx]
    is_dict = isinstance(msg, dict)
    content = msg.get("content") if is_dict else getattr(msg, "content", "")
    content_str = content if isinstance(content, str) else str(content)
    if content_str.lstrip().startswith(_USER_HINT_MARKER):
        return out

    # Build a compact snapshot of the previous turns, so the hint anchors the
    # model to concrete content instead of an abstract "history exists"
    # claim. Weak Ollama models (qwen3.5:9b confirmed) reject the abstract
    # claim and answer "no veo historial" — quoting the actual prior turns
    # in the hint forces them to engage with what's actually there.
    snippets: list[str] = []
    for m in out[:last_user_idx]:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role not in ("user", "assistant"):
            continue
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        text = (content if isinstance(content, str) else str(content)).strip()
        if not text:
            continue
        # Trim to one line, max 140 chars
        line = text.split("\n", 1)[0]
        if len(line) > 140:
            line = line[:137] + "…"
        snippets.append(f"  {role}: {line}")

    snapshot = "\n".join(snippets[-6:])  # last 3 user/assistant pairs is plenty
    hint = (
        f"[turno {user_count} · arriba TIENES el historial completo de esta "
        f"sesión, visible como mensajes user/assistant previos. estos son los "
        f"últimos turnos:\n{snapshot}\n\n"
        f"NO digas que no ves el historial — está justo ahí arriba. NO llames "
        f"ninguna tool (ls/grep/read_file/etc) para buscarlo — esas tools "
        f"trabajan sobre un filesystem virtual, no sobre los mensajes. solo "
        f"léelos con tu atención y responde.]\n\n"
    )
    new_content = hint + content_str
    if is_dict:
        out[last_user_idx] = {**msg, "content": new_content}
    else:
        try:
            out[last_user_idx] = type(msg)(role="user", content=new_content)
        except Exception:
            out[last_user_idx] = {"role": "user", "content": new_content}
    return out


class _StripTaskToolMiddleware(AgentMiddleware[Any, Any, Any]):
    """Hide deepagents' ``task`` subagent-spawner from the model.

    Why: deepagents always wires a ``general-purpose`` subagent and binds
    a ``task`` tool whose description is ~5 KB of XML example prose. On
    weaker open-source models (qwen3.5:9b confirmed) this single tool
    description is enough to flip the model into chat mode — it stops
    emitting structured ``tool_calls`` and starts dumping markdown JSON or
    code blocks for *any* tool, even simple ones like ``shell`` or
    ``python_interpreter``. Stripping ``task`` from the model request
    AND from the system prompt restores tool calling. Used only when the
    profile declares no subagents (in which case the tool is dead weight).
    """

    _TASK_SECTIONS = (
        "## `task` (subagent spawner)",
        "## Important Task Tool Usage Notes to Remember",
        "Available subagent types:",
    )

    @staticmethod
    def _tool_name(tool: Any) -> str | None:
        if isinstance(tool, dict):
            n = tool.get("name") or tool.get("function", {}).get("name") if "function" in tool else tool.get("name")
            return n if isinstance(n, str) else None
        return getattr(tool, "name", None)

    @classmethod
    def _strip_task_prose(cls, prompt: str) -> str:
        """Remove every section that references the ``task`` tool from the prompt.

        Scans for the known section headers deepagents injects and trims
        each one (and everything after it, up to the next ``##`` heading).
        """
        if not prompt:
            return prompt
        out = prompt
        for marker in cls._TASK_SECTIONS:
            idx = out.find(marker)
            if idx < 0:
                continue
            # Find the next "## " heading after this section. If none,
            # cut to end of prompt.
            tail_start = idx + len(marker)
            next_h = out.find("\n## ", tail_start)
            if next_h < 0:
                out = out[:idx].rstrip()
            else:
                out = (out[:idx] + out[next_h + 1:]).rstrip()
        return out

    def _filter_request(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        tools = [t for t in (request.tools or []) if self._tool_name(t) != "task"]
        new_prompt = self._strip_task_prose(request.system_prompt or "")
        return request.override(tools=tools, system_prompt=new_prompt)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(self._filter_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        return await handler(self._filter_request(request))


def _sanitize_tool_anyof(node: Any) -> Any:
    """Recursively rewrite ``{"anyOf": [...]}`` into a single-type branch.

    Preserves ``title`` / ``description`` when present. Picks the first
    non-null branch (or the first branch outright if all are null).
    """
    if isinstance(node, dict):
        if "anyOf" in node and isinstance(node["anyOf"], list) and node["anyOf"]:
            branches = [b for b in node["anyOf"] if isinstance(b, dict)]
            non_null = [b for b in branches if b.get("type") != "null"]
            chosen = (non_null or branches)[0] if branches else {"type": "string"}
            preserved = {
                k: node[k] for k in ("title", "description", "default") if k in node
            }
            cleaned = {**chosen}
            for k, v in preserved.items():
                cleaned.setdefault(k, v)
            # Recurse into the chosen branch in case it nests another anyOf.
            return _sanitize_tool_anyof(cleaned)
        return {k: _sanitize_tool_anyof(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_sanitize_tool_anyof(item) for item in node]
    return node


def assemble_agent(
    model: BaseChatModel,
    profile: AgentProfile,
    *,
    writable_root: Path | None = None,
) -> Any:
    """Wire a deepagents graph from an existing model + profile.

    Cheap (no network) — safe to call on every profile mutation.

    If `writable_root` is set, the agent's filesystem tools (read_file,
    write_file, ls, edit_file) operate on the real disk rooted there, with
    `virtual_mode=True` blocking traversal outside the root. Otherwise the
    default in-memory StateBackend is used (current behavior).
    """
    available: dict[str, BaseTool] = {t.name: t for t in TOOLS}
    main_tools = _resolve_tools(profile.tools, available, scope="main agent")
    subagents = [_build_subagent_spec(sa, available) for sa in profile.subagents]
    system_prompt = profile.system_prompt or SYSTEM_PROMPT
    # deepagents' base prompt enumerates its own tools (write_todos, ls,
    # read_file, …) but says NOTHING about the user's tools — those reach the
    # model only via the JSON `tools=` schema. Weaker open-source models
    # (qwen3.5, llama3:8b, gemma) often ignore JSON-only tools and answer
    # directly or dump markdown code blocks. Naming them in prose puts them
    # back on the model's radar.
    system_prompt = system_prompt + _render_tools_manifest(main_tools)

    backend: BackendProtocol | None = None
    if writable_root is not None:
        backend = FilesystemBackend(
            root_dir=str(writable_root.resolve()),
            virtual_mode=True,
        )

    # Skills get their OWN backend rooted at "/", so absolute paths to
    # ~/.config/free-agent/skills/ work even when the main backend is
    # the in-memory StateBackend or a cwd-scoped FilesystemBackend.
    #
    # IMPORTANT: only attach SkillsMiddleware when at least one SKILL.md
    # is actually present. Otherwise the middleware injects a verbose
    # "(No skills available yet) ... use read_file ..." prose block into
    # the system prompt, which trains many open-source models in-context
    # to format tool calls as markdown JSON instead of structured
    # tool_calls. With no skills to gain, the prose is pure regression.
    extra_middleware = []
    skill_sources = discover_skill_sources()
    if skill_sources and list_skills():
        skills_backend = FilesystemBackend(root_dir="/", virtual_mode=False)
        extra_middleware.append(
            SkillsMiddleware(backend=skills_backend, sources=skill_sources)
        )

    # When the profile defines no subagents, deepagents still auto-wires a
    # `general-purpose` subagent + the `task` spawner tool. That tool ships
    # with a ~5 KB XML/example description that breaks tool-calling on
    # smaller models (qwen3.5:9b confirmed — flips them into markdown mode
    # for ALL tools, not just task). Strip it.
    if not subagents:
        extra_middleware.append(_StripTaskToolMiddleware())

    return create_deep_agent(
        model=model,
        tools=main_tools,
        system_prompt=system_prompt,
        subagents=subagents or None,
        backend=backend,
        middleware=extra_middleware,
    )


def build_session(
    settings: Settings,
    profile: AgentProfile | None = None,
    *,
    writable_root: Path | None = None,
) -> tuple[BaseChatModel, Any]:
    """One-shot: make_chat_model + assemble_agent. Used at boot."""
    if profile is None:
        profile = AgentProfile.default()
    model = make_chat_model(settings)
    agent = assemble_agent(model, profile, writable_root=writable_root)
    return model, agent


# ─── helpers ────────────────────────────────────────────────────────────────


def _render_tools_manifest(tools: list[BaseTool]) -> str:
    """Format the user-provided tools as a prose block for the system prompt.

    Returns an empty string when there are no tools, so the prompt stays
    untouched in that case. Each entry shows the tool name + the first line
    of its docstring (deepagents' base prompt also lists tools as a flat
    name list, so we follow that style for consistency).
    """
    if not tools:
        return ""
    lines = ["", "", "## Available tools"]
    lines.append(
        "You have these additional tools beyond the deepagents built-ins. "
        "CALL them via tool_calls when relevant — do NOT just write code or "
        "shell commands inline as markdown:"
    )
    for t in tools:
        first_line = (t.description or "").strip().split("\n", 1)[0]
        lines.append(f"- `{t.name}` — {first_line}")
    return "\n".join(lines)


def _resolve_tools(
    names: list[str] | None,
    available: dict[str, BaseTool],
    *,
    scope: str,
) -> list[BaseTool]:
    if names is None:
        return list(available.values())
    resolved: list[BaseTool] = []
    missing: list[str] = []
    for n in names:
        if n in available:
            resolved.append(available[n])
        else:
            missing.append(n)
    if missing:
        raise ValueError(
            f"{scope}: tool(s) not registered: {missing}. "
            f"Registered tools: {sorted(available)}. "
            f"Add them in tools/basic.py."
        )
    return resolved


def _build_subagent_spec(
    profile: SubAgentProfile,
    available: dict[str, BaseTool],
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "name": profile.name,
        "description": profile.description,
        "system_prompt": profile.system_prompt,
    }
    if profile.tools is not None:
        spec["tools"] = _resolve_tools(
            profile.tools, available, scope=f"subagent '{profile.name}'"
        )
    return spec


def _preflight_ollama(base_url: str, model: str) -> None:
    client = ollama.Client(host=base_url)
    try:
        resp = client.list()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {base_url}. Is the daemon running? "
            f"Try `ollama serve`. ({exc})"
        ) from exc

    available = _extract_model_names(resp)
    if model in available:
        return

    listing = "\n".join(f"  - {n}" for n in available) if available else "  (none pulled)"
    raise RuntimeError(
        f"Ollama model {model!r} is not pulled locally.\n"
        f"Models available on this host:\n{listing}\n"
        f"Either run `ollama pull {model}` or set FREE_AGENT_OLLAMA_MODEL "
        f"in .env to one of the above."
    )


def _extract_model_names(resp: Any) -> list[str]:
    models = getattr(resp, "models", None)
    if models is None and isinstance(resp, dict):
        models = resp.get("models", [])
    names: list[str] = []
    for m in models or []:
        name = getattr(m, "model", None)
        if name is None and isinstance(m, dict):
            name = m.get("model") or m.get("name")
        if name:
            names.append(name)
    return names


# Back-compat alias — older callers.
def build_agent(settings: Settings, profile: AgentProfile | None = None) -> Any:
    _, agent = build_session(settings, profile)
    return agent
