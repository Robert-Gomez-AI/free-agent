from __future__ import annotations

from pathlib import Path
from typing import Any

import ollama
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.skills import SkillsMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama

from free_agent.agent.profile import AgentProfile, SubAgentProfile
from free_agent.agent.prompts import SYSTEM_PROMPT
from free_agent.agent.skills_registry import discover_skill_sources
from free_agent.config import Settings
from free_agent.tools import TOOLS


def make_chat_model(settings: Settings) -> BaseChatModel:
    """Build a fresh chat model. Performs the Ollama preflight (network)."""
    if settings.provider == "ollama":
        _preflight_ollama(settings.ollama_base_url, settings.ollama_model)
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            num_predict=settings.max_tokens,
        )
    if settings.provider == "anthropic":
        assert settings.anthropic_api_key is not None  # validated in Settings
        return ChatAnthropic(
            model=settings.anthropic_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            api_key=settings.anthropic_api_key.get_secret_value(),
        )
    raise ValueError(f"Unknown provider: {settings.provider!r}")


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

    backend: BackendProtocol | None = None
    if writable_root is not None:
        backend = FilesystemBackend(
            root_dir=str(writable_root.resolve()),
            virtual_mode=True,
        )

    # Skills get their OWN backend rooted at "/", so absolute paths to
    # ~/.config/free-agent/skills/ work even when the main backend is
    # the in-memory StateBackend or a cwd-scoped FilesystemBackend.
    extra_middleware = []
    skill_sources = discover_skill_sources()
    if skill_sources:
        skills_backend = FilesystemBackend(root_dir="/", virtual_mode=False)
        extra_middleware.append(
            SkillsMiddleware(backend=skills_backend, sources=skill_sources)
        )

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
