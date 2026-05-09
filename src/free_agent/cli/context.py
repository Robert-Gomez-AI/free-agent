from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from prompt_toolkit import PromptSession

from free_agent.agent.builder import assemble_agent, make_chat_model
from free_agent.agent.loader import find_config, load_profile
from free_agent.agent.profile import AgentProfile
from free_agent.config import Settings
from free_agent.session.history import Conversation
from free_agent.tools import reload_tools
from free_agent.workspace import Workspace, bind_active, set_active


@dataclass
class SessionContext:
    """Mutable bundle of state shared across the chat loop and slash commands."""

    conversation: Conversation
    settings: Settings
    profile: AgentProfile
    chat_model: BaseChatModel
    agent: Any
    prompt_session: PromptSession[str]
    workspace: Workspace
    config_path: Path | None = None
    writable_root: Path | None = None

    def rebuild_agent(self) -> None:
        """Re-assemble the deepagents graph from current profile.

        Reuses the existing chat_model and writable_root so this is cheap (no network)."""
        self.agent = assemble_agent(
            self.chat_model, self.profile, writable_root=self.writable_root
        )

    def switch_model(self, new_model: str, *, provider: str | None = None) -> None:
        """Swap the active model. Rebuilds chat_model (with preflight) and agent.

        If `provider` is given (and differs from the current one), also flips
        the provider — used for cross-provider switches like
        `/model use claude-opus-4-7` from an Ollama session.

        Reverts every mutation on failure so the session keeps working with
        the previous configuration.
        """
        old_provider = self.settings.provider
        old_ollama = self.settings.ollama_model
        old_anthropic = self.settings.anthropic_model

        target_provider = provider or self.settings.provider
        if target_provider not in ("ollama", "anthropic"):
            raise ValueError(f"unknown provider: {target_provider!r}")

        self.settings.provider = target_provider  # type: ignore[assignment]
        if target_provider == "ollama":
            self.settings.ollama_model = new_model
        else:
            self.settings.anthropic_model = new_model

        try:
            self.chat_model = make_chat_model(self.settings)
            self.rebuild_agent()
        except Exception:
            self.settings.provider = old_provider  # type: ignore[assignment]
            self.settings.ollama_model = old_ollama
            self.settings.anthropic_model = old_anthropic
            raise

        # Persist on success — same convention as the /settings panel, so
        # the next session boots into whatever the user just selected.
        try:
            from free_agent.config import save_user_settings

            save_user_settings(self.settings)
        except OSError as exc:
            log = __import__("logging").getLogger(__name__)
            log.warning("model switch applied but could not be persisted: %s", exc)

    def switch_workspace(self, ws: Workspace) -> None:
        """Activate a different workspace. Reloads profile + tools + skills.

        Reverts on failure so the session keeps working with the previous one.
        """
        old_ws = self.workspace
        old_profile = self.profile
        old_config_path = self.config_path

        bind_active(ws)
        try:
            set_active(ws.name)
            reload_tools()
            new_config = find_config()
            new_profile = load_profile(new_config)
            self.workspace = ws
            self.config_path = new_config
            self.profile = new_profile
            self.rebuild_agent()
        except Exception:
            bind_active(old_ws)
            try:
                set_active(old_ws.name)
            except Exception:
                pass
            self.workspace = old_ws
            self.profile = old_profile
            self.config_path = old_config_path
            reload_tools()
            try:
                self.rebuild_agent()
            except Exception:
                pass
            raise
