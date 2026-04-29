from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from prompt_toolkit import PromptSession

from free_agent.agent.builder import assemble_agent, make_chat_model
from free_agent.agent.profile import AgentProfile
from free_agent.config import Settings
from free_agent.session.history import Conversation


@dataclass
class SessionContext:
    """Mutable bundle of state shared across the chat loop and slash commands."""

    conversation: Conversation
    settings: Settings
    profile: AgentProfile
    chat_model: BaseChatModel
    agent: Any
    prompt_session: PromptSession[str]
    config_path: Path | None = None
    writable_root: Path | None = None

    def rebuild_agent(self) -> None:
        """Re-assemble the deepagents graph from current profile.

        Reuses the existing chat_model and writable_root so this is cheap (no network)."""
        self.agent = assemble_agent(
            self.chat_model, self.profile, writable_root=self.writable_root
        )

    def switch_model(self, new_model: str) -> None:
        """Swap the active model. Rebuilds chat_model (with preflight) and agent.

        Reverts on failure so the session keeps working with the previous model.
        """
        if self.settings.provider == "ollama":
            old = self.settings.ollama_model
            self.settings.ollama_model = new_model
        elif self.settings.provider == "anthropic":
            old = self.settings.anthropic_model
            self.settings.anthropic_model = new_model
        else:
            raise ValueError(f"unknown provider: {self.settings.provider!r}")

        try:
            self.chat_model = make_chat_model(self.settings)
            self.rebuild_agent()
        except Exception:
            if self.settings.provider == "ollama":
                self.settings.ollama_model = old
            else:
                self.settings.anthropic_model = old
            raise
