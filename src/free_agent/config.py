from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["ollama", "anthropic"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: Provider = Field("ollama", alias="FREE_AGENT_PROVIDER")

    ollama_model: str = Field("qwen2.5:7b", alias="FREE_AGENT_OLLAMA_MODEL")
    ollama_base_url: str = Field("http://localhost:11434", alias="FREE_AGENT_OLLAMA_BASE_URL")

    anthropic_model: str = Field("claude-sonnet-4-6", alias="FREE_AGENT_ANTHROPIC_MODEL")
    anthropic_api_key: SecretStr | None = Field(None, alias="ANTHROPIC_API_KEY")

    temperature: float = Field(0.7, alias="FREE_AGENT_TEMPERATURE")
    max_tokens: int = Field(4096, alias="FREE_AGENT_MAX_TOKENS")
    log_level: str = Field("WARNING", alias="FREE_AGENT_LOG_LEVEL")
    history_file: Path = Field(
        default_factory=lambda: Path.home() / ".free_agent_history",
        alias="FREE_AGENT_HISTORY_FILE",
    )

    # When True, the agent's read_file/write_file/ls/edit_file tools operate on
    # the real filesystem rooted at cwd (with virtual_mode guardrails — paths
    # cannot escape via .. / ~ / absolute outside-root). Defaults to False, in
    # which case files live in the in-memory deepagents StateBackend.
    writable: bool = Field(False, alias="FREE_AGENT_WRITABLE")

    @property
    def active_model(self) -> str:
        return self.ollama_model if self.provider == "ollama" else self.anthropic_model

    @model_validator(mode="after")
    def _check_provider_credentials(self) -> Settings:
        if self.provider == "anthropic":
            key = self.anthropic_api_key
            if key is None or not key.get_secret_value().strip():
                raise ValueError(
                    "ANTHROPIC_API_KEY is required when FREE_AGENT_PROVIDER=anthropic"
                )
        return self
