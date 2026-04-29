"""Declarative agent profile loaded from `free-agent.yaml`.

Schema mirrors deepagents' `SubAgent` TypedDict so the YAML maps 1:1 to the
runtime spec without translation gymnastics.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SubAgentProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    system_prompt: str
    # None  → inherit main agent's tools
    # []    → no tools (pure reasoning)
    # [...] → subset by name
    tools: list[str] | None = None


class AgentProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # None → use the built-in default in agent/prompts.py
    system_prompt: str | None = None
    # None → expose every registered tool
    tools: list[str] | None = None
    subagents: list[SubAgentProfile] = Field(default_factory=list)

    @classmethod
    def default(cls) -> "AgentProfile":
        return cls()
