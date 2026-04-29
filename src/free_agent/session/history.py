from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

Role = Literal["user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


@dataclass
class Conversation:
    """In-memory chat state. Plain types only — no LangChain leakage above the agent layer."""

    messages: list[Message] = field(default_factory=list)

    def append_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def append_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def pop_last(self) -> Message | None:
        return self.messages.pop() if self.messages else None

    def clear(self) -> None:
        self.messages.clear()

    def last_user_message(self) -> str | None:
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                return msg["content"]
        return None

    def to_payload(self) -> dict[str, list[Message]]:
        """Shape expected by deepagents/LangGraph."""
        return {"messages": list(self.messages)}

    def to_markdown(self) -> str:
        lines = ["# free-agent session\n"]
        for msg in self.messages:
            who = "**you**" if msg["role"] == "user" else "**agent**"
            lines.append(f"{who}\n\n{msg['content']}\n")
        return "\n".join(lines)
