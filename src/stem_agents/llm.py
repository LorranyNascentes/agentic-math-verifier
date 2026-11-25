"""LLM abstraction. Agents depend on the ``LLM`` protocol, never on a vendor SDK."""
from __future__ import annotations

import os
from typing import Protocol


class LLM(Protocol):
    def complete(self, role: str, system: str, user: str) -> str: ...


class ScriptedLLM:
    """Deterministic LLM for tests/demos/CI: replays canned replies per agent role."""

    def __init__(self, script: dict[str, list[str]]):
        self.script = script
        self.calls: list[tuple[str, str]] = []
        self._cursor: dict[str, int] = {}

    def complete(self, role: str, system: str, user: str) -> str:
        self.calls.append((role, user))
        replies = self.script[role]
        i = self._cursor.get(role, 0)
        self._cursor[role] = i + 1
        return replies[min(i, len(replies) - 1)]  # repeat last reply when exhausted


class AnthropicLLM:
    """Thin adapter over the Anthropic SDK (``pip install anthropic``)."""

    def __init__(self, model: str | None = None, max_tokens: int = 4000):
        import anthropic  # lazy: optional dependency

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or os.environ.get("STEM_AGENTS_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

    def complete(self, role: str, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
