"""Base Agent class. Wraps a `GeminiChat` configured with the agent's tools."""

from __future__ import annotations

import abc
import json
from typing import Any

from researcher.core.models import AgentResponse
from researcher.infra.llm import GeminiChat, ToolBinding
from researcher.tools.base import Tool


class Agent(abc.ABC):
    """Owns a tool set and a Gemini chat configured with their schemas."""

    name: str
    system_instruction: str

    def __init__(self, model_name: str, tools: list[Tool], max_steps: int = 8) -> None:
        self._tools = {t.name: t for t in tools}
        bindings = [
            ToolBinding(
                declaration=t.function_declaration(),
                executor=self._tool_executor,
            )
            for t in tools
        ]
        self._chat = GeminiChat(
            model_name=model_name,
            system_instruction=self.system_instruction,
            bindings=bindings,
            max_steps=max_steps,
        )

    async def _tool_executor(self, name: str, args: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"Unknown tool {name!r}"}
        result = await tool.run(args)
        # Convert ToolResult to a small JSON-friendly dict for the model.
        payload: dict[str, Any] = {"ok": result.ok}
        if result.error is not None:
            payload["error"] = result.error
        if result.data is not None:
            payload["data"] = json.loads(result.data.model_dump_json())
        return payload

    async def handle(self, user_message: str) -> AgentResponse:
        """Run a one-shot request through the agent's tool loop."""

        outcome = await self._chat.run_loop(user_message)
        return AgentResponse(text=outcome.text, trace=outcome.trace)
