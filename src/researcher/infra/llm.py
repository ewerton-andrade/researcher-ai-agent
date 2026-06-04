"""Gemini chat client with native function-calling tool loop.

Implements the protocol used by both leaf agents (`RAGAgent`, `AnalystAgent`)
and the orchestrator: send a turn, if the model emits one or more
`function_call`s, execute the requested tools, append `function_response`
parts, and loop until the model emits text only (or until `max_steps`
is reached).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

from researcher.core.logging import get_logger
from researcher.core.models import ToolCallTrace
from researcher.core.settings import Settings

logger = get_logger(__name__)


ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[Any]]


@dataclass(slots=True)
class ToolBinding:
    """Pairs a Gemini `FunctionDeclaration` with the coroutine that runs it."""

    declaration: types.FunctionDeclaration
    executor: ToolExecutor


@dataclass(slots=True)
class LoopOutcome:
    text: str
    trace: list[ToolCallTrace]


class GeminiChat:
    """Thin async wrapper around a Gemini generative model + function-calling loop."""

    def __init__(
        self,
        model_name: str,
        system_instruction: str | None = None,
        bindings: list[ToolBinding] | None = None,
        max_steps: int = 8,
    ) -> None:
        self._model_name = model_name
        self._system_instruction = system_instruction
        self._bindings = {b.declaration.name: b for b in (bindings or [])}
        self._max_steps = max_steps

        self._tools: list[types.Tool] | None = None
        if bindings:
            self._tools = [types.Tool(function_declarations=[b.declaration for b in bindings])]

    async def generate_text(
        self,
        prompt: str,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        """One-shot text generation, no tool loop."""

        contents: list[dict[str, Any]] = list(history or [])
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        response = await self._generate(contents)
        return _extract_text(response)

    async def run_loop(
        self,
        user_message: str,
        history: list[dict[str, Any]] | None = None,
    ) -> LoopOutcome:
        """Drive the tool-calling loop until the model returns text only."""

        contents: list[dict[str, Any]] = list(history or [])
        contents.append({"role": "user", "parts": [{"text": user_message}]})
        trace: list[ToolCallTrace] = []

        for step in range(self._max_steps):
            response = await self._generate(contents)
            calls = _extract_function_calls(response)
            if not calls:
                return LoopOutcome(text=_extract_text(response), trace=trace)

            # Persist the assistant turn containing the function_call parts.
            contents.append(
                {
                    "role": "model",
                    "parts": [{"function_call": {"name": c[0], "args": c[1]}} for c in calls],
                }
            )

            # Execute all requested tool calls in parallel.
            tasks = [self._invoke_tool(name, args, trace) for name, args in calls]
            results = await asyncio.gather(*tasks)
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": name,
                                "response": {"content": _json_safe(result)},
                            }
                        }
                        for (name, _), result in zip(calls, results, strict=True)
                    ],
                }
            )
            logger.debug("tool_loop_step", step=step, calls=[c[0] for c in calls])

        logger.warning("tool_loop_max_steps_reached", steps=self._max_steps)
        return LoopOutcome(
            text="(stopped: maximum tool-calling steps reached)",
            trace=trace,
        )

    async def _invoke_tool(
        self, name: str, args: dict[str, Any], trace: list[ToolCallTrace]
    ) -> Any:
        binding = self._bindings.get(name)
        if binding is None:
            err = f"Unknown tool {name!r}"
            trace.append(ToolCallTrace(tool=name, arguments=args, ok=False, error=err))
            return {"error": err}
        try:
            result = await binding.executor(name, args)
            trace.append(ToolCallTrace(tool=name, arguments=args, ok=True))
            return result
        except Exception as exc:  # noqa: BLE001 - surface to the model
            logger.exception("tool_execution_failed", tool=name)
            trace.append(ToolCallTrace(tool=name, arguments=args, ok=False, error=str(exc)))
            return {"error": str(exc)}

    async def _generate(self, contents: list[dict[str, Any]]) -> types.GenerateContentResponse:
        config = types.GenerateContentConfig(
            system_instruction=self._system_instruction,
            tools=self._tools,
        )

        def _call() -> types.GenerateContentResponse:
            return get_genai_client().models.generate_content(
                model=self._model_name,
                contents=contents,
                config=config,
            )

        return await asyncio.to_thread(_call)


def _extract_text(response: types.GenerateContentResponse) -> str:
    parts: list[str] = []
    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _extract_function_calls(response: types.GenerateContentResponse) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            fc = getattr(part, "function_call", None)
            if fc and fc.name:
                args = dict(fc.args) if fc.args is not None else {}
                calls.append((fc.name, args))
    return calls


def _json_safe(value: Any) -> Any:
    """Ensure tool outputs are JSON-serializable for the function_response part."""

    try:
        json.dumps(value)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str))


_client: genai.Client | None = None


def get_genai_client() -> genai.Client:
    if _client is None:
        raise RuntimeError("configure_genai() must be called before using Gemini")
    return _client


def configure_genai(settings: Settings) -> None:
    global _client
    _client = genai.Client(api_key=settings.google_api_key)
