"""Tool ABC + helpers to expose tools as Gemini function declarations."""

from __future__ import annotations

import abc
from typing import Any, Generic, TypeVar

from google.generativeai.types import FunctionDeclaration
from pydantic import BaseModel, ValidationError

from researcher.core.models import ToolResult

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class Tool(abc.ABC, Generic[InputT, OutputT]):
    """Atomic, reusable capability with a typed input/output schema."""

    name: str
    description: str
    InputModel: type[InputT]
    OutputModel: type[OutputT]

    @abc.abstractmethod
    async def _execute(self, payload: InputT) -> OutputT:
        """Concrete tool logic. Should raise on unrecoverable errors."""

    async def run(self, raw_input: dict[str, Any]) -> ToolResult[OutputT]:
        """Validate input, execute, wrap the result in a `ToolResult`."""

        try:
            payload = self.InputModel.model_validate(raw_input)
        except ValidationError as exc:
            return ToolResult(ok=False, error=f"Invalid input: {exc}")
        try:
            output = await self._execute(payload)
        except Exception as exc:  # noqa: BLE001 - surface to the model
            return ToolResult(ok=False, error=str(exc))
        return ToolResult(ok=True, data=output)

    def function_declaration(self) -> FunctionDeclaration:
        """Render this tool as a Gemini `FunctionDeclaration`."""

        schema = _sanitize_schema(self.InputModel.model_json_schema())
        return FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=schema,
        )


_ALLOWED_KEYS = {"type", "properties", "required", "items", "description", "enum"}


def _sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a Pydantic JSON schema into the subset Gemini accepts.

    Inlines $ref/$defs and strips keys Gemini does not understand
    (e.g. `title`, `default`, `anyOf`, `additionalProperties`).
    """

    defs = schema.get("$defs") or schema.get("definitions") or {}

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"].split("/")[-1]
                return _walk(defs.get(ref, {}))
            # Collapse a single-element anyOf (typical of `X | None`).
            if "anyOf" in node and len(node["anyOf"]) > 0:
                non_null = [a for a in node["anyOf"] if a.get("type") != "null"]
                base = non_null[0] if non_null else node["anyOf"][0]
                merged = {**{k: v for k, v in node.items() if k != "anyOf"}, **base}
                return _walk(merged)
            cleaned: dict[str, Any] = {}
            for key, value in node.items():
                if key in _ALLOWED_KEYS:
                    cleaned[key] = _walk(value)
            return cleaned
        if isinstance(node, list):
            return [_walk(item) for item in node]
        return node

    return _walk(schema)
