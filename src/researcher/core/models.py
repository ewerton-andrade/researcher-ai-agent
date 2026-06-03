"""Shared domain models used across tools, agents and the API."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ToolResult(BaseModel, Generic[T]):
    """Standard envelope returned by every tool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ok: bool = True
    data: T | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    paper_id: str
    arxiv_id: str
    title: str
    section: str
    chunk_index: int
    text: str
    score: float | None = None


class ToolCallTrace(BaseModel):
    """Lightweight record of a tool invocation for observability."""

    tool: str
    arguments: dict[str, Any]
    ok: bool
    error: str | None = None


class AgentResponse(BaseModel):
    """What an agent returns to its caller (orchestrator or API)."""

    text: str
    trace: list[ToolCallTrace] = Field(default_factory=list)


class StoredMessage(BaseModel):
    id: int
    thread_id: str
    role: Role
    content: str
    created_at: datetime
