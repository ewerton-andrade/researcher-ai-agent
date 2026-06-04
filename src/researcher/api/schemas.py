"""Request/response schemas for the REST API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from researcher.core.models import Role, ToolCallTrace


class CreateThreadResponse(BaseModel):
    thread_id: str = Field(
        ...,
        description="Unique thread identifier used for all subsequent message calls.",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )


class ThreadSummary(BaseModel):
    thread_id: str = Field(
        ...,
        description="Unique thread identifier.",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )


class ListThreadsResponse(BaseModel):
    threads: list[ThreadSummary] = Field(
        default_factory=list,
        description="All known conversation threads.",
    )


class PostMessageRequest(BaseModel):
    content: str = Field(
        ...,
        min_length=1,
        description="User message content to send to the orchestrator.",
        examples=["What is the main contribution of Attention Is All You Need?"],
    )


class PostMessageResponse(BaseModel):
    thread_id: str = Field(
        ...,
        description="Thread identifier that produced this answer.",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    response: str = Field(
        ...,
        description="Assistant answer generated for the current turn.",
        examples=["The paper introduces the Transformer architecture based on self-attention."],
    )
    trace: list[ToolCallTrace] = Field(
        default_factory=list,
        description="Tool invocation trace captured during orchestration.",
    )


class HealthResponse(BaseModel):
    status: str = Field(
        ...,
        description="Service health status.",
        examples=["ok"],
    )


class MessageOut(BaseModel):
    id: int = Field(..., description="Monotonic message identifier.", examples=[1])
    role: Role = Field(..., description="Message author role.", examples=["assistant"])
    content: str = Field(
        ...,
        description="Message body text.",
        examples=["ReAct combines reasoning and acting through tool calls."],
    )
    created_at: datetime = Field(
        ...,
        description="UTC timestamp when the message was stored.",
        examples=["2026-06-03T10:30:05Z"],
    )


class ListMessagesResponse(BaseModel):
    thread_id: str = Field(
        ...,
        description="Thread identifier for the returned messages.",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    messages: list[MessageOut] = Field(
        default_factory=list,
        description="Chronological message history for the thread.",
    )
