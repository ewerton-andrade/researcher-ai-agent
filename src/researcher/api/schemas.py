"""Request/response schemas for the REST API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from researcher.core.models import Role, ToolCallTrace


class CreateThreadResponse(BaseModel):
    thread_id: str


class ThreadSummary(BaseModel):
    thread_id: str


class ListThreadsResponse(BaseModel):
    threads: list[ThreadSummary]


class PostMessageRequest(BaseModel):
    content: str = Field(..., min_length=1)


class PostMessageResponse(BaseModel):
    thread_id: str
    response: str
    trace: list[ToolCallTrace] = Field(default_factory=list)


class MessageOut(BaseModel):
    id: int
    role: Role
    content: str
    created_at: datetime


class ListMessagesResponse(BaseModel):
    thread_id: str
    messages: list[MessageOut]
