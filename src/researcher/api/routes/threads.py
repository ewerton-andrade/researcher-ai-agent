"""Thread endpoints: create, list, post message, list messages."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy.ext.asyncio import AsyncSession

from researcher.agents.orchestrator import OrchestratorAgent
from researcher.api.deps import get_orchestrator, get_session
from researcher.api.schemas import (
    CreateThreadResponse,
    ListMessagesResponse,
    ListThreadsResponse,
    MessageOut,
    PostMessageRequest,
    PostMessageResponse,
    ThreadSummary,
)
from researcher.core.models import Role
from researcher.infra.repositories import MessageRepository, ThreadRepository

router = APIRouter(prefix="/threads", tags=["threads"])


@router.post(
    "",
    response_model=CreateThreadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a thread",
    description=(
        "Create a new empty conversation thread and return its identifier. "
        "Use the returned thread_id to post and retrieve messages."
    ),
    responses={
        status.HTTP_201_CREATED: {
            "description": "Thread created",
            "content": {
                "application/json": {
                    "example": {"thread_id": "550e8400-e29b-41d4-a716-446655440000"}
                }
            },
        }
    },
)
async def create_thread(session: AsyncSession = Depends(get_session)) -> CreateThreadResponse:
    """Create a thread container for a multi-turn conversation."""
    repo = ThreadRepository(session)
    thread_id = await repo.create()
    return CreateThreadResponse(thread_id=thread_id)


@router.get(
    "",
    response_model=ListThreadsResponse,
    summary="List threads",
    description="Return all thread identifiers currently stored.",
    responses={
        status.HTTP_200_OK: {
            "description": "Thread list",
            "content": {
                "application/json": {
                    "example": {
                        "threads": [
                            {"thread_id": "550e8400-e29b-41d4-a716-446655440000"},
                            {"thread_id": "f2a89f6b-0b2b-4d85-8d91-e5dd4e0f9d0a"},
                        ]
                    }
                }
            },
        }
    },
)
async def list_threads(session: AsyncSession = Depends(get_session)) -> ListThreadsResponse:
    """List all known conversation threads."""
    repo = ThreadRepository(session)
    ids = await repo.list_ids()
    return ListThreadsResponse(threads=[ThreadSummary(thread_id=tid) for tid in ids])


@router.get(
    "/{thread_id}/messages",
    response_model=ListMessagesResponse,
    summary="List thread messages",
    description=(
        "Return the full persisted message history for a thread in chronological order."
    ),
    responses={
        status.HTTP_200_OK: {
            "description": "Messages for the thread",
            "content": {
                "application/json": {
                    "example": {
                        "thread_id": "550e8400-e29b-41d4-a716-446655440000",
                        "messages": [
                            {
                                "id": 1,
                                "role": "user",
                                "content": "Compare ReAct and Toolformer.",
                                "created_at": "2026-06-03T10:30:00Z",
                            },
                            {
                                "id": 2,
                                "role": "assistant",
                                "content": "Both approaches teach models to use tools, but differ in supervision.",
                                "created_at": "2026-06-03T10:30:05Z",
                            },
                        ],
                    }
                }
            },
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Thread not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Thread not found"}
                }
            },
        },
    },
)
async def list_messages(
    thread_id: str = Path(..., description="Unique thread identifier"),
    session: AsyncSession = Depends(get_session),
) -> ListMessagesResponse:
    """Get all messages that belong to a single thread."""
    thread_repo = ThreadRepository(session)
    if not await thread_repo.exists(thread_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
    msg_repo = MessageRepository(session)
    messages = await msg_repo.list_for_thread(thread_id)
    return ListMessagesResponse(
        thread_id=thread_id,
        messages=[
            MessageOut(id=m.id, role=m.role, content=m.content, created_at=m.created_at)
            for m in messages
        ],
    )


@router.post(
    "/{thread_id}/messages",
    response_model=PostMessageResponse,
    summary="Post a user message",
    description=(
        "Append a user message to the target thread, run multi-agent orchestration, "
        "persist the assistant response, and return the answer with tool trace metadata."
    ),
    responses={
        status.HTTP_200_OK: {
            "description": "Assistant response for the new turn",
            "content": {
                "application/json": {
                    "example": {
                        "thread_id": "550e8400-e29b-41d4-a716-446655440000",
                        "response": "The Transformer introduced self-attention as the core sequence modeling mechanism.",
                        "trace": [
                            {
                                "tool": "ask_rag_agent",
                                "arguments": {
                                    "question": "What is the main contribution of Attention Is All You Need?"
                                },
                                "ok": True,
                                "error": None,
                            }
                        ],
                    }
                }
            },
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Thread not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Thread not found"}
                }
            },
        },
    },
)
async def post_message(
    body: PostMessageRequest,
    thread_id: str = Path(..., description="Unique thread identifier"),
    session: AsyncSession = Depends(get_session),
    orchestrator: OrchestratorAgent = Depends(get_orchestrator),
) -> PostMessageResponse:
    """Run one conversation turn and return the assistant answer."""
    thread_repo = ThreadRepository(session)
    if not await thread_repo.exists(thread_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")

    msg_repo = MessageRepository(session)
    history = await msg_repo.list_for_thread(thread_id)

    # Persist the user message first so it is visible if anything fails downstream.
    await msg_repo.add(thread_id, Role.USER, body.content)

    agent_response = await orchestrator.respond(user_message=body.content, history=history)

    await msg_repo.add(thread_id, Role.ASSISTANT, agent_response.text)
    return PostMessageResponse(
        thread_id=thread_id,
        response=agent_response.text,
        trace=agent_response.trace,
    )
