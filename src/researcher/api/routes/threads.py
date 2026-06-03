"""Thread endpoints: create, list, post message, list messages."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
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


@router.post("", response_model=CreateThreadResponse, status_code=status.HTTP_201_CREATED)
async def create_thread(session: AsyncSession = Depends(get_session)) -> CreateThreadResponse:
    repo = ThreadRepository(session)
    thread_id = await repo.create()
    return CreateThreadResponse(thread_id=thread_id)


@router.get("", response_model=ListThreadsResponse)
async def list_threads(session: AsyncSession = Depends(get_session)) -> ListThreadsResponse:
    repo = ThreadRepository(session)
    ids = await repo.list_ids()
    return ListThreadsResponse(threads=[ThreadSummary(thread_id=tid) for tid in ids])


@router.get("/{thread_id}/messages", response_model=ListMessagesResponse)
async def list_messages(
    thread_id: str,
    session: AsyncSession = Depends(get_session),
) -> ListMessagesResponse:
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


@router.post("/{thread_id}/messages", response_model=PostMessageResponse)
async def post_message(
    thread_id: str,
    body: PostMessageRequest,
    session: AsyncSession = Depends(get_session),
    orchestrator: OrchestratorAgent = Depends(get_orchestrator),
) -> PostMessageResponse:
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
