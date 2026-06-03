"""Repository layer abstracting persistence of threads and messages."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from researcher.core.models import Role, StoredMessage
from researcher.infra.db import MessageORM, ThreadORM


class ThreadRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self) -> str:
        thread_id = str(uuid.uuid4())
        self._session.add(ThreadORM(id=thread_id))
        await self._session.commit()
        return thread_id

    async def exists(self, thread_id: str) -> bool:
        result = await self._session.execute(
            select(ThreadORM.id).where(ThreadORM.id == thread_id)
        )
        return result.scalar_one_or_none() is not None

    async def list_ids(self) -> list[str]:
        result = await self._session.execute(
            select(ThreadORM.id).order_by(ThreadORM.created_at.desc())
        )
        return [row[0] for row in result.all()]


class MessageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, thread_id: str, role: Role, content: str) -> StoredMessage:
        row = MessageORM(thread_id=thread_id, role=role.value, content=content)
        self._session.add(row)
        await self._session.commit()
        await self._session.refresh(row)
        return StoredMessage(
            id=row.id,
            thread_id=row.thread_id,
            role=Role(row.role),
            content=row.content,
            created_at=row.created_at,
        )

    async def list_for_thread(self, thread_id: str) -> list[StoredMessage]:
        result = await self._session.execute(
            select(MessageORM)
            .where(MessageORM.thread_id == thread_id)
            .order_by(MessageORM.id)
        )
        rows = result.scalars().all()
        return [
            StoredMessage(
                id=r.id,
                thread_id=r.thread_id,
                role=Role(r.role),
                content=r.content,
                created_at=r.created_at,
            )
            for r in rows
        ]
