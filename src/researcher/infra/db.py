"""Async SQLAlchemy engine, session factory and ORM models."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from researcher.core.settings import Settings


class Base(DeclarativeBase):
    pass


class ThreadORM(Base):
    __tablename__ = "threads"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    messages: Mapped[list[MessageORM]] = relationship(
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="MessageORM.id",
    )


class MessageORM(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("threads.id", ondelete="CASCADE"), index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    thread: Mapped[ThreadORM] = relationship(back_populates="messages")


class Database:
    """Wraps an async engine + session factory."""

    def __init__(self, url: str) -> None:
        self._engine = create_async_engine(url, echo=False, future=True)
        self._sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def create_all(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self._engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._sessionmaker() as session:
            yield session


def build_database(settings: Settings) -> Database:
    """Create the Database, ensuring the parent dir of the SQLite file exists."""

    sqlite_path = settings.sqlite_path
    parent = os.path.dirname(sqlite_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return Database(settings.sqlalchemy_url)
