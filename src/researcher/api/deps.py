"""Dependency injection helpers for the API layer."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from researcher.agents.orchestrator import OrchestratorAgent
from researcher.infra.db import Database


@dataclass(slots=True)
class AppState:
    db: Database
    orchestrator: OrchestratorAgent


def get_state(request: Request) -> AppState:
    return request.app.state.app_state  # type: ignore[no-any-return]


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    state = get_state(request)
    async with state.db.session() as session:
        yield session


def get_orchestrator(request: Request) -> OrchestratorAgent:
    return get_state(request).orchestrator
