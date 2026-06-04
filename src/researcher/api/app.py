"""FastAPI application factory + lifespan wiring."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from researcher.agents.analyst_agent import AnalystAgent
from researcher.agents.orchestrator import OrchestratorAgent
from researcher.agents.rag_agent import RAGAgent
from researcher.api.deps import AppState
from researcher.api.routes.threads import router as threads_router
from researcher.api.schemas import HealthResponse
from researcher.core.logging import configure_logging, get_logger
from researcher.core.settings import Settings, get_settings
from researcher.infra.db import build_database
from researcher.infra.embeddings import build_embedding_client
from researcher.infra.llm import GeminiChat, configure_genai
from researcher.infra.vector_store import VectorStore

logger = get_logger(__name__)


async def _build_state(settings: Settings) -> AppState:
    configure_genai(settings)
    db = build_database(settings)
    await db.create_all()
    store = await VectorStore.connect(settings)
    embeddings = build_embedding_client(settings)
    # A shared "helper" chat used by analyst tools for grounded synthesis.
    helper_chat = GeminiChat(model_name=settings.gemini_model)
    rag_agent = RAGAgent(
        model_name=settings.gemini_model,
        embeddings=embeddings,
        store=store,
        max_steps=settings.agent_max_steps,
    )
    analyst_agent = AnalystAgent(
        model_name=settings.gemini_model,
        embeddings=embeddings,
        store=store,
        helper_chat=helper_chat,
        max_steps=settings.agent_max_steps,
    )
    orchestrator = OrchestratorAgent(
        model_name=settings.gemini_model,
        rag_agent=rag_agent,
        analyst_agent=analyst_agent,
        max_steps=settings.agent_max_steps,
    )
    return AppState(db=db, orchestrator=orchestrator)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("api_startup", model=settings.gemini_model)
    state = await _build_state(settings)
    app.state.app_state = state
    try:
        yield
    finally:
        await state.db.dispose()
        logger.info("api_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Researcher AI Agent",
        description=(
            "Multi-agent Q&A system over a fixed set of ML papers. "
            "Use threads endpoints to create a conversation, send user messages, "
            "and inspect message history with orchestration traces."
        ),
        version="0.1.0",
        openapi_tags=[
            {
                "name": "threads",
                "description": "Create conversations and exchange messages with the multi-agent backend.",
            },
            {"name": "meta", "description": "Service metadata and operational endpoints."},
        ],
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(threads_router)

    @app.get(
        "/health",
        tags=["meta"],
        response_model=HealthResponse,
        summary="Health check",
        description="Lightweight liveness endpoint for service monitoring.",
        responses={
            200: {
                "description": "Service is healthy",
                "content": {"application/json": {"example": {"status": "ok"}}},
            }
        },
    )
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    return app


app = create_app()
