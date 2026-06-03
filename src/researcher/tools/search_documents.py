"""Semantic search over the ChromaDB vector store."""

from __future__ import annotations

from pydantic import BaseModel, Field

from researcher.core.models import RetrievedChunk
from researcher.core.papers import PAPERS_BY_ID
from researcher.infra.embeddings import EmbeddingClient
from researcher.infra.vector_store import VectorStore
from researcher.tools.base import Tool


class SearchDocumentsInput(BaseModel):
    query: str = Field(..., description="Natural-language search query.")
    k: int = Field(default=5, ge=1, le=20, description="Number of chunks to return.")
    paper_ids: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of paper_id values to restrict the search to. "
            f"Valid ids: {', '.join(PAPERS_BY_ID)}."
        ),
    )


class SearchDocumentsOutput(BaseModel):
    chunks: list[RetrievedChunk]


class SearchDocumentsTool(Tool[SearchDocumentsInput, SearchDocumentsOutput]):
    name = "search_documents"
    description = (
        "Semantic search over the indexed ML papers. Returns the top-k most relevant "
        "text chunks for a given natural-language query, optionally filtered by paper."
    )
    InputModel = SearchDocumentsInput
    OutputModel = SearchDocumentsOutput

    def __init__(self, embeddings: EmbeddingClient, store: VectorStore) -> None:
        self._embeddings = embeddings
        self._store = store

    async def _execute(self, payload: SearchDocumentsInput) -> SearchDocumentsOutput:
        embedding = await self._embeddings.embed_query(payload.query)
        where = None
        if payload.paper_ids:
            ids = [pid for pid in payload.paper_ids if pid in PAPERS_BY_ID]
            if ids:
                where = {"paper_id": {"$in": ids}} if len(ids) > 1 else {"paper_id": ids[0]}
        chunks = await self._store.query(embedding, k=payload.k, where=where)
        return SearchDocumentsOutput(chunks=chunks)
