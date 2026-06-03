"""Unit test for SearchDocumentsTool."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from researcher.core.models import RetrievedChunk
from researcher.tools.search_documents import SearchDocumentsTool


@pytest.mark.asyncio
async def test_search_documents_happy_path() -> None:
    embeddings = AsyncMock()
    embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    store = AsyncMock()
    store.query.return_value = [
        RetrievedChunk(
            paper_id="rag",
            arxiv_id="2005.11401",
            title="RAG",
            section="abstract",
            chunk_index=0,
            text="Retrieval-Augmented Generation combines a retriever with a generator.",
            score=0.91,
        )
    ]

    tool = SearchDocumentsTool(embeddings=embeddings, store=store)
    result = await tool.run({"query": "what is RAG", "k": 3, "paper_ids": ["rag"]})

    assert result.ok is True
    assert result.data is not None
    assert len(result.data.chunks) == 1
    assert result.data.chunks[0].paper_id == "rag"
    embeddings.embed_query.assert_awaited_once_with("what is RAG")
    store.query.assert_awaited_once()
    # paper_ids of length 1 should produce equality filter, not $in
    _, kwargs = store.query.call_args
    assert kwargs["where"] == {"paper_id": "rag"}


@pytest.mark.asyncio
async def test_search_documents_invalid_input() -> None:
    tool = SearchDocumentsTool(embeddings=AsyncMock(), store=AsyncMock())
    result = await tool.run({"k": 3})  # missing required `query`
    assert result.ok is False
    assert "Invalid input" in (result.error or "")


@pytest.mark.asyncio
async def test_search_documents_ignores_unknown_paper_ids() -> None:
    embeddings = AsyncMock()
    embeddings.embed_query.return_value = [0.0]
    store = AsyncMock()
    store.query.return_value = []
    tool = SearchDocumentsTool(embeddings=embeddings, store=store)

    await tool.run({"query": "x", "paper_ids": ["does_not_exist"]})
    _, kwargs = store.query.call_args
    assert kwargs["where"] is None
