"""Unit test for ComparePapersTool."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from researcher.core.models import RetrievedChunk
from researcher.tools.compare_papers import ComparePapersTool


def _chunk(paper_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        paper_id=paper_id,
        arxiv_id="x",
        title=paper_id.upper(),
        section="method",
        chunk_index=0,
        text=text,
    )


@pytest.mark.asyncio
async def test_compare_papers_parses_model_json() -> None:
    embeddings = AsyncMock()
    embeddings.embed_query.return_value = [0.0]
    store = AsyncMock()
    store.query.side_effect = [
        [_chunk("react", "react interleaves reasoning and acting")],
        [_chunk("toolformer", "toolformer self-trains api calls")],
    ]
    chat = AsyncMock()
    chat.generate_text.return_value = json.dumps(
        {
            "per_paper": [
                {"paper_id": "react", "bullets": ["interleaves reasoning + acting"]},
                {"paper_id": "toolformer", "bullets": ["self-supervised tool use"]},
            ],
            "synthesis": "Both use external tools but differ in supervision.",
        }
    )

    tool = ComparePapersTool(embeddings=embeddings, store=store, chat=chat)
    result = await tool.run(
        {"paper_ids": ["react", "toolformer"], "aspect": "tool use strategy"}
    )

    assert result.ok is True
    assert result.data is not None
    assert [p.paper_id for p in result.data.per_paper] == ["react", "toolformer"]
    assert "supervision" in result.data.synthesis


@pytest.mark.asyncio
async def test_compare_papers_rejects_too_few_papers() -> None:
    tool = ComparePapersTool(embeddings=AsyncMock(), store=AsyncMock(), chat=AsyncMock())
    result = await tool.run({"paper_ids": ["react"], "aspect": "x"})
    assert result.ok is False
    assert "Invalid input" in (result.error or "")
