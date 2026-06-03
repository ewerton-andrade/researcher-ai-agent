"""Unit test for RankPapersTool."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from researcher.core.models import RetrievedChunk
from researcher.tools.rank_papers import RankPapersTool


@pytest.mark.asyncio
async def test_rank_papers_orders_and_fills_missing() -> None:
    embeddings = AsyncMock()
    embeddings.embed_query.return_value = [0.0]
    store = AsyncMock()
    store.query.return_value = [
        RetrievedChunk(
            paper_id="x",
            arxiv_id="x",
            title="x",
            section="x",
            chunk_index=0,
            text="ctx",
        )
    ]
    chat = AsyncMock()
    chat.generate_text.return_value = json.dumps(
        [
            {"paper_id": "toolformer", "justification": "self-supervised tool calls"},
            # rag omitted intentionally — should be appended
        ]
    )

    tool = RankPapersTool(embeddings=embeddings, store=store, chat=chat)
    result = await tool.run(
        {"paper_ids": ["toolformer", "rag"], "criterion": "tool use"}
    )
    assert result.ok is True
    assert result.data is not None
    assert [r.paper_id for r in result.data.ranking] == ["toolformer", "rag"]
    assert result.data.ranking[0].rank == 1
    assert result.data.ranking[1].rank == 2
    assert "appended" in result.data.ranking[1].justification
