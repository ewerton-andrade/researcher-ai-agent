"""Unit test for SummarizeTool."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from researcher.core.models import RetrievedChunk
from researcher.tools.summarize import SummarizeTool


@pytest.mark.asyncio
async def test_summarize_returns_bullets_from_json() -> None:
    embeddings = AsyncMock()
    embeddings.embed_query.return_value = [0.0]
    store = AsyncMock()
    store.query.return_value = [
        RetrievedChunk(
            paper_id="attention_is_all_you_need",
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            section="abstract",
            chunk_index=0,
            text="The Transformer relies entirely on self-attention.",
        )
    ]
    chat = AsyncMock()
    chat.generate_text.return_value = json.dumps(
        ["Self-attention replaces recurrence.", "Encoder-decoder stacks.", "SOTA on WMT'14."]
    )

    tool = SummarizeTool(embeddings=embeddings, store=store, chat=chat)
    result = await tool.run({"paper_id": "attention_is_all_you_need", "max_bullets": 3})

    assert result.ok is True
    assert result.data is not None
    assert len(result.data.bullets) == 3
    assert result.data.bullets[0].startswith("Self-attention")


@pytest.mark.asyncio
async def test_summarize_fallback_to_lines() -> None:
    embeddings = AsyncMock()
    embeddings.embed_query.return_value = [0.0]
    store = AsyncMock()
    store.query.return_value = []
    chat = AsyncMock()
    chat.generate_text.return_value = "- one\n- two\n- three"
    tool = SummarizeTool(embeddings=embeddings, store=store, chat=chat)
    result = await tool.run({"paper_id": "bert", "max_bullets": 2})
    assert result.ok is True
    assert result.data is not None
    assert result.data.bullets == ["one", "two"]
