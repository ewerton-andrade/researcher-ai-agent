"""Unit test for ExtractSectionTool."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from researcher.core.models import RetrievedChunk
from researcher.tools.extract_section import ExtractSectionTool


def _chunk(i: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        paper_id="bert",
        arxiv_id="1810.04805",
        title="BERT",
        section="abstract",
        chunk_index=i,
        text=text,
    )


@pytest.mark.asyncio
async def test_extract_section_returns_ordered_text() -> None:
    store = AsyncMock()
    # Intentionally returned out-of-order to validate the sort step.
    store.get.return_value = [_chunk(1, "second"), _chunk(0, "first")]
    tool = ExtractSectionTool(store=store)

    result = await tool.run({"paper_id": "bert", "section": "Abstract"})

    assert result.ok is True
    assert result.data is not None
    assert result.data.text.startswith("first")
    assert "second" in result.data.text
    assert result.data.truncated is False


@pytest.mark.asyncio
async def test_extract_section_truncates() -> None:
    store = AsyncMock()
    store.get.return_value = [_chunk(0, "x" * 500)]
    tool = ExtractSectionTool(store=store)
    result = await tool.run({"paper_id": "bert", "section": "abstract", "max_chars": 200})
    assert result.ok is True
    assert result.data is not None
    assert result.data.truncated is True
    assert len(result.data.text) == 200


@pytest.mark.asyncio
async def test_extract_section_unknown_paper() -> None:
    tool = ExtractSectionTool(store=AsyncMock())
    result = await tool.run({"paper_id": "unknown", "section": "abstract"})
    assert result.ok is False
    assert "Unknown paper_id" in (result.error or "")
