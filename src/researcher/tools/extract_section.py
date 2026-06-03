"""Extract a named section of a specific paper using metadata filtering."""

from __future__ import annotations

from pydantic import BaseModel, Field

from researcher.core.papers import PAPERS_BY_ID, get_paper
from researcher.infra.vector_store import VectorStore
from researcher.tools.base import Tool

# Allowed section names (mirrors what the ingest pipeline normalizes to).
_KNOWN_SECTIONS = (
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "approach",
    "model",
    "architecture",
    "experiments",
    "experimental setup",
    "results",
    "evaluation",
    "discussion",
    "analysis",
    "conclusion",
    "limitations",
    "appendix",
    "frontmatter",
    "body",
)


class ExtractSectionInput(BaseModel):
    paper_id: str = Field(
        ..., description=f"Target paper. Valid ids: {', '.join(PAPERS_BY_ID)}."
    )
    section: str = Field(
        ...,
        description=(
            "Section name to extract. Common values: "
            + ", ".join(_KNOWN_SECTIONS)
            + ". Matched case-insensitively."
        ),
    )
    max_chars: int = Field(
        default=6000, ge=200, le=20000, description="Maximum number of characters to return."
    )


class ExtractSectionOutput(BaseModel):
    paper_id: str
    title: str
    section: str
    text: str
    truncated: bool


class ExtractSectionTool(Tool[ExtractSectionInput, ExtractSectionOutput]):
    name = "extract_section"
    description = (
        "Return the concatenated text of a specific section (e.g. abstract, method, "
        "conclusion) from a specific paper."
    )
    InputModel = ExtractSectionInput
    OutputModel = ExtractSectionOutput

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    async def _execute(self, payload: ExtractSectionInput) -> ExtractSectionOutput:
        paper = get_paper(payload.paper_id)
        section = payload.section.strip().lower()
        chunks = await self._store.get(
            where={
                "$and": [
                    {"paper_id": paper.paper_id},
                    {"section": section},
                ]
            }
        )
        chunks.sort(key=lambda c: c.chunk_index)
        joined = "\n\n".join(c.text for c in chunks)
        truncated = len(joined) > payload.max_chars
        if truncated:
            joined = joined[: payload.max_chars]
        return ExtractSectionOutput(
            paper_id=paper.paper_id,
            title=paper.title,
            section=section,
            text=joined,
            truncated=truncated,
        )
