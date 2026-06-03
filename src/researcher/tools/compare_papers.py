"""Compare a set of papers on a given aspect, with grounded per-paper bullets."""

from __future__ import annotations

import asyncio
import json

from pydantic import BaseModel, Field

from researcher.core.papers import PAPERS_BY_ID, get_paper
from researcher.infra.embeddings import EmbeddingClient
from researcher.infra.llm import GeminiChat
from researcher.infra.vector_store import VectorStore
from researcher.tools._context import format_chunks, retrieve_for
from researcher.tools.base import Tool


class ComparePapersInput(BaseModel):
    paper_ids: list[str] = Field(
        ...,
        min_length=2,
        description=f"List of papers to compare. Valid ids: {', '.join(PAPERS_BY_ID)}.",
    )
    aspect: str = Field(
        ..., description="What to compare on (e.g. 'tool-use mechanism', 'training data')."
    )


class PaperPoint(BaseModel):
    paper_id: str
    title: str
    bullets: list[str]


class ComparePapersOutput(BaseModel):
    aspect: str
    per_paper: list[PaperPoint]
    synthesis: str


_PROMPT = """You compare ML papers grounded strictly in the supplied excerpts.

Aspect to compare: {aspect}

Excerpts:
{context}

Return ONLY a JSON object with this exact schema (no markdown fences, no prose):
{{
  "per_paper": [
    {{"paper_id": "<id>", "bullets": ["...", "..."]}}
  ],
  "synthesis": "2-4 sentence overall comparison."
}}
Use 3-5 bullets per paper. Only cite content present in the excerpts. The paper_id values must come from this set: {ids}.
"""


class ComparePapersTool(Tool[ComparePapersInput, ComparePapersOutput]):
    name = "compare_papers"
    description = (
        "Compare two or more papers on a specific aspect. Returns per-paper bullet points "
        "plus a short synthesis, grounded in retrieved excerpts."
    )
    InputModel = ComparePapersInput
    OutputModel = ComparePapersOutput

    def __init__(
        self,
        embeddings: EmbeddingClient,
        store: VectorStore,
        chat: GeminiChat,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._chat = chat

    async def _execute(self, payload: ComparePapersInput) -> ComparePapersOutput:
        # Validate all paper ids up front.
        papers = [get_paper(pid) for pid in payload.paper_ids]

        # Retrieve aspect-focused chunks per paper in parallel.
        chunk_lists = await asyncio.gather(
            *[
                retrieve_for(
                    query=f"{payload.aspect}",
                    embeddings=self._embeddings,
                    store=self._store,
                    paper_id=p.paper_id,
                    k=6,
                )
                for p in papers
            ]
        )
        all_chunks = [c for sub in chunk_lists for c in sub]

        prompt = _PROMPT.format(
            aspect=payload.aspect,
            context=format_chunks(all_chunks),
            ids=", ".join(p.paper_id for p in papers),
        )
        raw = await self._chat.generate_text(prompt)
        parsed = _parse_json(raw)

        per_paper: list[PaperPoint] = []
        seen: set[str] = set()
        for entry in parsed.get("per_paper", []):
            pid = entry.get("paper_id")
            if pid in PAPERS_BY_ID and pid not in seen:
                seen.add(pid)
                per_paper.append(
                    PaperPoint(
                        paper_id=pid,
                        title=PAPERS_BY_ID[pid].title,
                        bullets=[str(b) for b in entry.get("bullets", [])],
                    )
                )

        return ComparePapersOutput(
            aspect=payload.aspect,
            per_paper=per_paper,
            synthesis=str(parsed.get("synthesis", "")).strip(),
        )


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        # Strip markdown fences if the model added them despite the instruction.
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to locate the first JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise
