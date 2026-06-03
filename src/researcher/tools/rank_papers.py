"""Rank papers by a given criterion, with justifications."""

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


class RankPapersInput(BaseModel):
    paper_ids: list[str] = Field(
        ...,
        min_length=2,
        description=f"Papers to rank. Valid ids: {', '.join(PAPERS_BY_ID)}.",
    )
    criterion: str = Field(
        ..., description="Ranking criterion (e.g. 'usefulness for building tool-using agents')."
    )


class RankedPaper(BaseModel):
    rank: int
    paper_id: str
    title: str
    justification: str


class RankPapersOutput(BaseModel):
    criterion: str
    ranking: list[RankedPaper]


_PROMPT = """Rank the following papers by this criterion: {criterion}

Use only the supplied excerpts as evidence. Higher rank (1 = best) = stronger fit to the criterion.

Excerpts:
{context}

Return ONLY a JSON array, each item:
{{"paper_id": "<id>", "justification": "1-2 sentences citing the excerpts"}}
Items in order from best to worst. Include every paper exactly once. Valid ids: {ids}."""


class RankPapersTool(Tool[RankPapersInput, RankPapersOutput]):
    name = "rank_papers"
    description = (
        "Rank a set of papers by a free-form criterion, returning an ordered list "
        "with a justification per position grounded in the texts."
    )
    InputModel = RankPapersInput
    OutputModel = RankPapersOutput

    def __init__(
        self,
        embeddings: EmbeddingClient,
        store: VectorStore,
        chat: GeminiChat,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._chat = chat

    async def _execute(self, payload: RankPapersInput) -> RankPapersOutput:
        papers = [get_paper(pid) for pid in payload.paper_ids]
        chunk_lists = await asyncio.gather(
            *[
                retrieve_for(
                    query=payload.criterion,
                    embeddings=self._embeddings,
                    store=self._store,
                    paper_id=p.paper_id,
                    k=5,
                )
                for p in papers
            ]
        )
        all_chunks = [c for sub in chunk_lists for c in sub]
        prompt = _PROMPT.format(
            criterion=payload.criterion,
            context=format_chunks(all_chunks),
            ids=", ".join(p.paper_id for p in papers),
        )
        raw = await self._chat.generate_text(prompt)
        items = _parse_array(raw)

        ranked: list[RankedPaper] = []
        position = 1
        used: set[str] = set()
        for item in items:
            pid = item.get("paper_id")
            if pid in PAPERS_BY_ID and pid not in used:
                used.add(pid)
                ranked.append(
                    RankedPaper(
                        rank=position,
                        paper_id=pid,
                        title=PAPERS_BY_ID[pid].title,
                        justification=str(item.get("justification", "")).strip(),
                    )
                )
                position += 1
        # Fill in any missing papers at the bottom to satisfy the contract.
        for p in papers:
            if p.paper_id not in used:
                ranked.append(
                    RankedPaper(
                        rank=position,
                        paper_id=p.paper_id,
                        title=p.title,
                        justification="(not ranked by the model; appended for completeness)",
                    )
                )
                position += 1
        return RankPapersOutput(criterion=payload.criterion, ranking=ranked)


def _parse_array(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    return []
