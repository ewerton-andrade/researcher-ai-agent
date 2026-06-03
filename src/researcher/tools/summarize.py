"""Bullet-point summarization of a single paper."""

from __future__ import annotations

from pydantic import BaseModel, Field

from researcher.core.papers import PAPERS_BY_ID, get_paper
from researcher.infra.embeddings import EmbeddingClient
from researcher.infra.llm import GeminiChat
from researcher.infra.vector_store import VectorStore
from researcher.tools._context import format_chunks, retrieve_for
from researcher.tools.base import Tool


class SummarizeInput(BaseModel):
    paper_id: str = Field(
        ..., description=f"Paper to summarize. Valid ids: {', '.join(PAPERS_BY_ID)}."
    )
    max_bullets: int = Field(default=5, ge=2, le=10)


class SummarizeOutput(BaseModel):
    paper_id: str
    title: str
    bullets: list[str]


_PROMPT = """Summarize the paper titled "{title}" using the excerpts below.

Excerpts:
{context}

Return ONLY a JSON array of {n} concise bullet points (strings). No prose, no markdown.
Cover: core contribution, method, key results, limitations, and why it matters."""


class SummarizeTool(Tool[SummarizeInput, SummarizeOutput]):
    name = "summarize"
    description = (
        "Produce an executive-style bullet-point summary of a single paper "
        "(default: 5 bullets), grounded in retrieved excerpts."
    )
    InputModel = SummarizeInput
    OutputModel = SummarizeOutput

    def __init__(
        self,
        embeddings: EmbeddingClient,
        store: VectorStore,
        chat: GeminiChat,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._chat = chat

    async def _execute(self, payload: SummarizeInput) -> SummarizeOutput:
        paper = get_paper(payload.paper_id)
        chunks = await retrieve_for(
            query=f"{paper.title} contribution method results limitations",
            embeddings=self._embeddings,
            store=self._store,
            paper_id=paper.paper_id,
            k=10,
        )
        prompt = _PROMPT.format(
            title=paper.title,
            context=format_chunks(chunks),
            n=payload.max_bullets,
        )
        raw = await self._chat.generate_text(prompt)
        bullets = _parse_bullets(raw, payload.max_bullets)
        return SummarizeOutput(paper_id=paper.paper_id, title=paper.title, bullets=bullets)


def _parse_bullets(text: str, n: int) -> list[str]:
    import json

    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(item).strip() for item in data][:n]
    except json.JSONDecodeError:
        pass
    # Fallback: split on lines and strip common bullet prefixes one char at a time.
    bullet_chars = set("-•*  \t")
    lines = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        while stripped and stripped[0] in bullet_chars:
            stripped = stripped[1:]
        stripped = stripped.strip()
        if stripped:
            lines.append(stripped)
    return lines[:n]
