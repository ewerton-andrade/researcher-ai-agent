"""Shared helpers used by analyst-style tools (retrieve + summarize via LLM)."""

from __future__ import annotations

from researcher.core.models import RetrievedChunk
from researcher.infra.embeddings import EmbeddingClient
from researcher.infra.vector_store import VectorStore


async def retrieve_for(
    query: str,
    *,
    embeddings: EmbeddingClient,
    store: VectorStore,
    paper_id: str | None = None,
    k: int = 8,
) -> list[RetrievedChunk]:
    """Retrieve top-k chunks for a query, optionally restricted to one paper."""

    embedding = await embeddings.embed_query(query)
    where = {"paper_id": paper_id} if paper_id else None
    return await store.query(embedding, k=k, where=where)


def format_chunks(chunks: list[RetrievedChunk]) -> str:
    """Render chunks as a numbered, citation-friendly context block."""

    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] paper={c.paper_id} section={c.section} (chunk {c.chunk_index})\n{c.text}"
        )
    return "\n\n".join(lines)
