"""Async wrapper around the ChromaDB HTTP client."""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.api.async_api import AsyncClientAPI
from chromadb.config import Settings as ChromaSettings

from researcher.core.logging import get_logger
from researcher.core.models import RetrievedChunk
from researcher.core.settings import Settings

logger = get_logger(__name__)


class VectorStore:
    """Thin async facade over a single Chroma collection."""

    def __init__(self, client: AsyncClientAPI, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name

    @classmethod
    async def connect(cls, settings: Settings) -> VectorStore:
        client = await chromadb.AsyncHttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        await client.get_or_create_collection(name=settings.chroma_collection)
        return cls(client, settings.chroma_collection)

    async def _collection(self):
        return await self._client.get_or_create_collection(name=self._collection_name)

    async def reset_collection(self) -> None:
        try:
            await self._client.delete_collection(name=self._collection_name)
        except Exception as exc:  # pragma: no cover - chromadb raises generic ValueError
            logger.info("collection_delete_skipped", error=str(exc))
        await self._client.get_or_create_collection(name=self._collection_name)

    async def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        collection = await self._collection()
        await collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    async def count(self) -> int:
        collection = await self._collection()
        return await collection.count()

    async def query(
        self,
        query_embedding: list[float],
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        collection = await self._collection()
        result = await collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
        )
        return _to_chunks(result)

    async def get(
        self,
        where: dict[str, Any],
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        collection = await self._collection()
        result = await collection.get(where=where, limit=limit)
        return _to_chunks_from_get(result)


def _to_chunks(result: dict[str, Any]) -> list[RetrievedChunk]:
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[None] * len(docs)])[0]
    chunks: list[RetrievedChunk] = []
    for doc, meta, dist in zip(docs, metas, dists, strict=False):
        chunks.append(_chunk_from(doc, meta, dist))
    return chunks


def _to_chunks_from_get(result: dict[str, Any]) -> list[RetrievedChunk]:
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    return [_chunk_from(doc, meta, None) for doc, meta in zip(docs, metas, strict=False)]


def _chunk_from(text: str, meta: dict[str, Any], distance: float | None) -> RetrievedChunk:
    return RetrievedChunk(
        paper_id=str(meta.get("paper_id", "")),
        arxiv_id=str(meta.get("arxiv_id", "")),
        title=str(meta.get("title", "")),
        section=str(meta.get("section", "")),
        chunk_index=int(meta.get("chunk_index", 0)),
        text=text,
        score=None if distance is None else 1.0 - float(distance),
    )
