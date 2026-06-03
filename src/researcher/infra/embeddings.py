"""Async embedding client backed by Google's Gemini embeddings."""

from __future__ import annotations

import asyncio

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from researcher.core.logging import get_logger
from researcher.core.settings import Settings

logger = get_logger(__name__)


class EmbeddingClient:
    """Wraps `genai.embed_content` with batching and retry."""

    def __init__(self, model: str, batch_size: int = 100) -> None:
        self._model = model
        self._batch_size = batch_size

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20))
    async def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        # `embed_content` is sync in google-generativeai; run in a worker thread
        # to keep the surrounding code async.
        def _call() -> list[list[float]]:
            resp = genai.embed_content(
                model=f"models/{self._model}" if not self._model.startswith("models/") else self._model,
                content=texts,
                task_type=task_type,
            )
            embeddings = resp["embedding"]
            # When `content` is a list, the response is `{'embedding': [[...], [...], ...]}`.
            # When it is a single string, it is `{'embedding': [...]}`. Normalize here.
            if embeddings and isinstance(embeddings[0], float):
                return [embeddings]  # type: ignore[list-item]
            return embeddings  # type: ignore[return-value]

        return await asyncio.to_thread(_call)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed_in_batches(texts, task_type="retrieval_document")

    async def embed_query(self, text: str) -> list[float]:
        result = await self._embed_batch([text], task_type="retrieval_query")
        return result[0]

    async def _embed_in_batches(self, texts: list[str], task_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            logger.debug("embedding_batch", size=len(batch), offset=i)
            out.extend(await self._embed_batch(batch, task_type=task_type))
        return out


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    genai.configure(api_key=settings.google_api_key)
    return EmbeddingClient(model=settings.embedding_model)
