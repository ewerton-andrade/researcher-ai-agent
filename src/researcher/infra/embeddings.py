"""Async embedding client backed by Google's Gemini embeddings."""

from __future__ import annotations

import asyncio

from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from researcher.core.logging import get_logger
from researcher.core.settings import Settings
from researcher.infra.llm import get_genai_client

logger = get_logger(__name__)


class EmbeddingClient:
    """Wraps `genai.embed_content` with batching and retry."""

    def __init__(self, model: str, batch_size: int = 100) -> None:
        self._model = model
        self._batch_size = batch_size

    @retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=2, min=4, max=60))
    async def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        # `models.embed_content` is sync; run in a worker thread to stay async.
        def _call() -> list[list[float]]:
            model = self._model if self._model.startswith("models/") else f"models/{self._model}"
            resp = get_genai_client().models.embed_content(
                model=model,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type.upper()),
            )
            return [list(e.values) for e in resp.embeddings]

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
    return EmbeddingClient(model=settings.embedding_model)
