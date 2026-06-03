"""RAGAgent: retrieves grounded context from the indexed papers."""

from __future__ import annotations

from researcher.agents.base import Agent
from researcher.infra.embeddings import EmbeddingClient
from researcher.infra.vector_store import VectorStore
from researcher.tools.extract_section import ExtractSectionTool
from researcher.tools.search_documents import SearchDocumentsTool

_SYSTEM = """You are RAGAgent, a retrieval specialist over a fixed corpus of 5 ML papers.
You answer factual questions by calling `search_documents` (for semantic queries) and
`extract_section` (when the user asks for a specific section of a specific paper).

Rules:
- Always cite the paper_id and section of every claim you make.
- Do not invent content that is not present in the retrieved chunks.
- If retrieval returns nothing relevant, say so explicitly.
- Reply in the same language as the question (Portuguese or English).
"""


class RAGAgent(Agent):
    name = "rag_agent"
    system_instruction = _SYSTEM

    def __init__(
        self,
        model_name: str,
        embeddings: EmbeddingClient,
        store: VectorStore,
        max_steps: int = 6,
    ) -> None:
        tools = [
            SearchDocumentsTool(embeddings=embeddings, store=store),
            ExtractSectionTool(store=store),
        ]
        super().__init__(model_name=model_name, tools=tools, max_steps=max_steps)
