"""AnalystAgent: comparative analysis, summarization and ranking across papers."""

from __future__ import annotations

from researcher.agents.base import Agent
from researcher.infra.embeddings import EmbeddingClient
from researcher.infra.llm import GeminiChat
from researcher.infra.vector_store import VectorStore
from researcher.tools.compare_papers import ComparePapersTool
from researcher.tools.rank_papers import RankPapersTool
from researcher.tools.summarize import SummarizeTool

_SYSTEM = """You are AnalystAgent. You perform comparative analysis, summarization and
ranking across a fixed corpus of 5 ML papers, using exclusively your three tools:
`compare_papers`, `summarize`, `rank_papers`. The tools handle retrieval and grounding
internally.

Rules:
- Choose the smallest tool set that answers the request.
- Pass clean `paper_id` values (one of: attention_is_all_you_need, bert, rag, react, toolformer).
- Synthesize the tool outputs into a clear final answer in the user's language.
"""


class AnalystAgent(Agent):
    name = "analyst_agent"
    system_instruction = _SYSTEM

    def __init__(
        self,
        model_name: str,
        embeddings: EmbeddingClient,
        store: VectorStore,
        helper_chat: GeminiChat,
        max_steps: int = 8,
    ) -> None:
        tools = [
            ComparePapersTool(embeddings=embeddings, store=store, chat=helper_chat),
            SummarizeTool(embeddings=embeddings, store=store, chat=helper_chat),
            RankPapersTool(embeddings=embeddings, store=store, chat=helper_chat),
        ]
        super().__init__(model_name=model_name, tools=tools, max_steps=max_steps)
