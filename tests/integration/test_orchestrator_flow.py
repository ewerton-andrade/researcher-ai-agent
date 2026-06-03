"""Integration test: orchestrator → sub-agent → tool → response.

Drives the real OrchestratorAgent/RAGAgent code paths but with the Gemini chat
loop replaced by a scripted fake that executes the bindings deterministically.
This validates the wiring (tool declarations, executor proxying, history) end-to-end.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from researcher.agents.analyst_agent import AnalystAgent
from researcher.agents.orchestrator import OrchestratorAgent
from researcher.agents.rag_agent import RAGAgent
from researcher.core.models import RetrievedChunk
from researcher.infra.llm import GeminiChat, LoopOutcome


def _scripted_loop(
    plan: list[tuple[str, dict[str, Any]] | str],
):
    """Build a `run_loop` substitute that walks through a plan.

    Each entry is either a `(tool_name, args)` tuple — meaning "issue this function
    call" — or a string, meaning "emit this final text and stop".
    """

    plan = list(plan)

    async def run_loop(
        self: GeminiChat,
        user_message: str,
        history: list[dict[str, Any]] | None = None,
    ) -> LoopOutcome:
        trace = []
        for step in plan:
            if isinstance(step, str):
                return LoopOutcome(text=step, trace=trace)
            tool_name, args = step
            binding = self._bindings[tool_name]  # type: ignore[attr-defined]
            await binding.executor(tool_name, args)
            from researcher.core.models import ToolCallTrace
            trace.append(ToolCallTrace(tool=tool_name, arguments=args, ok=True))
        return LoopOutcome(text="(end)", trace=trace)

    return run_loop


@pytest.mark.asyncio
async def test_orchestrator_delegates_to_rag_agent_and_returns() -> None:
    # Mocked infra used by RAGAgent's tools.
    embeddings = AsyncMock()
    embeddings.embed_query.return_value = [0.0]
    store = AsyncMock()
    store.query.return_value = [
        RetrievedChunk(
            paper_id="attention_is_all_you_need",
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            section="abstract",
            chunk_index=0,
            text="The Transformer relies on self-attention instead of recurrence.",
        )
    ]

    # Stub `genai.GenerativeModel` so Agent construction does not contact the network.
    with patch("researcher.infra.llm.genai.GenerativeModel"):
        rag_agent = RAGAgent(model_name="m", embeddings=embeddings, store=store)
        analyst_agent = AnalystAgent(
            model_name="m",
            embeddings=embeddings,
            store=store,
            helper_chat=AsyncMock(spec=GeminiChat),
        )
        orchestrator = OrchestratorAgent(
            model_name="m",
            rag_agent=rag_agent,
            analyst_agent=analyst_agent,
        )

    # Script the RAGAgent: it should call its `search_documents` tool, then answer.
    rag_loop = _scripted_loop(
        [
            ("search_documents", {"query": "transformer self-attention", "k": 3}),
            "The Transformer replaces recurrence with self-attention.",
        ]
    )
    # Script the Orchestrator: it should delegate to ask_rag_agent, then return.
    orch_loop = _scripted_loop(
        [
            ("ask_rag_agent", {"question": "What is the core mechanism of the Transformer?"}),
            "Final answer: the Transformer uses self-attention.",
        ]
    )

    with (
        patch.object(GeminiChat, "run_loop", autospec=True) as mocked_run_loop,
    ):
        # autospec=True so first arg is `self`; route per-instance.
        async def dispatch(self: GeminiChat, user_message: str, history=None) -> LoopOutcome:
            if self is orchestrator._chat:  # type: ignore[attr-defined]
                return await orch_loop(self, user_message, history)
            if self is rag_agent._chat:  # type: ignore[attr-defined]
                return await rag_loop(self, user_message, history)
            return LoopOutcome(text="", trace=[])

        mocked_run_loop.side_effect = dispatch

        response = await orchestrator.respond(
            user_message="What is the core mechanism of the Transformer?",
            history=[],
        )

    assert "self-attention" in response.text.lower()
    # Orchestrator's trace records its delegation to the RAG sub-agent.
    assert any(t.tool == "ask_rag_agent" and t.ok for t in response.trace)
    # The underlying retrieval tool was actually invoked through the binding chain.
    embeddings.embed_query.assert_awaited()
    store.query.assert_awaited()
