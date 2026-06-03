"""OrchestratorAgent: routes user questions to specialist agents via function calling."""

from __future__ import annotations

from typing import Any

from google.generativeai.types import FunctionDeclaration

from researcher.agents.analyst_agent import AnalystAgent
from researcher.agents.rag_agent import RAGAgent
from researcher.core.models import AgentResponse, Role, StoredMessage, ToolCallTrace
from researcher.infra.llm import GeminiChat, ToolBinding

_SYSTEM = """You are OrchestratorAgent. You answer the user by delegating to specialist agents.

You have two function tools:
- `ask_rag_agent(question)`: ask the retrieval specialist factual or grounded questions
  about the 5 indexed ML papers. Use it for "what does paper X say about Y?".
- `ask_analyst_agent(task)`: ask the analyst for comparisons, summaries or rankings
  across papers. Use it for "compare", "summarize", "rank" requests.

Workflow:
1. Read the conversation history (previous turns are provided to you).
2. Decide which specialist(s) to call. You may call both, possibly in the same turn
   (they will be executed in parallel) when their work is independent.
3. Use their outputs to compose a clear, well-structured final answer for the user.
4. Always answer in the same language as the latest user message (Portuguese or English).
5. Never fabricate content not supported by the agents' replies.
"""


def _ask_agent_declaration(name: str, description: str, arg: str) -> FunctionDeclaration:
    return FunctionDeclaration(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                arg: {
                    "type": "string",
                    "description": "Self-contained question or task in natural language.",
                }
            },
            "required": [arg],
        },
    )


class OrchestratorAgent:
    """Top-level agent driving the conversation."""

    def __init__(
        self,
        model_name: str,
        rag_agent: RAGAgent,
        analyst_agent: AnalystAgent,
        max_steps: int = 6,
    ) -> None:
        self._rag = rag_agent
        self._analyst = analyst_agent

        bindings = [
            ToolBinding(
                declaration=_ask_agent_declaration(
                    "ask_rag_agent",
                    "Delegate a factual/grounded question about the 5 ML papers to the "
                    "retrieval specialist (RAGAgent).",
                    "question",
                ),
                executor=self._exec_rag,
            ),
            ToolBinding(
                declaration=_ask_agent_declaration(
                    "ask_analyst_agent",
                    "Delegate a comparison, summarization or ranking task across papers "
                    "to the analyst (AnalystAgent).",
                    "task",
                ),
                executor=self._exec_analyst,
            ),
        ]
        self._chat = GeminiChat(
            model_name=model_name,
            system_instruction=_SYSTEM,
            bindings=bindings,
            max_steps=max_steps,
        )

    async def respond(
        self,
        user_message: str,
        history: list[StoredMessage],
    ) -> AgentResponse:
        gemini_history = _history_to_gemini(history)
        outcome = await self._chat.run_loop(user_message=user_message, history=gemini_history)
        return AgentResponse(text=outcome.text, trace=outcome.trace)

    async def _exec_rag(self, _name: str, args: dict[str, Any]) -> dict[str, Any]:
        question = str(args.get("question", "")).strip()
        if not question:
            return {"ok": False, "error": "Missing 'question'."}
        result = await self._rag.handle(question)
        return {"ok": True, "agent": "rag_agent", "answer": result.text}

    async def _exec_analyst(self, _name: str, args: dict[str, Any]) -> dict[str, Any]:
        task = str(args.get("task", "")).strip()
        if not task:
            return {"ok": False, "error": "Missing 'task'."}
        result = await self._analyst.handle(task)
        return {"ok": True, "agent": "analyst_agent", "answer": result.text}


def _history_to_gemini(history: list[StoredMessage]) -> list[dict[str, Any]]:
    """Convert stored messages into Gemini `contents` history format.

    Only user/assistant turns are forwarded; tool/system messages are excluded
    (system instruction is already configured on the model).
    """

    out: list[dict[str, Any]] = []
    for msg in history:
        if msg.role == Role.USER:
            out.append({"role": "user", "parts": [{"text": msg.content}]})
        elif msg.role == Role.ASSISTANT:
            out.append({"role": "model", "parts": [{"text": msg.content}]})
    return out


__all__ = ["OrchestratorAgent", "ToolCallTrace"]
