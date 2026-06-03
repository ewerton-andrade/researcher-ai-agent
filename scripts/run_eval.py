"""Run the 5 spec evaluation questions against the live API."""

from __future__ import annotations

import asyncio
import os

import httpx

QUESTIONS: list[str] = [
    "Qual é o mecanismo central proposto no paper Attention Is All You Need e como ele se diferencia de RNNs?",
    "Como o RAG combina recuperação e geração? Quais são suas limitações apontadas pelos autores?",
    "Compare a abordagem do ReAct com a do Toolformer para uso de ferramentas em LLMs.",
    "Qual paper você considera mais relevante para construir um agente com uso de ferramentas externas? Justifique com base nos textos.",
    "Faça um resumo executivo dos 5 papers em no máximo 5 bullet points cada.",
]


async def main() -> None:
    base_url = os.environ.get("EVAL_API_URL", "http://localhost:8080")
    async with httpx.AsyncClient(base_url=base_url, timeout=300.0) as client:
        thread_resp = await client.post("/threads")
        thread_resp.raise_for_status()
        thread_id = thread_resp.json()["thread_id"]
        print(f"Thread: {thread_id}\n")

        for i, question in enumerate(QUESTIONS, start=1):
            print(f"=== Question {i} ===")
            print(f"Q: {question}")
            resp = await client.post(
                f"/threads/{thread_id}/messages",
                json={"content": question},
            )
            resp.raise_for_status()
            data = resp.json()
            print(f"A: {data['response']}\n")


if __name__ == "__main__":
    asyncio.run(main())
