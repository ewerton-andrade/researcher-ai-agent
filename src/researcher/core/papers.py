"""Fixed corpus: the 5 arXiv papers the system is required to answer about."""

from __future__ import annotations

from pydantic import BaseModel


class PaperRef(BaseModel):
    paper_id: str
    arxiv_id: str
    title: str
    short: str


PAPERS: list[PaperRef] = [
    PaperRef(
        paper_id="attention_is_all_you_need",
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        short="Transformer / self-attention",
    ),
    PaperRef(
        paper_id="bert",
        arxiv_id="1810.04805",
        title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        short="BERT",
    ),
    PaperRef(
        paper_id="rag",
        arxiv_id="2005.11401",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        short="RAG",
    ),
    PaperRef(
        paper_id="react",
        arxiv_id="2210.03629",
        title="ReAct: Synergizing Reasoning and Acting in Language Models",
        short="ReAct",
    ),
    PaperRef(
        paper_id="toolformer",
        arxiv_id="2302.04761",
        title="Toolformer: Language Models Can Teach Themselves to Use Tools",
        short="Toolformer",
    ),
]


PAPERS_BY_ID: dict[str, PaperRef] = {p.paper_id: p for p in PAPERS}


def get_paper(paper_id: str) -> PaperRef:
    try:
        return PAPERS_BY_ID[paper_id]
    except KeyError as exc:
        valid = ", ".join(PAPERS_BY_ID)
        raise ValueError(f"Unknown paper_id={paper_id!r}. Valid: {valid}") from exc
