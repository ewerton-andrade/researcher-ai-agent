"""Ingestion pipeline: PDF → section-aware chunks → embeddings → ChromaDB."""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass

from pypdf import PdfReader

from researcher.core.logging import configure_logging, get_logger
from researcher.core.papers import PAPERS, PaperRef
from researcher.core.settings import Settings, get_settings
from researcher.infra.embeddings import EmbeddingClient, build_embedding_client
from researcher.infra.llm import configure_genai
from researcher.infra.vector_store import VectorStore

logger = get_logger(__name__)


SECTION_HEADINGS = [
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "approach",
    "model",
    "architecture",
    "experiments",
    "experimental setup",
    "results",
    "evaluation",
    "discussion",
    "analysis",
    "conclusion",
    "conclusions",
    "limitations",
    "references",
    "acknowledgments",
    "acknowledgements",
    "appendix",
]

_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s+)?(" + "|".join(SECTION_HEADINGS) + r")\b\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


@dataclass(slots=True)
class Chunk:
    paper: PaperRef
    section: str
    chunk_index: int
    text: str


def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("pdf_page_extract_failed", error=str(exc))
            pages.append("")
    return "\n".join(pages)


def _split_by_section(text: str) -> list[tuple[str, str]]:
    """Return list of (section_name, section_body) using detected headings."""

    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("body", text)]

    sections: list[tuple[str, str]] = []
    # Prefix before the first heading (often title + authors) → keep as "frontmatter"
    if matches[0].start() > 0:
        prefix = text[: matches[0].start()].strip()
        if prefix:
            sections.append(("frontmatter", prefix))

    for i, match in enumerate(matches):
        name = match.group(1).lower().strip()
        # Normalize plurals/variants
        name = {"methods": "method", "methodology": "method", "conclusions": "conclusion",
                "acknowledgements": "acknowledgments"}.get(name, name)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((name, body))
    return sections


def _chunk_text(text: str, max_chars: int = 1000, overlap: int = 150) -> list[str]:
    """Recursive-ish character split with paragraph awareness."""

    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{para}" if buffer else para
            continue
        if buffer:
            chunks.append(buffer)
        if len(para) <= max_chars:
            buffer = para
        else:
            # Hard split long paragraph with overlap.
            step = max_chars - overlap
            for i in range(0, len(para), step):
                chunks.append(para[i : i + max_chars])
            buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


def _build_chunks(paper: PaperRef, raw_text: str) -> list[Chunk]:
    out: list[Chunk] = []
    counter = 0
    for section, body in _split_by_section(raw_text):
        # Skip noisy/irrelevant sections.
        if section in {"references"}:
            continue
        for piece in _chunk_text(body):
            piece = piece.strip()
            if len(piece) < 50:
                continue
            out.append(Chunk(paper=paper, section=section, chunk_index=counter, text=piece))
            counter += 1
    return out


async def _ingest_paper(
    paper: PaperRef,
    settings: Settings,
    embeddings: EmbeddingClient,
    store: VectorStore,
) -> int:
    path = os.path.join(settings.pdf_dir, f"{paper.paper_id}.pdf")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PDF for {paper.paper_id} not found at {path}. Run download_papers.py first."
        )
    logger.info("ingesting_paper", paper_id=paper.paper_id)
    text = await asyncio.to_thread(_read_pdf, path)
    chunks = _build_chunks(paper, text)
    if not chunks:
        logger.warning("no_chunks_produced", paper_id=paper.paper_id)
        return 0

    vectors = await embeddings.embed_documents([c.text for c in chunks])
    ids = [f"{paper.paper_id}:{c.chunk_index}" for c in chunks]
    metadatas = [
        {
            "paper_id": c.paper.paper_id,
            "arxiv_id": c.paper.arxiv_id,
            "title": c.paper.title,
            "section": c.section,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]
    await store.upsert(
        ids=ids,
        documents=[c.text for c in chunks],
        embeddings=vectors,
        metadatas=metadatas,
    )
    logger.info("ingested_paper", paper_id=paper.paper_id, chunks=len(chunks))
    return len(chunks)


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_genai(settings)
    embeddings = build_embedding_client(settings)
    store = await VectorStore.connect(settings)

    # Fresh ingest each run to keep the corpus consistent.
    await store.reset_collection()

    totals: list[int] = []
    for paper in PAPERS:
        totals.append(await _ingest_paper(paper, settings, embeddings, store))
    total_count = await store.count()
    logger.info(
        "ingestion_complete",
        per_paper=dict(zip([p.paper_id for p in PAPERS], totals, strict=True)),
        total_chunks=total_count,
    )


if __name__ == "__main__":
    asyncio.run(main())
