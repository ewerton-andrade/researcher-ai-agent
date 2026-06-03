"""Download the 5 required arXiv PDFs into the local PDF directory."""

from __future__ import annotations

import asyncio
import os

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from researcher.core.logging import configure_logging, get_logger
from researcher.core.papers import PAPERS, PaperRef
from researcher.core.settings import get_settings

logger = get_logger(__name__)

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=10))
async def _download_one(client: httpx.AsyncClient, paper: PaperRef, target_dir: str) -> str:
    out_path = os.path.join(target_dir, f"{paper.paper_id}.pdf")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        logger.info("pdf_already_present", paper_id=paper.paper_id, path=out_path)
        return out_path

    url = ARXIV_PDF_URL.format(arxiv_id=paper.arxiv_id)
    logger.info("downloading_pdf", paper_id=paper.paper_id, url=url)
    response = await client.get(url, follow_redirects=True, timeout=60.0)
    response.raise_for_status()
    with open(out_path, "wb") as fh:
        fh.write(response.content)
    logger.info("downloaded_pdf", paper_id=paper.paper_id, bytes=len(response.content))
    return out_path


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    os.makedirs(settings.pdf_dir, exist_ok=True)

    headers = {"User-Agent": "researcher-ai-agent/0.1 (academic use)"}
    async with httpx.AsyncClient(headers=headers) as client:
        await asyncio.gather(
            *(_download_one(client, paper, settings.pdf_dir) for paper in PAPERS)
        )
    logger.info("download_complete", count=len(PAPERS), pdf_dir=settings.pdf_dir)


if __name__ == "__main__":
    asyncio.run(main())
