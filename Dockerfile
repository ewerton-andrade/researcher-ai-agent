# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build essentials only if needed for any wheel; keep slim.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip \
 && pip install .

# Non-root user
RUN useradd --create-home --uid 1000 appuser \
 && mkdir -p /data/pdfs /data/sqlite \
 && chown -R appuser:appuser /app /data
USER appuser

COPY --chown=appuser:appuser scripts ./scripts

EXPOSE 8080

CMD ["uvicorn", "researcher.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
