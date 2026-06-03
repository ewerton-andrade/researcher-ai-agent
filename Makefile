SHELL := /bin/bash

COMPOSE ?= docker compose
API_URL ?= http://localhost:$${API_PORT:-8080}

.PHONY: help setup build up ingest run test down logs lint clean

help:
	@echo "Targets:"
	@echo "  setup   - copy .env, build images, start stack, download PDFs, ingest into Chroma"
	@echo "  run     - execute the 5 evaluation questions against the running API"
	@echo "  test    - run pytest suite inside the api container"
	@echo "  down    - stop containers (volumes preserved)"
	@echo "  logs    - tail container logs"
	@echo "  lint    - run ruff inside the api container"
	@echo "  clean   - down + remove named volumes"

.env:
	@if [ ! -f .env ]; then cp .env.example .env && echo "Created .env from .env.example - edit GOOGLE_API_KEY before running."; fi

build:
	$(COMPOSE) build

up: .env
	$(COMPOSE) up -d

setup: .env build up
	@echo ">>> Waiting for API health..."
	@for i in $$(seq 1 30); do \
	  if curl -sf $(API_URL)/health >/dev/null 2>&1; then echo "API is up."; break; fi; \
	  sleep 2; \
	done
	@echo ">>> Downloading PDFs..."
	$(COMPOSE) exec -T api python scripts/download_papers.py
	@echo ">>> Ingesting into ChromaDB..."
	$(COMPOSE) exec -T api python scripts/ingest.py
	@echo ">>> Setup complete. Swagger UI: $(API_URL)/docs"

ingest:
	$(COMPOSE) exec -T api python scripts/ingest.py

run:
	$(COMPOSE) exec -T api python scripts/run_eval.py

test:
	$(COMPOSE) exec -T api pytest -q

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=200

lint:
	$(COMPOSE) exec -T api ruff check src tests

clean:
	$(COMPOSE) down -v
