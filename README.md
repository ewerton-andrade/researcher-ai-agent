# Researcher AI Agent

Multi-agent system that answers analytical questions over a fixed corpus of 5 Machine Learning papers from arXiv, exposed as an async **FastAPI** REST API and runnable end-to-end with **Docker Compose**.

The stack is built around **Google Gemini 2.0 Flash** (chat + native function calling) and **ChromaDB** (HTTP, persisted via Docker volume) for retrieval. All I/O is asynchronous.

---

## 1. Architecture overview

```
                ┌──────────────────────┐
   HTTP POST    │      FastAPI         │  REST endpoints:
   ───────────▶ │  (api/routes/...)    │  POST/GET /threads
                └──────────┬───────────┘  POST/GET /threads/{id}/messages
                           │
                           ▼
                ┌──────────────────────┐
                │  OrchestratorAgent   │   reads thread history from SQLite
                │  (function calling)  │   decides which specialist(s) to call
                └──────────┬───────────┘   composes the final answer
                ┌──────────┴──────────┐
                ▼                     ▼
        ┌──────────────┐      ┌──────────────────┐
        │   RAGAgent   │      │  AnalystAgent    │
        ├──────────────┤      ├──────────────────┤
        │ search_      │      │ compare_papers   │
        │   documents  │      │ summarize        │
        │ extract_     │      │ rank_papers      │
        │   section    │      │                  │
        └──────┬───────┘      └────────┬─────────┘
               │                       │
               ▼                       ▼
        ┌────────────────────────────────────┐
        │  ChromaDB (separate container,     │
        │  persistent volume) — Gemini       │
        │  text-embedding-004 vectors        │
        └────────────────────────────────────┘
```

The corpus (5 PDFs) is downloaded once and ingested into ChromaDB with section-aware chunking. Each thread is isolated in SQLite (`sqlite+aiosqlite`) — the orchestrator receives the full thread history at every turn so follow-ups like *"pode detalhar o segundo ponto?"* work without extra plumbing.

### The 5 papers

| paper_id | arXiv | Title |
|---|---|---|
| `attention_is_all_you_need` | 1706.03762 | Attention Is All You Need |
| `bert` | 1810.04805 | BERT: Pre-training of Deep Bidirectional Transformers |
| `rag` | 2005.11401 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks |
| `react` | 2210.03629 | ReAct: Synergizing Reasoning and Acting in Language Models |
| `toolformer` | 2302.04761 | Toolformer: Language Models Can Teach Themselves to Use Tools |

---

## 2. Tools vs. Agents — distinction

**Tool** — atomic, stateless capability. One operation, typed input/output, no memory, no decisions.

> Implemented as subclasses of `researcher.tools.base.Tool[InputModel, OutputModel]`. Each tool exposes a Pydantic input/output schema, an async `_execute`, and a `function_declaration()` method that renders it as a Gemini `FunctionDeclaration`. Outputs are wrapped in a generic `ToolResult[T]`.

**Agent** — entity with a responsibility and an **exclusive** tool set. Owns a `GeminiChat` configured with its tools' declarations; runs a bounded tool-calling loop; returns a final answer to its caller.

> Implemented in `researcher.agents.base.Agent`. Each leaf agent (`RAGAgent`, `AnalystAgent`) is constructed with a fixed list of tools and a system instruction that constrains its role.

**Orchestrator** — top-level agent. Does not own retrieval/analysis tools; instead it sees the two specialist agents as two **Gemini function tools** (`ask_rag_agent`, `ask_analyst_agent`). It receives the thread history, decides which specialist(s) to call (possibly in parallel — Gemini may emit multiple `function_call`s in one turn and we execute them with `asyncio.gather`), and synthesizes the final user-facing reply.

| Layer | What it knows | What it does |
|---|---|---|
| Tool | nothing | runs one operation |
| Agent | its toolset + system prompt | drives a tool loop to fulfill a task |
| Orchestrator | the existence of the two specialists, plus thread history | routes and composes |

`AnalystAgent`'s tools (`compare_papers`, `summarize`, `rank_papers`) internally consult the vector store to ground their outputs. That retrieval is an implementation detail of those tools — `search_documents` is exclusive to `RAGAgent`, as specified.

---

## 3. Setup (run from zero)

### Prerequisites
- Docker + Docker Compose v2
- A Google AI Studio API key (free): https://aistudio.google.com/app/apikey

### Steps

```bash
# 1. Clone and enter the repo
git clone <repo-url> researcher-ai-agent
cd researcher-ai-agent

# 2. Configure
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=...

# 3. Bring everything up + download PDFs + ingest into ChromaDB
make setup

# 4. Run the 5 evaluation questions through the API
make run

# 5. Tests (unit + integration)
make test

# 6. Stop the stack (volumes preserved)
make down
```

After `make setup` the Swagger UI is at **http://localhost:8080/docs**.

### Example flow

```bash
TID=$(curl -s -X POST http://localhost:8080/threads | jq -r .thread_id)
curl -s -X POST http://localhost:8080/threads/$TID/messages \
     -H 'content-type: application/json' \
     -d '{"content":"Compare ReAct e Toolformer"}' | jq -r .response

curl -s -X POST http://localhost:8080/threads/$TID/messages \
     -H 'content-type: application/json' \
     -d '{"content":"Pode detalhar o ponto 2?"}' | jq -r .response
```

---

## 4. Technical decisions

| Choice | What | Why |
|---|---|---|
| **Agent framework** | `google-generativeai` SDK, no wrapper framework | Native Gemini function calling, minimal deps, deterministic loop control (we own the tool-call loop in [`infra/llm.py`](src/researcher/infra/llm.py)), easy to unit-test and Dockerize. Frameworks like LangGraph/CrewAI add abstraction we did not need for 3 agents and 5 tools. |
| **LLM** | `gemini-2.0-flash` | Free quota on AI Studio, native multi-function calls per turn, async-friendly. |
| **Embeddings** | Gemini `text-embedding-004` (768-d) | Same provider as the chat model, generous free quota, one-shot at ingestion (no runtime image bloat from a local model). Swap to `sentence-transformers` is a one-file change in [`infra/embeddings.py`](src/researcher/infra/embeddings.py). |
| **Vector store** | ChromaDB in its own container, HTTP client + persistent volume | The spec calls for a separate vector-store container; Chroma has a first-class async HTTP client, simple ops, and survives `make down`. |
| **Chunking** | Section-aware (heading regex) + recursive char split (~1000 chars, 150 overlap) | Enables `extract_section` to be a metadata filter (no second index). Heading regex covers standard arXiv section names (Abstract, Introduction, Method, Results, Conclusion, …). |
| **PDF parsing** | `pypdf` | Pure-Python, no system dependencies — keeps the image slim. |
| **Persistence** | SQLite via SQLAlchemy 2.0 async (`aiosqlite`) | Spec mandates SQLite; SQLAlchemy 2 gives us typed ORM with async sessions. Bind-mounted in a Docker volume. |
| **Config** | `pydantic-settings` reading `.env` | No bare `os.environ`; settings are typed, cached, and surfaced through one `Settings` object. |
| **Logging** | `structlog` (stdlib bridge) | Structured key/value logs out of the box, never any `print`. |
| **Validation** | Pydantic v2 everywhere — API schemas, tool I/O, settings, ORM-edge DTOs. |

### Tool-call loop

`infra/llm.py::GeminiChat.run_loop` implements the canonical Gemini function-calling loop:

1. Send `contents` to the model.
2. If the response contains one or more `function_call` parts, run them **in parallel** (`asyncio.gather`) via the bindings, then append a `model` turn with the `function_call` parts and a `user` turn with the corresponding `function_response` parts.
3. Loop until the model returns text only, or `AGENT_MAX_STEPS` is reached.

The same machinery drives both leaf agents (where the bindings are `Tool` instances) and the orchestrator (where the bindings proxy to whole agents).

---

## 5. Known limitations

- **No authentication / rate limiting / multi-tenancy beyond `thread_id`.** Out of scope for the brief.
- **No streaming responses (SSE/WebSocket).** Each API call waits for the full orchestrator loop before responding.
- **The Gemini free tier rate-limits embedding requests.** Ingestion uses batching (100/req) and tenacity exponential backoff, but very fast repeated `make setup` runs can still hit 429s. Wait a few seconds and retry, or switch to local embeddings.
- **Chunking quality depends on extractable text.** A small number of arXiv PDFs (LaTeX-generated) parse cleanly with `pypdf`; image-heavy or scanned ones would need OCR (not needed for these 5).
- **No observability stack** (Prometheus/OTel/traces beyond structlog console).
- **No CI workflow files** — only the local `make test` target is provided.
- **The integration test scripts the Gemini chat loop** to keep tests offline; a true live-API smoke test would require a real key and is left to `make run`.
- **Cross-thread isolation** is enforced by `thread_id` foreign keys but no row-level security / per-user separation is implemented.
