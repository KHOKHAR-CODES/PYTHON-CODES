# RAG Question Answering API

## Day 7 — Conversational RAG

The project now extends the Day 6 RAG API with persistent, session-aware conversational retrieval. Existing `/ask`, `/evaluate`, `/ask-evaluate`, health, readiness, and document APIs remain available.

### Conversational architecture

```text
Question → Session → History → Session Documents → Embedding → Filtered Vector Search
         → History + Retrieved Context → Grounded Prompt → LLM → Sources → Persistent History
```

Conversation data is stored in SQLite. `sessions.sqlite3` contains chat/session state and document associations; the existing vector database continues to store document chunks and embeddings.

### Day 7 endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/chat` | Session-aware conversational RAG |
| POST | `/attach-document` | Extract, index, and attach a local TXT/Markdown/PDF/DOCX file |
| GET | `/history/{session_id}` | Read one session's conversation |
| DELETE | `/history/{session_id}` | Clear history while preserving session documents |
| POST | `/summarize-history` | Create/update a persistent conversation summary |
| GET | `/session/{session_id}/documents` | List documents attached to one session |

Example chat request:

```json
{
  "session_id": "student_001",
  "question": "What is RAG?",
  "top_k": 3
}
```

A follow-up can reuse the same session:

```json
{
  "session_id": "student_001",
  "question": "How does it use embeddings?"
}
```

Session retrieval is filtered at the vector-store query boundary, so unrelated session documents are not sent to the prompt.

### Session documents

For local development, `/attach-document` accepts a path under the approved `data/` or `uploads/` directories. Supported formats are `.txt`, `.md`, `.markdown`, `.pdf`, and `.docx`. Paths are resolved before validation to prevent traversal/symlink escapes. The existing `/documents/index` endpoint remains the lightweight JSON text/Markdown ingestion API.

### Conversation context control

The following settings bound prompt growth:

- `CHAT_HISTORY_MAX_MESSAGES`
- `CHAT_HISTORY_MAX_CHARS`
- `RECENT_MESSAGES_COUNT`
- `SUMMARY_THRESHOLD`
- `MAX_SESSION_DOCUMENTS`
- `MAX_CONTEXT_CHARS`
- `DEFAULT_TOP_K` / `MAX_TOP_K`

When the configured summary threshold is reached and no summary exists, the chat service creates a summary before building the next prompt. Manual summarization is also available through `/summarize-history`.

### Streamlit frontend

A premium dark/graphite/gold Streamlit client is included under `streamlit_app/`. It is a UI client only; all RAG, retrieval, memory, and LLM logic stays in FastAPI.

Run:

```bash
streamlit run streamlit_app/app.py
```

The UI supports session switching/new sessions, chat and follow-ups, approved file attachment, session document display, retrieval controls, history refresh, history clearing, summarization, sources, and retrieval/latency metadata. Set `STREAMLIT_API_URL` when the backend is not at `http://localhost:8000`.

### Docker

```bash
docker compose up --build
```

This starts the API on `http://localhost:8000` and Streamlit on `http://localhost:8501`. The API container mounts `vector_db/` persistently and `data/` read-only.



A modular FastAPI backend implementing a grounded RAG pipeline:

`Question → Validation → Embedding → Vector Search → Relevance Filter → Context → Grounded Prompt → LLM → Answer + Sources`

The application is designed so the embedding provider, vector store, and LLM client can be replaced independently.

## Architecture

- `api/`: HTTP routes only; no RAG business logic.
- `services/`: application logic and orchestration.
- `infrastructure/`: provider/vector-store adapters.
- `schemas/`: Pydantic request/response contracts.
- `core/`: configuration, logging, and exceptions.
- `tests/`: deterministic unit/API tests without paid external APIs.

The default vector store is a persistent SQLite-backed cosine-search adapter. It is intentionally dependency-light and can be replaced by FAISS, Chroma, Qdrant, or another vector backend behind `VectorStore`.

The default embedding provider is a deterministic local hash embedding so the project can run without downloading a model or providing a paid API key. It is a compatibility baseline, not a semantic-quality substitute for a production embedding model. For production retrieval quality, configure `EMBEDDING_PROVIDER=openai_compatible` or add another adapter implementing `EmbeddingProvider`.

The LLM is fixed to Hugging Face Inference Providers using the OpenAI-compatible `openai` SDK and model `openai/gpt-oss-20b`. Set `HF_TOKEN` in `.env`; the application uses `https://router.huggingface.co/v1` and does not call the OpenAI API directly.

## Requirements

- Python 3.13 recommended (3.11+ should work with the pinned dependencies)
- pip
- Optional: Docker

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
```

No real secrets are required for the default local configuration.

## Environment configuration

Copy `.env.example` to `.env` and set:

- `HF_TOKEN`: Hugging Face access token
- `HF_MODEL`: defaults to `openai/gpt-oss-20b`
- `LLM_TIMEOUT_SECONDS`, `LLM_MAX_OUTPUT_TOKENS`
- `EMBEDDING_PROVIDER`: `hash` or `openai_compatible`
- `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`, `EMBEDDING_DIMENSION`
- vector store and retrieval limits
- CORS/auth/rate-limit settings

Never commit `.env` or real credentials.

## Run locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

- http://localhost:8000/docs
- http://localhost:8000/redoc
- http://localhost:8000/openapi.json

## Index a document

The sample project includes a document-management API for indexing text/Markdown content. It intentionally avoids a multipart upload dependency so the core project remains lightweight.

```bash
curl -X POST http://localhost:8000/documents/index \
  -H 'Content-Type: application/json' \
  -d '{"document_name":"sample.txt","content":"FastAPI is a modern Python web framework for building APIs."}'
```

List documents:

```bash
curl http://localhost:8000/documents
```

## Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is FastAPI?","top_k":3}'
```

A successful response includes a request ID, answer, source metadata, selected/retrieved counts, and timing metadata.


## Evaluation endpoints

### `POST /evaluate`

Evaluates an already-generated response using the supplied question, answer, and source names.

```json
{
  "question": "What is FastAPI?",
  "answer": "FastAPI is a Python framework for building APIs.",
  "sources": ["sample.pdf"]
}
```

Response fields are `relevance_score`, `source_available`, `answer_supported`, and `status`. The score is calculated by the evaluator; it is not generated by the LLM.

### `POST /ask-evaluate`

Runs retrieval, grounded generation, and evaluation in one request.

```json
{
  "question": "What is RAG?",
  "top_k": 3
}
```

The response contains the generated answer, sources, retrieved chunk count, and an `evaluation` object. Weak or missing context is reported as `insufficient context` instead of being marked as strongly supported.

## No-context behavior

If no retrieved chunk meets the configured similarity threshold, the LLM is not called. The API returns a controlled grounded response:

> The available documents do not contain enough information to answer this question.

This avoids unnecessary model calls and reduces hallucination risk.

## Health and readiness

`GET /health` is a cheap liveness check.

`GET /ready` checks vector store, embedding service, and LLM initialization. It does not perform LLM generation.

## API authentication

Authentication is disabled by default for local development. To enable the built-in API-key mechanism:

```dotenv
API_KEY_ENABLED=true
API_KEY=replace-with-a-long-random-secret
```

Clients must send `X-API-Key`.

For production behind a gateway, a managed authentication layer is generally preferable. The application does not log API keys.

## Rate limiting

The optional rate limiter is in-process and therefore intended for single-instance deployments or development. Enable with:

```dotenv
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW_SECONDS=60
```

For multiple replicas, use a gateway or shared/distributed limiter.

## Document ingestion

The document API accepts text/Markdown content in JSON, enforces a configurable size limit, and detects duplicate content by SHA-256 hash. File-upload/PDF extraction is intentionally not included in this build; a future adapter can add it without changing the RAG service.

Each indexed chunk stores document ID, document name, chunk ID, page where available, source, creation time, content hash, and embedding.

## Postman

Import `postman/rag_api.postman_collection.json` and set the `base_url` environment variable to `http://localhost:8000`.

The collection includes health, readiness, root, normal questions, RAG questions, small top-k, irrelevant questions, and validation cases. Tests validate response structure rather than exact generated wording.

## Apidog / OpenAPI

Start the API, then import `http://localhost:8000/openapi.json` into Apidog, or import the generated OpenAPI JSON file exported from the running server. FastAPI supplies operation IDs, schemas, examples, descriptions, and response definitions.

## Testing

```bash
pytest -q
```

The test suite uses the mock LLM and local embedding/vector store; it does not require paid external services.

## Docker

Create `.env` first, then:

```bash
docker compose up --build
```

The API is available on port 8000 and the vector database is persisted under `./vector_db`.

## Security notes

- Secrets are environment-based.
- Error responses do not expose stack traces.
- Request IDs are generated or safely accepted only when UUID-formatted.
- Retrieved documents are treated as untrusted data in the prompt.
- Source previews are disabled by default.
- CORS is configurable rather than `*` by default.
- Upload size and extension checks are enforced.
- API keys are never returned by diagnostics.

## Known limitations

- The default SQLite/numpy vector adapter performs linear scan cosine similarity. It is suitable for small/medium local corpora, not very large-scale vector workloads.
- The default hash embedding is deterministic but has limited semantic retrieval quality. Configure a real embedding model for production.
- The optional in-process rate limiter is not distributed.
- The OpenAI-compatible adapters expect `/embeddings` and `/chat/completions` compatible endpoints.
