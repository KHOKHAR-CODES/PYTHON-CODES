# RAG API — Retrieval-Augmented Generation Service

FastAPI-based backend + HTML/JS frontend for document-based Q&A using vector embeddings and LLM generation.

## Stack
- **Backend**: FastAPI (Python 3.14), SQLite/Numpy vector store, OpenAI-compatible LLM & embedding clients
- **Frontend**: Static HTML/JS (`frontend/index.html`, `script.js`)
- **LLM Provider**: OpenRouter (`https://openrouter.ai/api/v1`) — configured via `backend/.env`
- **Vector DB**: SQLite-backed numpy array store (`sqlite_numpy`)

## Structure
```
backend/
  app/
    api/routes_ask.py, routes_chat.py, routes_documents.py, routes_health.py
    services/rag_service.py, search_service.py, embedding_service.py, document_service.py
    infrastructure/llm_client.py, embeddings.py, vector_store.py
  tests/             # pytest suite
  .env               # keys / config (rotate if leaked)
frontend/
  index.html, script.js, style.css
```

## Quick Start
```bash
# 1. Configure backend/.env (rotate HF_TOKEN/OpenRouter key if needed)
# 2. Start backend
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# 3. Start frontend (new terminal)
cd frontend
python -m http.server 3000
# 4. Open http://localhost:3000/index.html
```

## Key Endpoints (from `backend/app/api/`)
- `POST /ask` — RAG query with retrieval
- `POST /chat` — conversational session
- `POST /documents` — upload / index document
- `GET /health` — status
- `/docs` — Swagger UI

## Recent Repairs (`docs/project_report.md`)
- Fixed unreachable empty-list checks in `vector_store.py`
- Fixed exception reference bugs in `llm_client.py`
- Pre-compiled regex in `embeddings.py`
- Added retry / FormData handling / diagnostics in `frontend/script.js`
- Configured OpenRouter base URL + model (`meta-llama/llama-3.1-8b-instruct`)
- Created startup scripts (`start.bat`, `run_backend.bat`, `run_frontend.bat`)

## Security Note
- `.env` previously contained a live `sk-or-v1-...` token — **rotate immediately** via https://openrouter.ai/keys
- `.env` should remain out of version control; verify `.gitignore`

## Verification Status
- `/health`: 200 OK
- `/docs`: accessible
- OpenRouter auth: 200 OK
- End-to-end chat: path verified; full response depends on OpenRouter billing/plan (free tier has rate limits)
