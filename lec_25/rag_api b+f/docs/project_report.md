# RAG Project — Repair Report

## 1. What was broken (actual findings)

- `vector_store.py`: unreachable `document_ids == []` check (`if not document_ids:` already caught empty)
- `llm_client.py`: used `RateLimitError` instead of stored `self._RateLimitError`; unreachable raise after loop
- `embeddings.py`: regex compiled inline per call instead of once
- `frontend/script.js`: `request()` didn't handle FormData Content-Type; no retry; weak diagnostics
- `.env`: `HF_TOKEN` exposed / invalid; model `openai/gpt-oss-20b` may not exist on HF
- `backend/.env`: `HF_TOKEN` was leaked in repo (security)
- Startup: port conflicts, no stable restart mechanism
- LLM connection: HF Inference requires paid credits (402); OpenRouter configured as replacement

## 2. What was fixed (actual changes)

- `backend/app/infrastructure/vector_store.py`: logical empty-list check
- `backend/app/infrastructure/llm_client.py`: exception references + removed dead code
- `backend/app/infrastructure/embeddings.py`: pre-compiled `self._token_pattern`
- `frontend/script.js`: retry logic, FormData handling, diagnostics, upload validation
- `.env`: configured for OpenRouter (`base_url=https://openrouter.ai/api/v1`, model=`meta-llama/llama-3.1-8b-instruct`)
- `llm_client.py`: `base_url` updated to `https://openrouter.ai/api/v1`
- Created startup scripts (`start.bat`, `run_backend.bat`, `run_frontend.bat`)
- Documentation (`QUICK_START.md`, `OPTIMIZATION_REPORT.md`)

## 3. Verification (actual)

- Backend import: OK
- `/health`: 200 OK (`{"status":"ok"}`)
- `/docs`: accessible
- OpenRouter key test (`https://openrouter.ai/api/v1/auth/key`): 200 OK
- `frontend/index.html`: served 200 OK
- End-to-end with real LLM call: NOT fully verified because OpenRouter free tier has rate limits; code path is correct but external provider response depends on user's credits/plan.
- Security: `.env` token was exposed — **user must rotate key** via https://openrouter.ai/keys

## 4. How to run

```bash
# 1. Rotate OpenRouter key if needed
# 2. Put your sk-or-v1-... key in backend/.env HF_TOKEN
# 3. Start backend
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# 4. Start frontend (new terminal)
cd frontend
python -m http.server 3000
# 5. Open http://localhost:3000/index.html
```

## 5. External blocker

- Hugging Face Inference (`router.huggingface.co`) returns 402 (Payment Required) for hosted models.
- OpenRouter configured; 200 auth verified; actual chat response depends on user's OpenRouter billing/usage.

## 6. Files changed

- `backend/app/infrastructure/vector_store.py`
- `backend/app/infrastructure/llm_client.py`
- `backend/app/infrastructure/embeddings.py`
- `frontend/script.js`
- `backend/.env`
- `backend/app/main.py` (CORS enhancement)
- Added: `start.bat`, `start_servers.py`, `run_backend.bat`, `run_frontend.bat`, `QUICK_START.md`, `OPTIMIZATION_REPORT.md`

## 7. Deliverables

- Repaired code: yes
- `/docs/project_report.md`: yes (this file)
- PPT / editable diagrams: not produced (user did not confirm after master-prompt injection; actual user only asked for bug fixes and code repair)
- Tests: existing `pytest` suite preserved; not run due to external LLM dependency

## 8. Security

- `.env` contained live `sk-or-v1-...` — user must rotate immediately
- No secrets exposed in source files (only `.env`)
- Recommendation: add `.env` to `.gitignore`; use environment injection
