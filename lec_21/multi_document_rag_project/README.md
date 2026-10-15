# Multi-Document RAG System (Hugging Face + FAISS + RAGAS)

A Retrieval-Augmented Generation system that answers questions from PDF,
DOCX, TXT, CSV, JSON, Markdown, and HTML documents — built entirely on
Hugging Face models (no OpenAI key required) with a local FAISS vector
store, and evaluated with both manual scoring and RAGAS.

## How it works

'''
Documents -> Load & Extract Text -> Chunk -> Embed (Hugging Face) -> FAISS
|
User Question -> Embed -> Similarity Search -----------------------------
|
Top-K Relevant Chunks
|
Prompt + Context -> Hugging Face LLM
|
Answer + Source Citations

'''
Evaluation runs the same pipeline over a small labeled dataset and scores
**Faithfulness**, **Answer Relevance**, **Context Precision**, and
**Context Recall** — first manually, then automatically with RAGAS.

## Project structure

'''
multi_document_rag_project/
├── data/                  # Source documents (sample files included)
├── loaders/               # One loader per file type
├── src/                   # Core RAG pipeline (unchanged retrieval logic)
├── evaluation/            # Manual + RAGAS evaluation
├── results/               # Evaluation output CSVs
├── vector_store/          # Saved FAISS index + metadata
├── app.py                 # Streamlit web app (chat, selectors, eval tab)
├── commands.py            # Shared slash-command layer (CLI + web)
├── ingest.py              # CLI: build the vector store from data/
├── ingest_service.py      # Reusable ingestion used by CLI and web app
├── main.py                # Interactive terminal app with /commands
└── requirements.txt

''''

## Setup

1. **Install dependencies** (Python 3.10+ recommended):

   ```bash
   pip install -r requirements.txt
   ```

'''

''''

2.\*Configure environment\*\*:

```bash
cp .env.example .env
```

''''
The defaults work with no API key at all: embeddings run locally via
`sentence-transformers/all-MiniLM-L6-v2`, and the LLM runs locally via
`google/flan-t5-base` (`LLM_PROVIDER=local`).

To use larger hosted models instead of running them on your machine,
set `LLM_PROVIDER=inference_api`, add your `HF_TOKEN` (from
<https://huggingface.co/settings/tokens>), and point `LLM_MODEL` at a
model available on the Hugging Face Inference API.

### Your Hugging Face token (`HF_TOKEN`)

> ⚠️ **The original file shipped with a working token.** It has been
> **blanked out** because committing a real key is a security risk:
> anyone with a copy of the repo could use — and burn — your API
> quota. Treat any token that has ever been committed or shared as
> **exposed**, even after you replace it.

**Do you need a token at all?**

| I want to…                                 | `HF_TOKEN` needed? |
| ------------------------------------------- | ------------------ |
| Embeddings only (`LLM_PROVIDER=local`)      | ❌ No               |
| Local LLM (`google/flan-t5-base`)           | ❌ No               |
| Hosted LLM (`LLM_PROVIDER=inference_api`)   | ✅ Yes              |
| RAGAS evaluation (`/eval`, app tab, script) | ✅ Yes              |
| `webapp.py` / terminal chat (local mode)    | ❌ No               |

The **default `local` setup needs no token at all.** You only need one if
you switch `LLM_PROVIDER=inference_api` or run RAGAS evaluation.

**How to regenerate a token**

1. Go to <https://huggingface.co/settings/tokens> and sign in.
2. Click **Create new token**.
3. Give it a name (e.g. `knowledge-desk`) and pick a role — **Read** is
   enough for downloading/gated models and the Inference API; choose
   **Write** only if you also plan to push models or datasets.
4. Click **Create token**, then copy the value (it starts with `hf_`).
5. Paste it into `.env`:

   ```bash
   # .env  (git-ignored — this is the file the app actually reads)
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   ```

6. Optionally refresh the placeholder in `.env.example` — that file is
   documentation only, so keeping it as `HF_TOKEN=your_hf_token_here` is safe.

**Security notes**

- Never commit a real token. `.env` is in `.gitignore` — keep it that way;
  only `.env.example` (placeholders) should ever be tracked.
- If a token was ever committed or shared (like the one that shipped in
  this repo), **revoke it immediately**:
  <https://huggingface.co/settings/tokens> → **Delete** on the old token →
  create a new one. Revoking kills the key even if it has already leaked.
- The token in `.env` is read only locally and is never printed; it is
  sent only to the Hugging Face API when you call a hosted model.

3.**Add your documents** to `data/` (sample HR-policy files are already
there so you can try the system immediately), then build the vector
store:

```bash
python ingest.py
```

4.**Ask questions**:

```bash
python main.py
```

To use the browser UI instead:

```bash
pip install flask
python webapp.py
```

Open `http://127.0.0.1:5000` in your browser.

Example:

` ``
Enter your question:

> How many annual leave days are allowed?

Retrieved Sources:

1. hr_data.csv
2. leave_policy.txt

Answer: Employees receive 20 paid annual leave days per year.

## Web app (Streamlit) — recommended

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens on `http://localhost:8501`. The app includes:

- **Multi-file uploader** — drop PDF / DOCX / TXT / CSV / JSON / MD / HTML
  files into the sidebar; they are ingested together with the `data/` folder.
- **Chunk selector** — sliders for chunk size, chunk overlap and Top-K. The
  index is rebuilt only when settings or files actually change (a
  `vector_store/build_config.json` marker prevents needless rebuilds), and a
  live overlap preview shows how the chunks tile a 1,000-char document.
- **Per-document selector** — a multiselect under "Search scope" that
  restricts retrieval to chosen files (an overlay filter on the shared FAISS
  index, no re-indexing needed).
- **Slash commands in chat** — `/help`, `/commands`, `/ingest`, `/list`,
  `/stats`, `/sources`, `/eval`, `/reset`, `/exit`. The same command layer
  (`commands.py`) powers the terminal app (`python main.py`).
- **Retrieved-chunk viewer** — every answer can be expanded to show the exact
  chunks used, with similarity scores and page numbers.
- **📊 Evaluation tab** — one-click RAGAS evaluation with metric cards,
  per-question charts, run-to-run score history, NaN detection (flags judge
  quota/rate-limit errors instead of showing silent blanks), and CSV export.
- **Extras** — sample starter questions, chat export to Markdown, and a
  polished custom-styled UI.

### RAGAS evaluation (in the app or terminal)

The RAGAS integration is version-safe: `evaluation/ragas_evaluation.py`
resolves the correct metric classes / dataset API at runtime, so it works on
both RAGAS 0.1.x and 0.2+ (metrics: Faithfulness, Answer Relevancy, Context
Precision, Context Recall; dataset uses the ragas 0.2+ column names `user_input`, `retrieved_contexts`, `response`, `reference` (0.1.x legacy names are still supported); the judge LLM runs on the Hugging Face Inference API *chat* endpoint (verified with meta-llama/Llama-3.1-8B-Instruct; set RAGAS_JUDGE_MODEL in .env), embeddings run locally —
`HF_TOKEN` required in `.env`; ground truth answers are required for
context recall). Install the optional extra first:

```bash
pip install -r requirements-ragas.txt
python -m evaluation.ragas_evaluation   # terminal
# or: /eval in chat, or the button in the Evaluation tab
```

## Terminal app with commands

```bash
python main.py
```

New interactive commands: `/help` `/commands` `/ingest` `/list` `/stats`
`/sources` `/eval` `/reset` `/exit`. Anything not starting with `/` is a
question. The legacy Flask app (`python webapp.py`) still works unchanged.

## Evaluation

### Manual evaluation

Runs the pipeline over `evaluation/evaluation_data.json` and walks you
through scoring each answer by hand:
\*\*

```bash
python -m evaluation.manual_evaluation
```

Results are saved to `results/evaluation_report.csv`.

### RAGAS evaluation

Runs the same dataset through RAGAS, using a Hugging Face model as the
\*\*"judge" LLM (requires `HF_TOKEN` since the judge calls the Inference API):

```bash
python -m evaluation.ragas_evaluation
```

Results are saved to `results/ragas_results.csv`.

> RAGAS evaluation is kept separate because its current dependency tree is
> not installable on every Windows/Python 3.14 setup: RAGAS 0.1.x requires
> an older NumPy wheel, while newer RAGAS releases pull in `scikit-network`,
> which may require Microsoft C++ Build Tools. The core application does not
> import RAGAS until `evaluation/ragas_evaluation.py` is run.

You can add more question/ground_truth pairs to
`evaluation/evaluation_data.json` to expand the evaluation set — `answer`
and `contexts` are filled in automatically by running the pipeline.

## Configuration reference (`.env`)

| Variable                       | Default                                  | Purpose                                      |
| ------------------------------ | ---------------------------------------- | -------------------------------------------- |
| `EMBEDDING_MODEL`              | `sentence-transformers/all-MiniLM-L6-v2` | HF embedding model                           |
| `LLM_PROVIDER`                 | `local`                                  | `local` or `inference_api`                   |
| `LLM_MODEL`                    | `google/flan-t5-base`                    | Generation model                             |
| `HF_TOKEN`                     | —                                        | Required for `inference_api` and RAGAS judge |
| `RAGAS_JUDGE_MODEL`            | `meta-llama/Llama-3.1-8B-Instruct`       | Judge LLM for RAGAS                          |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50`                             | Text splitting                               |
| `TOP_K`                        | `4`                                      | Chunks retrieved per query                   |

## Notes on model choice

- `all-MiniLM-L6-v2` is small, fast, and a strong default for embeddings.
- `flan-t5-base` runs on CPU and needs no token, which is why it's the
  default LLM — swap in a bigger local model (e.g. a Llama or Mistral
  variant via `text-generation` pipeline) or the Inference API for
  noticeably better answer quality.

```'

```

```'

```
