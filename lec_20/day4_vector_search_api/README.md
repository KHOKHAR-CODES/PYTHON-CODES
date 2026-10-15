# Day 4 Vector Search API

FastAPI implementation of the **retrieval stage** of a RAG pipeline: load a document,
clean and chunk its text, embed the chunks, store them in a FAISS index, and search
that index with natural language questions.

## Project structure

```
day4_vector_search_api/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app entry point
│   ├── routes.py          # /index-file and /search endpoints
│   ├── schemas.py         # Pydantic request/response models
│   ├── file_loader.py     # TXT / PDF / DOCX loading
│   ├── text_cleaner.py    # Whitespace + control-char normalization
│   ├── text_splitter.py   # Overlapping chunk splitting
│   ├── embeddings.py      # sentence-transformers embedding functions
│   ├── vector_store.py    # FAISS index wrapper with persistence
│   └── search_service.py  # Orchestrates the full pipeline
├── data/
│   └── sample.txt         # Placeholder file for quick testing
├── vector_db/              # FAISS index + metadata are persisted here
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

> Note: the first run will download the `all-MiniLM-L6-v2` sentence-transformers
> model (a few hundred MB) — this happens once and is cached locally afterward.

## Running the server

From the project root (the folder containing the `app/` directory):

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`, with interactive Swagger docs
at `http://127.0.0.1:8000/docs`.

## Testing in Apidog / Postman

### 1. POST `/index-file`

**URL:** `http://127.0.0.1:8000/index-file`

**Body (JSON):**

```json
{
  "file_path": "data/sample.txt"
}
```

**Example response:**

```json
{
  "file_name": "sample.txt",
  "total_chunks": 4,
  "vectors_stored": 4,
  "status": "success"
}
```

You can also point `file_path` at any `.pdf` or `.docx` file on disk, e.g.
`"data/my_notes.pdf"`.

### 2. POST `/search`

**URL:** `http://127.0.0.1:8000/search`

**Body (JSON):**

```json
{
  "question": "How does the retrieval stage of RAG work?",
  "top_k": 3
}
```

**Example response:**

```json
{
  "question": "How does the retrieval stage of RAG work?",
  "results": [
    {
      "chunk": "At query time, the user's question is embedded using the same model...",
      "source": "sample.txt"
    }
  ]
}
```

### Error cases to try

- `/index-file` with a `file_path` that doesn't exist → `400` with a clear message.
- `/index-file` with an unsupported extension (e.g. `.csv`) → `400`.
- `/search` with `top_k: 0` → `400`.
- `/search` before indexing anything → `400` ("index is empty").

## Notes

- The vector store uses FAISS's `IndexFlatL2` (exact search) for simplicity and
  reproducibility; it's persisted as `vector_db/index.faiss` with a parallel
  `vector_db/metadata.pkl` holding chunk text, source file name, and chunk id.
- Re-indexing appends new chunks to the existing index rather than overwriting it,
  so you can index multiple files before searching.
- Delete the contents of `vector_db/` to reset the index from scratch.
