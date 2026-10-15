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
├── data/ # Source documents (sample files included)
├── loaders/ # One loader per file type
├── src/ # Core RAG pipeline
├── evaluation/ # Manual + RAGAS evaluation
├── results/ # Evaluation output CSVs
├── vector_store/ # Saved FAISS index + metadata
├── ingest.py # Build the vector store from data/
├── main.py # Ask questions interactively
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
| `RAGAS_JUDGE_MODEL`            | `mistralai/Mistral-7B-Instruct-v0.2`     | Judge LLM for RAGAS                          |
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
