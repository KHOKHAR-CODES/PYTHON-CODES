"""Ingestion service shared by the CLI (ingest.py) and the Streamlit app.

Wraps the original ingest flow in a reusable function so the web app can
rebuild the vector store whenever chunk settings or uploaded files change.
"""

import os
import shutil
import tempfile
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from src.document_loader import load_document, load_documents_from_directory
from src.text_splitter import chunk_records
from src.embeddings import HFEmbeddings
from src.vector_database import VectorDatabase

DATA_DIR = os.getenv("DATA_DIR", "data")
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_store")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


def run_ingestion(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    uploaded_files: Optional[List] = None,
    include_data_dir: bool = True,
    log=print,
) -> Dict:
    """Load documents, chunk, embed, and save the FAISS index.

    uploaded_files: optional list of Streamlit UploadedFile objects (they
    carry .name and .getbuffer()); they are written to a temp dir and
    loaded with the same per-type loaders used for data/.
    Returns stats about the new index.
    """
    records: List[Dict] = []

    if include_data_dir and os.path.isdir(DATA_DIR):
        log(f"Loading documents from '{DATA_DIR}'…")
        records.extend(load_documents_from_directory(DATA_DIR))

    if uploaded_files:
        tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
        for upload in uploaded_files:
            dest = os.path.join(tmp_dir, os.path.basename(upload.name))
            with open(dest, "wb") as f:
                f.write(upload.getbuffer())
            try:
                loaded = load_document(dest)
                records.extend(loaded)
                log(f"Loaded {len(loaded)} record(s) from {upload.name}")
            except Exception as e:
                log(f"⚠️ Could not load {upload.name}: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not records:
        raise ValueError("No documents found to ingest. Upload files or add them to data/.")

    chunks = chunk_records(records, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    log(f"Created {len(chunks)} chunk(s) from {len(records)} record(s).")

    embeddings = HFEmbeddings()
    log("Generating embeddings…")
    vectors = embeddings.embed_documents([c["text"] for c in chunks])

    vector_db = VectorDatabase(
        dimension=embeddings.dimension,
        index_path=os.path.join(VECTOR_STORE_DIR, "index.faiss"),
        meta_path=os.path.join(VECTOR_STORE_DIR, "meta.pkl"),
    )
    vector_db.add(vectors, chunks)
    vector_db.save()
    log("Index saved.")

    per_source: Dict[str, int] = {}
    for meta in vector_db.metadata:
        src = os.path.basename(str(meta.get("source")))
        per_source[src] = per_source.get(src, 0) + 1

    return {
        "chunks": len(chunks),
        "records": len(records),
        "dimension": embeddings.dimension,
        "sources": dict(sorted(per_source.items())),
    }
