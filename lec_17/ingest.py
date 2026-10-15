"""Ingestion pipeline: load documents from data/, split into chunks,
embed with a Hugging Face model, and save to the local FAISS vector store.

Run this once whenever you add or change files in data/:
    python ingest.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

from src.document_loader import load_documents_from_directory
from src.text_splitter import chunk_records
from src.embeddings import HFEmbeddings
from src.vector_database import VectorDatabase

DATA_DIR = os.getenv("DATA_DIR", "data")
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_store")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


def main():
    print(f"[ingest] Loading documents from '{DATA_DIR}'...")
    records = load_documents_from_directory(DATA_DIR)
    print(f"[ingest] Loaded {len(records)} raw text record(s).")

    chunks = chunk_records(records, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"[ingest] Created {len(chunks)} chunk(s).")

    if not chunks:
        print("[ingest] No chunks created. Add supported files to the data/ folder and try again.")
        return

    print("[ingest] Loading Hugging Face embedding model (this may download the model on first run)...")
    embeddings = HFEmbeddings()

    texts = [c["text"] for c in chunks]
    print("[ingest] Generating embeddings...")
    vectors = embeddings.embed_documents(texts)

    vector_db = VectorDatabase(
        dimension=embeddings.dimension,
        index_path=os.path.join(VECTOR_STORE_DIR, "index.faiss"),
        meta_path=os.path.join(VECTOR_STORE_DIR, "meta.pkl"),
    )
    vector_db.add(vectors, chunks)
    vector_db.save()
    print(f"[ingest] Done. Vector store saved to '{VECTOR_STORE_DIR}'.")


if __name__ == "__main__":
    main()
