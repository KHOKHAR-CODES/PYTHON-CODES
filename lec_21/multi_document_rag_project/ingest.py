"""Ingestion pipeline CLI: load documents from data/, split into chunks,
embed with a Hugging Face model, and save to the local FAISS vector store.

Run this once whenever you add or change files in data/:
    python ingest.py

The heavy lifting lives in ingest_service.run_ingestion, which is also
what the Streamlit app (app.py) calls when chunk settings change.
"""

import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from ingest_service import CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_STORE_DIR, run_ingestion


def main():
    stats = run_ingestion(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        include_data_dir=True,
    )
    print(f"[ingest] Done. {stats['chunks']} chunk(s) from {stats['records']} record(s), "
          f"dim {stats['dimension']}, saved to '{VECTOR_STORE_DIR}'.")
    return stats


if __name__ == "__main__":
    main()
