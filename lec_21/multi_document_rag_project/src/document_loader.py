"""Detects file type by extension and routes to the right loader in
loaders/. Also walks a directory and loads every supported file in it."""

import os
from typing import List, Dict

from loaders.pdf_loader import load_pdf
from loaders.docx_loader import load_docx
from loaders.txt_loader import load_txt
from loaders.csv_loader import load_csv
from loaders.json_loader import load_json
from loaders.markdown_loader import load_markdown
from loaders.html_loader import load_html

LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
    ".csv": load_csv,
    ".json": load_json,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".html": load_html,
    ".htm": load_html,
}


def load_document(file_path: str) -> List[Dict]:
    """Load a single file, dispatching on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file type: '{ext}' ({file_path})")
    return loader(file_path)


def load_documents_from_directory(directory: str) -> List[Dict]:
    """Walk a directory and load every file with a supported extension.
    Unsupported files are skipped; load failures are logged and skipped
    so one bad file doesn't stop the whole ingestion run."""
    records: List[Dict] = []
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in LOADERS:
                continue
            fpath = os.path.join(root, fname)
            try:
                loaded = load_document(fpath)
                records.extend(loaded)
                print(f"[document_loader] Loaded {len(loaded)} record(s) from {fpath}")
            except Exception as e:
                print(f"[document_loader] Failed to load {fpath}: {e}")
    return records
