"""Loader for PDF files. Extracts text page-by-page so citations can
reference the exact page a chunk came from."""

from typing import List, Dict

from pypdf import PdfReader


def load_pdf(file_path: str) -> List[Dict]:
    """Load a PDF and return one record per non-empty page.

    Each record: {"text": str, "source": file_path, "page": int}
    """
    reader = PdfReader(file_path)
    records = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            records.append({"text": text, "source": file_path, "page": i})
    return records
