"""Loader for plain .txt files."""

from typing import List, Dict


def load_txt(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    if not text:
        return []
    return [{"text": text, "source": file_path, "page": None}]
