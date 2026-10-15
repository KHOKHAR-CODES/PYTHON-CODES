"""Loader for .html / .htm files. Strips scripts/styles and tags,
keeping only visible text content."""

from typing import List, Dict

from bs4 import BeautifulSoup


def load_html(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    if not text:
        return []
    return [{"text": text, "source": file_path, "page": None}]
