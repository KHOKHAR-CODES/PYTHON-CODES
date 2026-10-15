"""Loader for .md / .markdown files. Strips common markdown syntax so
the embedded text is cleaner prose rather than raw markup."""

import re
from typing import List, Dict


def load_markdown(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)  # headers
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # code fences
    text = re.sub(r"[*_`]", "", text)  # bold/italic/inline-code markers
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)  # links -> link text
    text = text.strip()

    if not text:
        return []
    return [{"text": text, "source": file_path, "page": None}]
