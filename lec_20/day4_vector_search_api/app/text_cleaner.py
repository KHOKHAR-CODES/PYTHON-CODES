"""Normalize and clean raw extracted text before chunking."""

import re
import unicodedata


def clean_text(raw_text: str) -> str:
    """Normalize whitespace, strip control characters, and remove empty lines.

    Args:
        raw_text: The raw text extracted from a source file.

    Returns:
        Cleaned text with normalized whitespace and no empty lines.
    """
    if not raw_text:
        return ""

    # Strip control characters (category "C*") except for standard whitespace.
    stripped = "".join(
        char for char in raw_text
        if unicodedata.category(char)[0] != "C" or char in ("\n", "\t", " ")
    )

    lines = stripped.splitlines()
    cleaned_lines = []
    for line in lines:
        # Collapse repeated whitespace within a line.
        normalized_line = re.sub(r"[ \t]+", " ", line).strip()
        if normalized_line:
            cleaned_lines.append(normalized_line)

    return "\n".join(cleaned_lines)
