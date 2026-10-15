"""Split cleaned text into overlapping chunks for embedding."""

from typing import List


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks of a target character length.

    Args:
        text: The cleaned text to split.
        chunk_size: Maximum number of characters per chunk. Defaults to 500.
        overlap: Number of characters each chunk overlaps with the previous one.
            Defaults to 50.

    Returns:
        A list of text chunks. Empty if the input text is empty.

    Raises:
        ValueError: If chunk_size is not positive, or overlap is negative or
            greater than or equal to chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if overlap < 0:
        raise ValueError("overlap must be a non-negative integer.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    if not text:
        return []

    chunks: List[str] = []
    step = chunk_size - overlap
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start += step

    return chunks
