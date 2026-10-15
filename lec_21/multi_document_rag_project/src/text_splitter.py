"""Splits long text into smaller overlapping chunks before embedding.
Uses a recursive strategy: tries to split on paragraph breaks first,
then lines, then sentences, then falls back to a hard character cut."""

from typing import List, Dict

SEPARATORS = ["\n\n", "\n", ". ", " "]


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(chunk_size // 5, 0)

    def _split(piece: str, seps: List[str]) -> List[str]:
        if len(piece) <= chunk_size:
            return [piece]
        if not seps:
            step = max(chunk_size - chunk_overlap, 1)
            return [piece[i:i + chunk_size] for i in range(0, len(piece), step)]

        sep = seps[0]
        parts = piece.split(sep)
        chunks: List[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > chunk_size:
                    chunks.extend(_split(part, seps[1:]))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks

    raw_chunks = _split(text, SEPARATORS)

    final_chunks = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and chunk_overlap > 0:
            prev_tail = raw_chunks[i - 1][-chunk_overlap:]
            chunk = f"{prev_tail} {chunk}"
        chunk = chunk.strip()
        if chunk:
            final_chunks.append(chunk)
    return final_chunks


def chunk_records(records: List[Dict], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
    """Turn loader records ({text, source, page}) into chunk records
    ready for embedding, each carrying a traceable chunk_id."""
    chunked: List[Dict] = []
    for rec in records:
        pieces = split_text(rec["text"], chunk_size, chunk_overlap)
        for i, piece in enumerate(pieces):
            chunked.append({
                "text": piece,
                "source": rec.get("source"),
                "page": rec.get("page"),
                "chunk_id": f"{rec.get('source')}::{rec.get('page')}::{i}",
            })
    return chunked
