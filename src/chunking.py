from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a chunk of text from a document.

    Attributes:
        doc_id: Identifier of the source document
        chunk_id: Sequential identifier for this chunk within the document
        text: The text content of the chunk

    """

    doc_id: str
    chunk_id: int
    text: str


def chunk_text_simple(
    text: str, doc_id: str, max_chars: int = 700, overlap: int = 100
) -> list[Chunk]:
    """Chunk text using simple character-based chunking.

    - max_chars: chunk size in characters
    - overlap: overlap between consecutive chunks

    This is intentionally simple for Session 02.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(Chunk(doc_id=doc_id, chunk_id=chunk_id, text=chunk))
            chunk_id += 1
        if end == len(text):
            break
        start = max(0, end - overlap)

    return chunks
