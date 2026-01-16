from __future__ import annotations

import os
from pathlib import Path

from src.chunking import chunk_text_simple
from src.embeddings import embed_texts, embed_text
from src.similarity import dot_similarity

TOP_K = 3
DOCS_DIR = Path("data/sample_docs")


def read_docs() -> list[tuple[str, str]]:
    docs = []
    for fp in sorted(DOCS_DIR.glob("*.md")):
        docs.append((fp.stem, fp.read_text(encoding="utf-8")))
    if not docs:
        raise FileNotFoundError(f"No se encontraron .md en {DOCS_DIR}")
    return docs


def main() -> None:
    docs = read_docs()

    # 1) chunking
    chunks = []
    for doc_id, text in docs:
        chunks.extend(chunk_text_simple(text, doc_id=doc_id, max_chars=600, overlap=80))

    # 2) embeddings de chunks
    chunk_texts = [c.text for c in chunks]
    chunk_embs = embed_texts(chunk_texts)

    # 3) embedding de la pregunta
    question = "¿Qué aprendimos para mejorar el reclutamiento y la tasa de respuesta?"
    q_emb = embed_text(question)

    # 4) similitud
    scored = []
    for c, e in zip(chunks, chunk_embs):
        scored.append((dot_similarity(q_emb, e), c))

    scored.sort(key=lambda x: x[0], reverse=True)

    print("\nPregunta:")
    print(question)
    print("\nTop chunks recuperados:\n")

    for rank, (score, c) in enumerate(scored[:TOP_K], start=1):
        preview = c.text.replace("\n", " ")
        preview = (preview[:220] + "...") if len(preview) > 220 else preview
        print(f"#{rank}  score={score:.4f}  doc={c.doc_id}  chunk={c.chunk_id}")
        print(preview)
        print("---")


if __name__ == "__main__":
    main()
