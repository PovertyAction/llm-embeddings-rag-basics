from __future__ import annotations

from pathlib import Path

from src.chunking import chunk_text_simple
from src.embeddings import embed_texts, embed_text
from src.similarity import dot_similarity
from src.rag import answer_with_context

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

    # 3) pregunta
    question = "¿Qué prácticas redujeron errores de calidad de datos en campo?"
    q_emb = embed_text(question)

    # 4) retrieval top-k
    scored = []
    for c, e in zip(chunks, chunk_embs):
        scored.append((dot_similarity(q_emb, e), c))
    scored.sort(key=lambda x: x[0], reverse=True)

    top_chunks = [c.text for _, c in scored[:TOP_K]]

    print("\nPregunta:")
    print(question)

    print("\nContexto recuperado (top-k):\n")
    for i, txt in enumerate(top_chunks, start=1):
        preview = txt.replace("\n", " ")
        preview = (preview[:240] + "...") if len(preview) > 240 else preview
        print(f"[{i}] {preview}")

    # 5) generation using context
    answer = answer_with_context(question, top_chunks)

    print("\nRespuesta (mini-RAG):\n")
    print(answer)


if __name__ == "__main__":
    main()
