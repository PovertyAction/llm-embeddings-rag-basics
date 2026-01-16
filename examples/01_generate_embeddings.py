from __future__ import annotations

from src.embeddings import embed_text, DEFAULT_EMBED_MODEL


def main() -> None:
    text = "Un embedding es una representación numérica del significado de un texto."
    vec = embed_text(text, model=DEFAULT_EMBED_MODEL)

    print("✅ Embedding generado")
    print(f"Model: {DEFAULT_EMBED_MODEL}")
    print(f"Dimensions: {len(vec)}")
    print("First 8 values:")
    print(vec[:8])


if __name__ == "__main__":
    main()
