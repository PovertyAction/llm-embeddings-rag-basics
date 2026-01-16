from __future__ import annotations

from typing import Iterable

from src.openai_client import get_client


DEFAULT_EMBED_MODEL = "text-embedding-3-large"


def embed_text(text: str, model: str = DEFAULT_EMBED_MODEL) -> list[float]:
    """Create a single embedding vector for the provided text."""
    client = get_client()
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding


def embed_texts(texts: Iterable[str], model: str = DEFAULT_EMBED_MODEL) -> list[list[float]]:
    """Create embeddings for a list/iterable of texts.

    NOTE: For simplicity, we call the API once with the list of inputs.
    """
    client = get_client()
    resp = client.embeddings.create(model=model, input=list(texts))
    return [item.embedding for item in resp.data]
