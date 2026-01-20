from __future__ import annotations

import os
from collections.abc import Iterable

from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def _get_embedding_client() -> OpenAI:
    """Get OpenAI client for embeddings.

    Note: We always use OpenAI for embeddings since Anthropic doesn't
    have a native embeddings API. This works regardless of which LLM
    provider is configured.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY requerida para generar embeddings. "
            "Configura OPENAI_API_KEY en tu archivo .env"
        )
    return OpenAI(api_key=api_key)


def embed_text(text: str, model: str = DEFAULT_EMBED_MODEL) -> list[float]:
    """Create a single embedding vector for the provided text.

    Note: Uses OpenAI embeddings regardless of LLM provider setting.
    """
    client = _get_embedding_client()
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding


def embed_texts(
    texts: Iterable[str],
    model: str = DEFAULT_EMBED_MODEL,
) -> list[list[float]]:
    """Create embeddings for a list/iterable of texts.

    NOTE: For simplicity, we call the API once with the list of inputs.
    Note: Uses OpenAI embeddings regardless of LLM provider setting.
    """
    client = _get_embedding_client()
    resp = client.embeddings.create(model=model, input=list(texts))
    return [item.embedding for item in resp.data]
