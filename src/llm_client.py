from __future__ import annotations

import os

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI


def get_provider() -> str:
    """Detect which LLM provider to use.

    Priority:
    1. LLM_PROVIDER env var (explicit override)
    2. Auto-detect from available API keys
    3. Default to anthropic when both keys present

    Returns:
        str: Provider name ("openai" or "anthropic")

    Raises:
        ValueError: When no API keys are configured

    """
    load_dotenv()

    # Check for explicit provider override
    explicit_provider = os.getenv("LLM_PROVIDER")
    if explicit_provider:
        provider = explicit_provider.lower()
        if provider not in ("openai", "anthropic"):
            msg = (
                f"LLM_PROVIDER debe ser 'openai' o 'anthropic', "
                f"no '{explicit_provider}'"
            )
            raise ValueError(msg)
        return provider

    # Auto-detect from available keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if anthropic_key and openai_key:
        # Both keys present - default to Anthropic
        return "anthropic"
    elif anthropic_key:
        return "anthropic"
    elif openai_key:
        return "openai"
    else:
        raise ValueError(
            "No se encontró ninguna API key. "
            "Configura OPENAI_API_KEY o ANTHROPIC_API_KEY en tu archivo .env"
        )


def get_client() -> OpenAI | Anthropic:
    """Create an authenticated client for the detected provider.

    Returns:
        OpenAI or Anthropic client instance

    """
    load_dotenv()
    provider = get_provider()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY no encontrada en el entorno")
        return OpenAI(api_key=api_key)
    else:  # anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY no encontrada en el entorno")
        return Anthropic(api_key=api_key)


def create_completion(
    client: OpenAI | Anthropic,
    provider: str,
    model: str,
    messages: list[dict],
    max_tokens: int | None = None,
    **kwargs,
) -> str:
    """Unified interface for chat completions across providers.

    Args:
        client: OpenAI or Anthropic client instance
        provider: Provider name ("openai" or "anthropic")
        model: Model name
        messages: Chat messages in standard format
        max_tokens: Maximum tokens to generate (required for Anthropic)
        **kwargs: Additional provider-specific parameters

    Returns:
        str: Generated text response

    """
    if provider == "openai":
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content

    else:  # anthropic
        # Extract system message if present
        system_message = None
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)

        # Anthropic requires max_tokens
        if max_tokens is None:
            max_tokens = 1024

        response = client.messages.create(
            model=model,
            messages=user_messages,
            system=system_message,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.content[0].text


def create_embeddings(
    client: OpenAI | Anthropic,
    provider: str,
    model: str,
    input: str | list[str],
) -> list[list[float]]:
    """Unified interface for embeddings across providers.

    Args:
        client: OpenAI or Anthropic client instance
        provider: Provider name ("openai" or "anthropic")
        model: Embedding model name
        input: Text or list of texts to embed

    Returns:
        list[list[float]]: List of embedding vectors

    Note:
        Currently only OpenAI embeddings are supported.
        Anthropic doesn't have native embeddings API.

    """
    if provider == "openai":
        # Ensure input is a list
        if isinstance(input, str):
            input = [input]

        response = client.embeddings.create(model=model, input=input)
        return [item.embedding for item in response.data]

    else:  # anthropic
        raise NotImplementedError(
            "Anthropic no tiene API de embeddings nativa. "
            "Para embeddings con Anthropic, considera usar OpenAI embeddings "
            "o integrar Voyage AI."
        )
