from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI


def get_client() -> OpenAI:
    """Create an OpenAI client using OPENAI_API_KEY from the environment.

    We load .env (if present) to support local development.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY no encontrada. Crea un archivo .env en la raíz con OPENAI_API_KEY=..."
        )
    return OpenAI(api_key=api_key)
