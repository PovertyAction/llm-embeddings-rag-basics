from __future__ import annotations

from src.openai_client import get_client


def answer_with_context(question: str, context_chunks: list[str], model: str = "gpt-5-mini") -> str:
    """Ask the LLM to answer using provided context.

    We instruct the model to rely on context and to say when information is missing.
    """
    client = get_client()

    context = "\n\n---\n\n".join(context_chunks)

    messages = [
        {
            "role": "developer",
            "content": (
                "Eres un asistente útil. Responde la pregunta usando SOLO el contexto provisto. "
                "Si el contexto no contiene la respuesta, dilo explícitamente. "
                "Mantén la respuesta concisa."
            ),
        },
        {
            "role": "user",
            "content": f"PREGUNTA:\n{question}\n\nCONTEXTO:\n{context}",
        },
    ]

    resp = client.responses.create(model=model, input=messages)
    return resp.output_text
