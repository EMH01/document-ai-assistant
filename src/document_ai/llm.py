from collections.abc import Sequence

from openai import OpenAI

from .models import Chunk

SYSTEM_INSTRUCTIONS = (
    "You are a document question-answering assistant. "
    "Answer from the supplied retrieved context only. If the context does not support an answer, "
    "say so clearly. When you use information from the context, cite the corresponding source "
    "markers such as [1] or [2]. Do not claim to have inspected parts of the document that are not "
    "included in the retrieved context."
)


def answer_question(
    client: OpenAI,
    question: str,
    retrieved: Sequence[tuple[Chunk, float]],
    model: str,
) -> str:
    if not retrieved:
        return "I could not retrieve enough document context to answer that question."

    context_parts = []
    for index, (chunk, score) in enumerate(retrieved, start=1):
        context_parts.append(
            f"[{index}] {chunk.citation} | similarity={score:.3f}\n{chunk.text}"
        )

    prompt = (
        f"Question:\n{question}\n\nRetrieved context:\n\n"
        + "\n\n".join(context_parts)
    )
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=prompt,
    )
    return response.output_text.strip()
