import base64

import fitz
from openai import OpenAI

from .models import Chunk
from .retrieval import chunk_text

IMAGE_CAPTION_PROMPT = (
    "Describe the information conveyed by this image from a PDF. Focus on facts, labels, "
    "relationships, chart trends, table-like content, or diagram structure that could help answer "
    "questions about the document. Be concise and do not invent unreadable details."
)


def caption_image(
    client: OpenAI,
    image_bytes: bytes,
    model: str,
    mime_type: str = "image/png",
) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": IMAGE_CAPTION_PROMPT},
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{encoded}",
                    },
                ],
            }
        ],
    )
    return response.output_text.strip()


def extract_chunks(
    pdf_bytes: bytes,
    source: str,
    client: OpenAI | None = None,
    vision_model: str | None = None,
    include_image_captions: bool = False,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            chunks.extend(chunk_text(page.get_text("text"), page_number, source))

            if not include_image_captions or client is None or vision_model is None:
                continue

            for image_info in page.get_images(full=True):
                image = document.extract_image(image_info[0])
                extension = image.get("ext", "png").lower()
                mime_type = "image/jpeg" if extension in {"jpg", "jpeg"} else f"image/{extension}"
                caption = caption_image(
                    client, image["image"], vision_model, mime_type=mime_type
                )
                if caption:
                    chunks.append(
                        Chunk(
                            text=caption,
                            page=page_number,
                            kind="image",
                            source=source,
                        )
                    )
    return chunks
