"""Structure-aware PDF extraction. Optional Docling, safe PyMuPDF fallback."""
import os
from functools import lru_cache
from io import BytesIO

import pymupdf

from .models import Chunk
from .retrieval import chunk_text


def needs_ocr(pdf_bytes: bytes, min_chars: int = 35) -> bool:
    """Enable OCR only if at least one page has little selectable text."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        return any(len(page.get_text("text").strip()) < min_chars for page in pdf)


@lru_cache(maxsize=2)
def _converter(use_ocr: bool, ocr_backend: str = "onnxruntime"):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = use_ocr
    if use_ocr:
        options.ocr_options = RapidOcrOptions(backend=ocr_backend, lang=["iso:es"])
    options.do_table_structure = True
    # Keep CPU as the portable default for Streamlit Community Cloud.
    options.enable_remote_services = False
    return DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options)
    })


def extract_document(pdf_bytes: bytes, source: str, require_docling: bool = False) -> list[Chunk]:
    try:
        from docling.datamodel.base_models import InputFormat, DocumentStream
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        if require_docling:
            raise RuntimeError("Docling OCR unavailable: install document-ai-assistant[docling].") from exc
        from .pdf import extract_chunks
        return extract_chunks(pdf_bytes, source=source)

    backend = os.getenv("DOC_OCR_BACKEND", "onnxruntime").strip().lower()
    if backend not in {"onnxruntime", "torch"}:
        raise ValueError("DOC_OCR_BACKEND must be onnxruntime or torch")
    converter = _converter(needs_ocr(pdf_bytes), backend)
    result = converter.convert(DocumentStream(name=source, stream=BytesIO(pdf_bytes)))
    chunks: list[Chunk] = []
    document = result.document
    for item, _ in document.iterate_items():
        text = getattr(item, "text", None)
        if not text and hasattr(item, "export_to_markdown"):
            text = item.export_to_markdown(doc=document)
        if not text or not text.strip():
            continue
        provenance = getattr(item, "prov", None) or []
        page = provenance[0].page_no if provenance else 1
        extracted = chunk_text(text, page=page, source=source, chunk_size=1100, overlap=120)
        if first := (provenance[0] if provenance else None):
            box = getattr(first, "bbox", None)
            page_info = document.pages.get(page)
            size = getattr(page_info, "size", None)
            if box is not None and size is not None:
                try:
                    left, bottom, right, top = float(box.l), float(box.b), float(box.r), float(box.t)
                    if "bottom" in str(box.coord_origin).lower():
                        coords = (left, float(size.height) - top, right, float(size.height) - bottom)
                    elif "top" in str(box.coord_origin).lower():
                        coords = (left, top, right, bottom)
                    else:
                        coords = None
                    if coords and coords[0] < coords[2] and coords[1] < coords[3]:
                        for chunk in extracted:
                            chunk.bbox = coords
                except (TypeError, ValueError, AttributeError):
                    pass
        chunks.extend(extracted)
    if not chunks:
        raise ValueError("Docling did not extract readable content from this PDF.")
    return chunks
