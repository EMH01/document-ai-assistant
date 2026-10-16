"""Structure-aware PDF extraction. Optional Docling, safe PyMuPDF fallback."""
from io import BytesIO

from .models import Chunk
from .retrieval import chunk_text


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

    options = PdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = True
    options.enable_remote_services = False
    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options)
    })
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
        chunks.extend(chunk_text(text, page=page, source=source, chunk_size=1100, overlap=120))
    if not chunks:
        raise ValueError("Docling did not extract readable content from this PDF.")
    return chunks
