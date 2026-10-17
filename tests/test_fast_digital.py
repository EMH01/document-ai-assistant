"""Regression checks for the conservative digital-PDF fast path."""
import pymupdf

from document_ai.docling_extract import fast_digital_chunks


def _pdf(text: str) -> bytes:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_textbox(pymupdf.Rect(60, 70, 540, 750), text, fontsize=11)
    data = pdf.tobytes()
    pdf.close()
    return data


def test_digital_text_uses_fast_path_and_keeps_provenance():
    content = "This agreement describes a simple non-tabular service. " * 5
    chunks = fast_digital_chunks(_pdf(content), "contract.pdf")
    assert chunks
    assert all(item.source == "contract.pdf" for item in chunks)
    assert all(item.page == 1 for item in chunks)
    assert all(item.bbox is not None for item in chunks)
    assert "agreement" in " ".join(item.text for item in chunks)


def test_sparse_text_defers_to_docling():
    assert fast_digital_chunks(_pdf("Scanned form"), "form.pdf") is None


def test_image_only_pdf_defers_to_docling():
    pdf = pymupdf.open()
    pdf.new_page()
    data = pdf.tobytes()
    pdf.close()
    assert fast_digital_chunks(data, "scan.pdf") is None
