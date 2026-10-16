import pymupdf

from document_ai.docling_extract import needs_ocr


def make_pdf(text: str) -> bytes:
    pdf = pymupdf.open()
    page = pdf.new_page()
    if text:
        page.insert_text((72, 72), text)
    data = pdf.tobytes()
    pdf.close()
    return data


def test_ocr_is_skipped_for_selectable_text():
    assert needs_ocr(make_pdf("Texto digital seleccionable en este documento de prueba.")) is False


def test_ocr_is_enabled_for_pages_without_text():
    assert needs_ocr(make_pdf("")) is True


def test_ocr_is_enabled_if_any_page_is_scanned():
    pdf = pymupdf.open()
    first = pdf.new_page()
    first.insert_text((72, 72), "Documento con texto digital en la primera pagina.")
    pdf.new_page()
    data = pdf.tobytes()
    pdf.close()
    assert needs_ocr(data) is True
