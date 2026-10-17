import pymupdf

from document_ai.visual import pdf_visual_regions


def test_no_visual_regions_on_text_only_page():
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((40, 40), "This page contains selectable text only.")
    raw = pdf.tobytes()
    pdf.close()
    assert pdf_visual_regions(raw) == []


def test_scanned_page_background_is_not_a_figure():
    import io
    import pytest
    Image = pytest.importorskip("PIL.Image")

    image = Image.new("RGB", (750, 1000), "white")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    pdf = pymupdf.open()
    page = pdf.new_page(width=750, height=1000)
    page.insert_image(page.rect, stream=stream.getvalue())
    raw = pdf.tobytes()
    pdf.close()
    assert pdf_visual_regions(raw) == []
