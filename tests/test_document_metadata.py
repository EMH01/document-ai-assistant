from document_ai.models import Chunk


def test_source_metadata_keeps_coordinates():
    chunk = Chunk("La cancelación se contempla...", page=4, source="contrato.pdf", bbox=(10, 20, 200, 40))
    assert chunk.citation == "contrato.pdf · page 4"
    assert chunk.bbox == (10, 20, 200, 40)


def test_unknown_coordinates_are_explicitly_unavailable():
    chunk = Chunk("Texto recuperado por OCR", page=2, source="scan.pdf")
    assert chunk.bbox is None
