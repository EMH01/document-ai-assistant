from document_ai.hierarchical import build_search_index, expand_ranked, search_preview, page_context_blocks
from document_ai.models import Chunk


def test_compact_child_expands_into_complete_original_text():
    original_text = "Contrato de servicio. " + "Condiciones generales de prueba. " * 45 + "Desistimiento en 14 días."
    original = Chunk(text=original_text, page=4, source="contrato.pdf", bbox=(20, 30, 300, 180))
    children, mapping = build_search_index([original])
    assert len(children) == 1
    assert len(children[0].text) < len(original_text)
    assert children[0].page == original.page
    assert children[0].bbox == original.bbox
    expanded = expand_ranked([(children[0], 0.9)], mapping)
    assert expanded[0][0] is original
    assert expanded[0][0].text == original_text


def test_short_excerpts_are_not_modified():
    assert search_preview("Texto íntegro.") == "Texto íntegro."


def test_page_context_reassembles_items_and_continuation():
    parts = [
        Chunk(text="CURSO 2018-2019 (Segundo - Año)", page=1, source="notas.pdf"),
        Chunk(text="Educación Física III", page=1, source="notas.pdf"),
        Chunk(text="Metodología de la Investigación", page=2, source="notas.pdf"),
        Chunk(text="CURSO 2019-2020 (Tercero - Año)", page=2, source="otro.pdf"),
    ]
    blocks = page_context_blocks(parts, "notas.pdf", {1, 2})
    assert len(blocks) == 2
    assert "Educación Física III" in blocks[0].text
    assert "Metodología de la Investigación" in blocks[1].text
    assert "Tercero" not in blocks[1].text
    assert blocks[0].text.count("\\n") == 1
