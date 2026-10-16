from document_ai.hierarchical import build_search_index, expand_ranked, search_preview
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
