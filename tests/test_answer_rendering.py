from document_ai.providers import public_answer


def test_internal_thinking_is_hidden():
    assert public_answer("<think>private reasoning</think>Respuesta final [1].") == "Respuesta final [1]."


def test_incomplete_thinking_does_not_leak():
    assert "<think>" not in public_answer("<think>unfinished details")


def test_normal_answer_preserved():
    assert public_answer("Respuesta final [1].") == "Respuesta final [1]."


def test_citation_only_is_not_accepted_as_answer():
    assert "respuesta fiable" in public_answer("[1]")


def test_multiple_citation_only_is_not_accepted():
    assert "respuesta fiable" in public_answer("[1][2][3]")
