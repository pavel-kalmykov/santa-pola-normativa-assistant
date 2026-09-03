from santa_pola_rag.rag.citations import has_footnote_list, render_footnotes


def answer(*sources: str) -> str:
    lines = "\n\n".join(
        f"[{i}] {title}, p. {page}, {url}"
        for i, (title, page, url) in enumerate(sources, start=1)
    )
    return lines


NOISE = "https://santapola.es/ruidos.pdf"


def test_has_footnote_list_requires_a_parseable_source_line():
    assert has_footnote_list(f"Texto [1].\n\n[1] Título, p. 4, {NOISE}")
    assert not has_footnote_list("Texto sin fuentes ni citas [1].")


def test_answer_without_footnote_list_is_returned_unchanged():
    text = "El horario es de 8 a 20 h [1], según la ordenanza."
    assert render_footnotes(text, message_id=1, language_code="es") == text


def test_single_inline_marker_becomes_a_linked_footnote():
    rendered = render_footnotes(
        f"Puedes hacerlo de 8 a 20 h [1].\n\n{answer(('Ordenanza de Ruidos', '4', NOISE))}",
        message_id=1,
        language_code="es",
    )
    assert '<a href="#sp-fn-1-1"' in rendered
    assert 'id="sp-ref-1-1-1"' in rendered
    assert "Ordenanza de Ruidos, p. 4" in rendered


def test_comma_separated_marker_renders_every_number():
    # The model was observed writing "[1, 2]" despite the prompt asking for
    # "[1][2]"; the old single-number regex left it as dead, unlinked text.
    rendered = render_footnotes(
        "Vigilan las obras [1, 2].\n\n"
        + answer(
            ("Ordenanza de Ruidos", "4", NOISE),
            ("Ordenanza de Terrazas", "9", "https://santapola.es/terrazas.pdf"),
        ),
        message_id=3,
        language_code="es",
    )
    assert rendered.count("<sup>") == 2
    assert 'href="#sp-fn-3-1"' in rendered
    assert 'href="#sp-fn-3-2"' in rendered
    assert "[1, 2]" not in rendered


def test_duplicate_source_cited_twice_collapses_into_one_footnote():
    same_source = ("Ordenanza de Ruidos", "4", NOISE)
    rendered = render_footnotes(
        "Cita A [1]. Cita B [2].\n\n" + answer(same_source, same_source),
        message_id=5,
        language_code="es",
    )
    # Both inline markers resolve to the same canonical footnote anchor,
    # and the source list holds a single entry.
    assert 'href="#sp-fn-5-1"' in rendered
    assert "sp-fn-5-2" not in rendered
    assert rendered.count("<li") == 1


def test_marker_without_a_matching_source_stays_literal():
    rendered = render_footnotes(
        f"Texto [7].\n\n{answer(('Ordenanza de Ruidos', '4', NOISE))}",
        message_id=6,
        language_code="es",
    )
    assert "<sup>" not in rendered
    assert "[7]" in rendered


def test_stray_html_is_stripped_before_parsing():
    # The model once wrapped the whole answer in <p>...</p>, which broke the
    # footnote line anchors and glued the closing tag onto the last URL.
    rendered = render_footnotes(
        f"<p>Cita [1].</p>\n\n{answer(('Ordenanza de Ruidos', '4', NOISE))}",
        message_id=8,
        language_code="es",
    )
    assert "<p>" not in rendered
    assert 'href="#sp-fn-8-1"' in rendered
