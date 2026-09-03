from santa_pola_rag.app.i18n import AVAILABLE_LANGUAGES, strings_for


def test_every_locale_exposes_the_same_keys():
    # Key drift between the TOML files is a runtime KeyError on whatever
    # widget reads the missing string (this actually shipped once, with a
    # sidebar button added before its translations).
    reference = strings_for("es")
    assert reference, "the reference locale must not be empty"
    for code in AVAILABLE_LANGUAGES:
        strings = strings_for(code)
        missing = set(reference) - set(strings)
        assert not missing, f"{code} is missing keys: {sorted(missing)}"


def test_no_locale_value_is_empty():
    for code in AVAILABLE_LANGUAGES:
        for key, value in strings_for(code).items():
            assert value.strip(), f"{code}.{key} is empty"
