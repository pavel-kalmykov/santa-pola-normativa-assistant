from functools import lru_cache

from lingua import LanguageDetectorBuilder

# langdetect (a 2014 port of a 2010 algorithm) was tried first and rejected:
# it is genuinely non-deterministic on short text (verified: the same
# question returned Spanish and Italian across repeated calls with no code
# change), and even with its random seed pinned for determinism, it was
# consistently wrong on real short questions from this app. lingua is a
# modern, actively maintained detector built specifically to be accurate on
# short text, which is exactly this app's use case (single-sentence chat
# questions).


@lru_cache(maxsize=1)
def _detector():
    return LanguageDetectorBuilder.from_all_languages().build()


def detect_language(text: str) -> tuple[str, str] | None:
    """Returns (iso_639_1_code, display_name) for the detected language, e.g.
    ("es", "Spanish"), or None if no language could be determined."""
    language = _detector().detect_language_of(text)
    if language is None or language.iso_code_639_1 is None:
        return None
    return language.iso_code_639_1.name.lower(), language.name.title()
