import tomllib
from functools import lru_cache
from pathlib import Path

LOCALES_DIR = Path(__file__).parent / "locales"

# Native names as values so a language picker can list them without first
# knowing which language is currently selected (self-referential labels are
# what caused the interface's own language selector to need two clicks: see
# streamlit_app.py's comment on the `key=`-bound selectbox).
AVAILABLE_LANGUAGES = {
    "es": "Español",
    "en": "English",
    "fr": "Français",
    "de": "Deutsch",
    "ca": "Valencià",
}


@lru_cache(maxsize=len(AVAILABLE_LANGUAGES))
def strings_for(language_code: str) -> dict:
    """UI chrome strings for one interface language, loaded from
    locales/<code>.toml. Streamlit has no built-in i18n API (there's a long-
    standing open feature request for one: streamlit/streamlit#1353); the
    common community workaround, also used here, is a language choice kept
    in session_state plus strings kept in external per-language files or
    dicts instead of hardcoded in the widgets themselves."""
    path = LOCALES_DIR / f"{language_code}.toml"
    with path.open("rb") as f:
        return tomllib.load(f)
