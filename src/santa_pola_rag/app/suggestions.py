import json
from functools import lru_cache
from pathlib import Path

from santa_pola_rag.language import detect_language

GROUND_TRUTH_PATH = Path(__file__).resolve().parents[3] / "eval" / "ground_truth.json"
SUGGESTIONS_PER_LANGUAGE = 4


@lru_cache(maxsize=1)
def _questions_by_language() -> dict[str, list[str]]:
    """Ground truth questions have no stored language (see
    evaluation/ground_truth.py): each one was written in a language chosen
    at generation time but never persisted, so it has to be re-detected
    here from the question text itself."""
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        items = json.load(f)
    by_language: dict[str, list[str]] = {}
    for item in items:
        detected = detect_language(item["question"])
        if detected is None:
            continue
        by_language.setdefault(detected[0], []).append(item["question"])
    return by_language


def suggested_questions(language: str, n: int = SUGGESTIONS_PER_LANGUAGE) -> list[str]:
    return _questions_by_language().get(language, [])[:n]
