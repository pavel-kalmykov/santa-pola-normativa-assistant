import json
import random
from dataclasses import dataclass

from openai import OpenAI

from santa_pola_rag.config import settings
from santa_pola_rag.indexing.build_index import fetch_pages
from santa_pola_rag.indexing.chunking import Chunk, chunk_page

# Asking in a mix of languages exercises the multilingual embedding model the
# same way real users (many of them foreign residents) would. Valencian is
# included because it's one of the app's UI languages (see i18n.py) but had
# zero ground-truth coverage: language detection can only ever return "ca"
# for it (see language.py), same as for Catalan, so this doesn't collide with
# the other languages here.
LANGUAGES = ["Spanish", "English", "French", "German", "Valencian"]

QUESTION_PROMPT = """\
You are creating a retrieval evaluation set for a RAG system over Santa \
Pola's municipal ordinances. Read the excerpt below (in Spanish) and write \
ONE specific question, in {language}, that a resident could ask and that \
this excerpt alone answers. Do not mention "the excerpt" or "the document". \
Respond with only the question, no quotes, no preamble.

Excerpt (from "{title}"):
\"\"\"
{text}
\"\"\"
"""


@dataclass
class GroundTruthItem:
    question: str
    chunk_id: str
    document_url: str
    title: str


def _client() -> OpenAI:
    return OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)


def _sample_chunks(n_samples: int, seed: int = 42) -> list[Chunk]:
    pages = fetch_pages()
    all_chunks = []
    for page in pages:
        all_chunks.extend(
            chunk_page(
                document_url=page["document_url"],
                category_slug=page["category_slug"],
                title=page["title"],
                page_number=page["page_number"],
                page_count=page["page_count"],
                source=page["source"],
                text=page["text"] or "",
            )
        )
    # Only meaningful, non-trivial chunks make good evaluation questions.
    substantial = [c for c in all_chunks if len(c.text) > 200]
    random.Random(seed).shuffle(substantial)
    return substantial[:n_samples]


def generate_ground_truth(n_samples: int = 40, seed: int = 42) -> list[GroundTruthItem]:
    chunks = _sample_chunks(n_samples, seed=seed)
    client = _client()
    rng = random.Random(seed)

    items = []
    for chunk in chunks:
        language = rng.choice(LANGUAGES)
        prompt = QUESTION_PROMPT.format(
            language=language, title=chunk.title, text=chunk.text
        )
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        question = response.choices[0].message.content.strip()
        items.append(
            GroundTruthItem(
                question=question,
                chunk_id=chunk.chunk_id,
                document_url=chunk.document_url,
                title=chunk.title,
            )
        )
    return items


def save_ground_truth(items: list[GroundTruthItem], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([item.__dict__ for item in items], f, ensure_ascii=False, indent=2)


def load_ground_truth(path: str) -> list[GroundTruthItem]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [GroundTruthItem(**item) for item in raw]
