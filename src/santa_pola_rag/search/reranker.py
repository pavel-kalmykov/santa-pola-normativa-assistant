import os
from functools import lru_cache

# See indexing/embeddings.py for why this must be set before the model loads.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import CrossEncoder  # noqa: E402

# Multilingual (trained on mMARCO): the corpus is Spanish, questions can be
# in any language, so an English-only cross-encoder would defeat the point.
RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL_NAME, device="cpu")


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """Score each (query, text) pair jointly with a cross-encoder. Unlike
    RRF, which only ever sees rank positions from two separately-computed
    result lists, a cross-encoder reads the actual query and passage
    together, so it can recognize relevance even for a chunk neither the
    embedding model nor BM25 ranked highly on its own. Scores are raw
    logits (not probabilities), meaningful only for ordering, not as an
    absolute confidence value."""
    if not texts:
        return []
    pairs = [(query, text) for text in texts]
    return get_reranker().predict(pairs).tolist()
