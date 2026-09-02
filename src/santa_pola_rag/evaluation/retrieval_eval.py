import logging
from collections.abc import Callable

from santa_pola_rag.evaluation.ground_truth import GroundTruthItem
from santa_pola_rag.indexing import opensearch_index, pgvector_index
from santa_pola_rag.indexing.embeddings import embed_query
from santa_pola_rag.search.hybrid import hybrid_search

logger = logging.getLogger(__name__)

TOP_K = 5

RetrievalFn = Callable[[str, int], list[str]]


def vector_only(query: str, top_k: int) -> list[str]:
    results = pgvector_index.search(embed_query(query), top_k=top_k)
    return [r["chunk_id"] for r in results]


def text_only(query: str, top_k: int) -> list[str]:
    results = opensearch_index.search(query, top_k=top_k)
    return [r["chunk_id"] for r in results]


def hybrid(query: str, top_k: int) -> list[str]:
    results = hybrid_search(query, top_k=top_k)
    return [r.chunk_id for r in results]


def evaluate(
    ground_truth: list[GroundTruthItem], retrieval_fn: RetrievalFn, top_k: int = TOP_K
) -> dict:
    hits = 0
    reciprocal_ranks = []

    for item in ground_truth:
        ranked_chunk_ids = retrieval_fn(item.question, top_k)
        if item.chunk_id in ranked_chunk_ids:
            hits += 1
            rank = ranked_chunk_ids.index(item.chunk_id) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

    n = len(ground_truth)
    return {
        "n": n,
        "hit_rate": hits / n if n else 0.0,
        "mrr": sum(reciprocal_ranks) / n if n else 0.0,
    }


def evaluate_all(ground_truth: list[GroundTruthItem], top_k: int = TOP_K) -> dict:
    results = {}
    for name, fn in [
        ("vector_only", vector_only),
        ("text_only", text_only),
        ("hybrid", hybrid),
    ]:
        results[name] = evaluate(ground_truth, fn, top_k=top_k)
        logger.info("%s: %s", name, results[name])
    return results
