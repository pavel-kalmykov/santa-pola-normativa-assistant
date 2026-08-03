import re
from dataclasses import dataclass

from opentelemetry import trace

from santa_pola_rag.indexing import elasticsearch_index, qdrant_index
from santa_pola_rag.indexing.embeddings import embed_query
from santa_pola_rag.search.reranker import rerank_scores

# Standard Reciprocal Rank Fusion constant (Cormack et al., 2009): dampens the
# influence of rank position so a mediocre rank in one list doesn't dominate.
RRF_K = 60

# Fiscal ordinances get re-approved and re-published almost unchanged every
# year, so near-duplicate copies of the same tariff table compete for the
# same top-k slots across e.g. 2022-2026. A small recency bonus breaks that
# tie in favor of the current version without excluding older ones from the
# index or from being retrieved: a question about historical rates can still
# surface them, this only nudges the *default* ranking. Capped well below
# the smallest meaningful RRF score gap between two unrelated documents, so
# it can reorder near-duplicates but can't override a genuine relevance
# difference.
RECENCY_WEIGHT = 0.001
RECENCY_MIN_YEAR = 2015
RECENCY_MAX_YEAR = 2026
_YEAR_RE = re.compile(r"(19|20)\d{2}")

# How many of the top RRF-fused candidates get a second look from the
# cross-encoder before cutting to top_k. Larger than top_k on purpose: RRF's
# fusion can rank a genuinely relevant chunk just outside the final cut
# (e.g. rank 6-12) purely because one of the two channels missed it, and the
# cross-encoder gets a chance to recognize it anyway since it reads the
# actual text instead of only a rank position.
RERANK_POOL_SIZE = 20

_tracer = trace.get_tracer(__name__)


def _recency_bonus(document_url: str) -> float:
    """Small score bonus from the year embedded in the filename, e.g.
    "...-ESTABLECIMIENTOS-2026.pdf" -> 2026. Deliberately reads only the
    filename, not the full URL: santapola.es's upload path folder
    (".../uploads/2025/09/...") is the scrape/upload date, not the
    document's own date, and is actively misleading for bandos (a 2018
    notice can sit in a "2025/09" upload folder)."""
    basename = document_url.rsplit("/", 1)[-1]
    match = _YEAR_RE.search(basename)
    if match is None:
        return 0.0
    year = int(match.group(0))
    normalized = (year - RECENCY_MIN_YEAR) / (RECENCY_MAX_YEAR - RECENCY_MIN_YEAR)
    return RECENCY_WEIGHT * max(0.0, min(1.0, normalized))


@dataclass
class SearchResult:
    chunk_id: str
    document_url: str
    category_slug: str
    title: str
    page_number: int
    page_count: int
    source: str
    text: str
    score: float


def _to_rank_map(results: list[dict]) -> dict[str, tuple[int, dict]]:
    return {result["chunk_id"]: (rank, result) for rank, result in enumerate(results)}


def hybrid_search(
    query: str, top_k: int = 5, candidate_k: int = 50
) -> list[SearchResult]:
    """Fuse Qdrant vector search and Elasticsearch BM25 search results with RRF."""
    with _tracer.start_as_current_span("hybrid_search") as span:
        span.set_attribute("search.query", query)
        span.set_attribute("search.top_k", top_k)
        span.set_attribute("search.candidate_k", candidate_k)

        query_vector = embed_query(query)
        vector_results = qdrant_index.search(query_vector, top_k=candidate_k)
        text_results = elasticsearch_index.search(query, top_k=candidate_k)
        span.set_attribute("search.n_vector_results", len(vector_results))
        span.set_attribute("search.n_text_results", len(text_results))

        vector_ranks = _to_rank_map(vector_results)
        text_ranks = _to_rank_map(text_results)

        all_chunk_ids = set(vector_ranks) | set(text_ranks)
        fused = []
        for chunk_id in all_chunk_ids:
            score = 0.0
            payload = None
            if chunk_id in vector_ranks:
                rank, payload = vector_ranks[chunk_id]
                score += 1 / (RRF_K + rank + 1)
            if chunk_id in text_ranks:
                rank, payload = text_ranks[chunk_id]
                score += 1 / (RRF_K + rank + 1)
            score += _recency_bonus(payload["document_url"])
            fused.append((score, payload))

        fused.sort(key=lambda pair: pair[0], reverse=True)
        span.set_attribute("search.n_fused_results", len(fused))

        pool = fused[:RERANK_POOL_SIZE]
        pool_payloads = [payload for _, payload in pool]
        cross_encoder_scores = rerank_scores(
            query, [payload["text"] for payload in pool_payloads]
        )
        reranked = sorted(
            zip(cross_encoder_scores, pool_payloads),
            key=lambda pair: pair[0],
            reverse=True,
        )
        span.set_attribute("search.reranked", True)

        return [
            SearchResult(
                chunk_id=payload["chunk_id"],
                document_url=payload["document_url"],
                category_slug=payload["category_slug"],
                title=payload["title"],
                page_number=payload["page_number"],
                page_count=payload["page_count"],
                source=payload["source"],
                text=payload["text"],
                score=score,
            )
            for score, payload in reranked[:top_k]
        ]
