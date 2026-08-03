from datetime import UTC, datetime

from santa_pola_rag.indexing.elasticsearch_index import get_es_client
from santa_pola_rag.rag.citations import has_footnote_list

INDEX_NAME = "santa_pola_queries"

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "session_id": {"type": "keyword"},
            "question": {"type": "text"},
            "answer": {"type": "text"},
            "question_language": {"type": "keyword"},
            "latency_ms": {"type": "integer"},
            "search_time_ms": {"type": "integer"},
            "llm_time_ms": {"type": "integer"},
            "has_citation": {"type": "boolean"},
            "citation_label": {"type": "keyword"},
            "created_at": {"type": "date"},
        }
    }
}


def ensure_index() -> None:
    client = get_es_client()
    if client.indices.exists(index=INDEX_NAME):
        return
    client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)


def record_query(
    session_id: str,
    question: str,
    answer: str,
    question_language: str | None,
    latency_ms: int,
    search_time_ms: int = 0,
) -> None:
    cited = has_footnote_list(answer)
    client = get_es_client()
    client.index(
        index=INDEX_NAME,
        document={
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "question_language": question_language,
            "latency_ms": latency_ms,
            "search_time_ms": search_time_ms,
            # Whatever isn't spent in real search_ordinances calls is spent
            # generating text (narration + tool-call arguments + the final
            # answer); never negative even if the two clocks drift slightly.
            "llm_time_ms": max(0, latency_ms - search_time_ms),
            "has_citation": cited,
            # A precomputed human-readable keyword, not just the boolean:
            # Grafana's piechart panel renders a plain string term cleanly
            # (see "Question language distribution"), but mangles the
            # legend labels for a numeric field even with value mappings
            # configured, so the dashboard reads this field instead.
            "citation_label": "Con cita" if cited else "Sin cita",
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
