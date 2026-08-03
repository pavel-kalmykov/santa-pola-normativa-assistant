from datetime import UTC, datetime

from santa_pola_rag.indexing.elasticsearch_index import get_es_client

INDEX_NAME = "santa_pola_feedback"

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "session_id": {"type": "keyword"},
            "question": {"type": "text"},
            "answer": {"type": "text"},
            "rating": {"type": "byte"},
            "rating_label": {"type": "keyword"},
            "question_language": {"type": "keyword"},
            "created_at": {"type": "date"},
        }
    }
}


def ensure_index() -> None:
    client = get_es_client()
    if client.indices.exists(index=INDEX_NAME):
        return
    client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)


def record_feedback(
    session_id: str,
    question: str,
    answer: str,
    rating: int,
    question_language: str | None = None,
) -> None:
    client = get_es_client()
    client.index(
        index=INDEX_NAME,
        document={
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "rating": rating,
            # See query_log.py's citation_label for why: Grafana's piechart
            # renders a plain keyword term cleanly but not a numeric field
            # with value mappings.
            "rating_label": "👍 Positive" if rating > 0 else "👎 Negative",
            "question_language": question_language,
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
