import logging
from datetime import UTC, datetime

from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError

from santa_pola_rag.indexing.opensearch_index import get_client

logger = logging.getLogger(__name__)

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
    # Feedback is observability, not core RAG: a managed OpenSearch being
    # unreachable (paused free tier) must not block the chat UI from loading
    # at all.
    try:
        client = get_client()
        if client.indices.exists(index=INDEX_NAME):
            return
        client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
    except OpenSearchConnectionError:
        logger.warning("OpenSearch unreachable, feedback logging disabled")


def record_feedback(
    session_id: str,
    question: str,
    answer: str,
    rating: int,
    question_language: str | None = None,
) -> None:
    try:
        client = get_client()
        client.index(
            index=INDEX_NAME,
            body={
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
    except OpenSearchConnectionError:
        logger.warning("OpenSearch unreachable, dropped one feedback record")
