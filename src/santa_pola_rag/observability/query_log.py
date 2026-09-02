import logging
from datetime import UTC, datetime

from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError

from santa_pola_rag.indexing.opensearch_index import get_client
from santa_pola_rag.rag.citations import has_footnote_list

logger = logging.getLogger(__name__)

INDEX_NAME = "santa_pola_queries"

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "session_id": {"type": "keyword"},
            # Sequential per-session counter (matches streamlit_app.py's own
            # display_messages length at the time), so a session's full
            # conversation can be reconstructed in order with a single query
            # (session_id term + sort by turn_index) instead of relying on
            # created_at timestamps, which only give ordering by coincidence.
            "turn_index": {"type": "integer"},
            "question": {"type": "text"},
            "answer": {"type": "text"},
            "question_language": {"type": "keyword"},
            "latency_ms": {"type": "integer"},
            "search_time_ms": {"type": "integer"},
            "llm_time_ms": {"type": "integer"},
            "has_citation": {"type": "boolean"},
            "citation_label": {"type": "keyword"},
            # A turn that failed still gets a record (previously it didn't:
            # streamlit_app.py used to st.stop() before ever reaching
            # record_query() on any of its three exception branches, so a
            # failed turn left no trace anywhere, including no trace of
            # which question triggered it).
            "has_error": {"type": "boolean"},
            "error_type": {"type": "keyword"},
            "created_at": {"type": "date"},
        }
    }
}


def ensure_index() -> None:
    # Query logging is observability, not core RAG: a managed OpenSearch
    # being unreachable (paused free tier) must not block the chat UI from
    # loading at all.
    try:
        client = get_client()
        if client.indices.exists(index=INDEX_NAME):
            return
        client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
    except OpenSearchConnectionError:
        logger.warning("OpenSearch unreachable, query logging disabled")


def record_query(
    session_id: str,
    turn_index: int,
    question: str,
    answer: str,
    question_language: str | None,
    latency_ms: int,
    search_time_ms: int = 0,
    error_type: str | None = None,
) -> None:
    cited = has_footnote_list(answer)
    try:
        client = get_client()
        response = client.index(
            index=INDEX_NAME,
            body={
                "session_id": session_id,
                "turn_index": turn_index,
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
                "citation_label": "Cited" if cited else "Not cited",
                "has_error": error_type is not None,
                "error_type": error_type,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        # The only way to later find exactly which document a given question
        # ended up as, e.g. to inspect or replay it: previously nothing was
        # logged on the success path at all, only on failure.
        logger.info(
            "Logged query turn_index=%d session_id=%s opensearch_id=%s error_type=%s",
            turn_index,
            session_id,
            response.get("_id"),
            error_type,
        )
    except OpenSearchConnectionError:
        logger.warning(
            "OpenSearch unreachable, dropped query log record for turn_index=%d "
            "session_id=%s",
            turn_index,
            session_id,
        )
