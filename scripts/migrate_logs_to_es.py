import logging

import psycopg2
import psycopg2.extras
from elasticsearch.helpers import bulk

from santa_pola_rag.config import settings
from santa_pola_rag.indexing.elasticsearch_index import get_es_client
from santa_pola_rag.observability import feedback, query_log

logger = logging.getLogger(__name__)


def _fetch_postgres_rows(table: str) -> list[dict]:
    conn = psycopg2.connect(settings.postgres_dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            try:
                cur.execute(f"select * from {table}")
            except psycopg2.errors.UndefinedTable:
                return []
            return list(cur.fetchall())
    finally:
        conn.close()


def migrate_queries() -> int:
    rows = _fetch_postgres_rows("app_queries")
    if not rows:
        return 0

    query_log.ensure_index()
    client = get_es_client()
    actions = [
        {
            "_index": query_log.INDEX_NAME,
            "_source": {
                "session_id": row["session_id"],
                "question": row["question"],
                "answer": row["answer"],
                "question_language": row["question_language"],
                "latency_ms": row["latency_ms"],
                "has_citation": row["has_citation"],
                "created_at": row["created_at"].isoformat(),
            },
        }
        for row in rows
    ]
    bulk(client, actions)
    return len(actions)


def migrate_feedback() -> int:
    rows = _fetch_postgres_rows("app_feedback")
    if not rows:
        return 0

    feedback.ensure_index()
    client = get_es_client()
    actions = [
        {
            "_index": feedback.INDEX_NAME,
            "_source": {
                "session_id": row["session_id"],
                "question": row["question"],
                "answer": row["answer"],
                "rating": row["rating"],
                "question_language": row["question_language"],
                "created_at": row["created_at"].isoformat(),
            },
        }
        for row in rows
    ]
    bulk(client, actions)
    return len(actions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n_queries = migrate_queries()
    n_feedback = migrate_feedback()
    print(
        f"Migrated {n_queries} app_queries rows and {n_feedback} app_feedback rows to Elasticsearch"
    )
