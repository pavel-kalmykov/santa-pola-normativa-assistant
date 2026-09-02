import logging
from datetime import UTC, datetime

import psycopg2
from psycopg2.extras import RealDictCursor

from santa_pola_rag.config import settings

logger = logging.getLogger(__name__)

SCHEMA = "santa_pola_chats"
TABLE = "chats"

# The same unreachable-server pair hybrid_search retries on: a network-level
# failure to reach Postgres. Query errors (programming mistakes) are NOT
# swallowed here, unlike the old OpenSearch version where the client raised
# ConnectionError for both classes.
_PG_EXCEPTIONS = (psycopg2.OperationalError, psycopg2.InterfaceError)


def _connect() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(settings.postgres_dsn, connect_timeout=5)
    conn.autocommit = True
    return conn


def ensure_table() -> None:
    # Chat persistence is user state, not core RAG: a unreachable Postgres
    # must not block the chat UI from loading at all.
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE} (
                    session_id text PRIMARY KEY,
                    browser_id text NOT NULL,
                    title text NOT NULL,
                    created_at timestamptz NOT NULL,
                    updated_at timestamptz NOT NULL,
                    display_messages_json text NOT NULL,
                    pydantic_history_json text NOT NULL
                )
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS chats_browser_idx "
                f"ON {SCHEMA}.{TABLE} (browser_id, updated_at DESC)"
            )
    except _PG_EXCEPTIONS:
        logger.warning("Postgres unreachable, chat history disabled")


def save_chat(
    browser_id: str,
    session_id: str,
    title: str,
    created_at: str | datetime,
    display_messages_json: str,
    pydantic_history_json: str,
) -> None:
    # Upsert keyed by the chat's own session_id: every turn overwrites the
    # same row with the latest full state, rather than accumulating one row
    # per turn. In Postgres the write is immediately visible to the sidebar's
    # next list_chats(), unlike OpenSearch where it needed refresh=True.
    try:
        created = (
            datetime.fromisoformat(created_at)
            if isinstance(created_at, str)
            else created_at
        )
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {SCHEMA}.{TABLE}
                    (session_id, browser_id, title, created_at, updated_at,
                     display_messages_json, pydantic_history_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    browser_id = EXCLUDED.browser_id,
                    title = EXCLUDED.title,
                    updated_at = EXCLUDED.updated_at,
                    display_messages_json = EXCLUDED.display_messages_json,
                    pydantic_history_json = EXCLUDED.pydantic_history_json
                """,
                (
                    session_id,
                    browser_id,
                    title,
                    created,
                    datetime.now(UTC),
                    display_messages_json,
                    pydantic_history_json,
                ),
            )
    except _PG_EXCEPTIONS:
        logger.warning("Postgres unreachable, chat %s not saved", session_id)


def list_chats(browser_id: str, limit: int = 50) -> list[dict]:
    try:
        with _connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT session_id, title, updated_at FROM {SCHEMA}.{TABLE}
                WHERE browser_id = %s
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (browser_id, limit),
            )
            return [dict(row) for row in cur.fetchall()]
    except _PG_EXCEPTIONS:
        logger.warning("Postgres unreachable, cannot list past chats")
        return []


def load_chat(session_id: str) -> dict | None:
    try:
        with _connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT session_id, title, created_at,
                       display_messages_json, pydantic_history_json
                FROM {SCHEMA}.{TABLE}
                WHERE session_id = %s
                """,
                (session_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    except _PG_EXCEPTIONS:
        logger.warning("Postgres unreachable or chat %s missing", session_id)
        return None


def delete_chat(session_id: str) -> None:
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {SCHEMA}.{TABLE} WHERE session_id = %s",
                (session_id,),
            )
    except _PG_EXCEPTIONS:
        logger.warning("Could not delete chat %s", session_id)


def rename_chat(session_id: str, title: str) -> None:
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"UPDATE {SCHEMA}.{TABLE} SET title = %s WHERE session_id = %s",
                (title, session_id),
            )
    except _PG_EXCEPTIONS:
        logger.warning("Could not rename chat %s", session_id)
