import psycopg2
from pgvector import Vector
from pgvector.psycopg2 import register_vector
from psycopg2.extras import RealDictCursor, execute_values

from santa_pola_rag.config import settings
from santa_pola_rag.indexing.chunking import Chunk
from santa_pola_rag.indexing.embeddings import EMBEDDING_DIM

# Chosen over a separate managed vector database after benchmarking, reusing
# the same Neon/Postgres instance the ingestion pipeline's staging tables
# already require. Own schema, kept apart from dlt's santa_pola_raw so a
# `reset_table` here can never touch staged page data.
SCHEMA = "santa_pola_vectors"
TABLE = "chunks"


def _connect() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(settings.postgres_dsn)
    conn.autocommit = False
    return conn


def ensure_table() -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE} (
                    chunk_id text PRIMARY KEY,
                    document_url text NOT NULL,
                    category_slug text NOT NULL,
                    title text NOT NULL,
                    page_number integer NOT NULL,
                    page_count integer NOT NULL,
                    source text NOT NULL,
                    text text NOT NULL,
                    embedding vector({EMBEDDING_DIM}) NOT NULL
                )
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {TABLE}_embedding_hnsw_idx
                ON {SCHEMA}.{TABLE}
                USING hnsw (embedding vector_cosine_ops)
                """
            )
        conn.commit()
    finally:
        conn.close()


def reset_table() -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {SCHEMA}.{TABLE}")
        conn.commit()
    finally:
        conn.close()
    ensure_table()


def upsert_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    ensure_table()
    conn = _connect()
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {SCHEMA}.{TABLE}
                    (chunk_id, document_url, category_slug, title, page_number,
                     page_count, source, text, embedding)
                VALUES %s
                ON CONFLICT (chunk_id) DO UPDATE SET
                    document_url = EXCLUDED.document_url,
                    category_slug = EXCLUDED.category_slug,
                    title = EXCLUDED.title,
                    page_number = EXCLUDED.page_number,
                    page_count = EXCLUDED.page_count,
                    source = EXCLUDED.source,
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_url,
                        chunk.category_slug,
                        chunk.title,
                        chunk.page_number,
                        chunk.page_count,
                        chunk.source,
                        chunk.text,
                        Vector(vector),
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
        conn.commit()
    finally:
        conn.close()


def search(query_vector: list[float], top_k: int = 10) -> list[dict]:
    conn = _connect()
    try:
        register_vector(conn)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Cosine distance (`<=>`, matching the hnsw vector_cosine_ops index
            # above) is 0 for identical vectors and grows from there; `1 -
            # distance` flips it to a higher-is-better score, which is all
            # hybrid_search's RRF fusion relies on (it ranks by position, never
            # compares raw scores across backends).
            cur.execute(
                f"""
                SELECT chunk_id, document_url, category_slug, title, page_number,
                       page_count, source, text,
                       1 - (embedding <=> %s) AS score
                FROM {SCHEMA}.{TABLE}
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (Vector(query_vector), Vector(query_vector), top_k),
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()
