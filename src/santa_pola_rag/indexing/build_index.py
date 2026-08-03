import logging

import psycopg2
import psycopg2.extras

from santa_pola_rag.config import settings
from santa_pola_rag.indexing.chunking import Chunk, chunk_page
from santa_pola_rag.indexing.elasticsearch_index import index_chunks as es_index_chunks
from santa_pola_rag.indexing.embeddings import embed_texts
from santa_pola_rag.indexing.qdrant_index import upsert_chunks as qdrant_upsert_chunks

logger = logging.getLogger(__name__)

EMBEDDING_BATCH_SIZE = 64

FETCH_PAGES_SQL = """
    select
        p.document_url,
        p.category_slug,
        p.page_number,
        p.page_count,
        p.source,
        p.text,
        d.title
    from santa_pola_raw.pages p
    join santa_pola_raw.documents d on d.url = p.document_url
    order by p.document_url, p.page_number
"""


def fetch_pages() -> list[dict]:
    conn = psycopg2.connect(settings.postgres_dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(FETCH_PAGES_SQL)
            return list(cur.fetchall())
    finally:
        conn.close()


def build_chunks(pages: list[dict]) -> list[Chunk]:
    chunks = []
    for page in pages:
        chunks.extend(
            chunk_page(
                document_url=page["document_url"],
                category_slug=page["category_slug"],
                title=page["title"],
                page_number=page["page_number"],
                page_count=page["page_count"],
                source=page["source"],
                text=page["text"] or "",
            )
        )
    logger.info("Built %d chunk(s) from %d page(s)", len(chunks), len(pages))
    return chunks


def index_all_chunks(chunks: list[Chunk]) -> None:
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
        vectors = embed_texts([chunk.text for chunk in batch])
        qdrant_upsert_chunks(batch, vectors)
        es_index_chunks(batch)
        logger.info(
            "Indexed chunks %d-%d of %d", start, start + len(batch), len(chunks)
        )


def run() -> int:
    pages = fetch_pages()
    chunks = build_chunks(pages)
    index_all_chunks(chunks)
    return len(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n_chunks = run()
    print(f"Indexed {n_chunks} chunks into Qdrant and Elasticsearch")
