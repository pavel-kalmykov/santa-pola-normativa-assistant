from functools import lru_cache

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

from santa_pola_rag.config import settings
from santa_pola_rag.indexing.chunking import Chunk

INDEX_NAME = "santa_pola_chunks"

# BM25 text search only: benchmarked against its own k-NN plugin for the
# vector side too, but pgvector won that comparison (see retrieval_eval.py's
# vector_only/text_only results), so this index carries no embedding field.
# Spanish analyzer mirrors what the retired Elasticsearch index used, so the
# BM25 side is a genuine like-for-like replacement, not an accidentally
# weaker one (confirmed identical hit_rate/MRR in the benchmark).
INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "document_url": {"type": "keyword"},
            "category_slug": {"type": "keyword"},
            "title": {"type": "text"},
            "page_number": {"type": "integer"},
            "page_count": {"type": "integer"},
            "source": {"type": "keyword"},
            "text": {"type": "text", "analyzer": "spanish"},
        }
    }
}


@lru_cache(maxsize=1)
def get_client() -> OpenSearch:
    # Token-auth offerings only need the api_key; basic-auth ones like Aiven
    # also need the real username, which OPENSEARCH_USER carries. The "x"
    # placeholder keeps token-only setups working unchanged.
    return OpenSearch(
        settings.opensearch_url,
        http_auth=(settings.opensearch_user or "x", settings.opensearch_api_key)
        if settings.opensearch_api_key
        else None,
        use_ssl=settings.opensearch_url.startswith("https"),
        verify_certs=True,
    )


def ensure_index(client: OpenSearch | None = None) -> None:
    client = client or get_client()
    if client.indices.exists(index=INDEX_NAME):
        return
    client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)


def reset_index(client: OpenSearch | None = None) -> None:
    client = client or get_client()
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
    ensure_index(client)


def index_chunks(chunks: list[Chunk], client: OpenSearch | None = None) -> None:
    client = client or get_client()
    ensure_index(client)

    actions = [
        {
            "_index": INDEX_NAME,
            "_id": chunk.chunk_id,
            "_source": {
                "chunk_id": chunk.chunk_id,
                "document_url": chunk.document_url,
                "category_slug": chunk.category_slug,
                "title": chunk.title,
                "page_number": chunk.page_number,
                "page_count": chunk.page_count,
                "source": chunk.source,
                "text": chunk.text,
            },
        }
        for chunk in chunks
    ]
    bulk(client, actions)


def search(query: str, top_k: int = 10, client: OpenSearch | None = None) -> list[dict]:
    client = client or get_client()
    response = client.search(
        index=INDEX_NAME,
        body={"query": {"match": {"text": query}}, "size": top_k},
    )
    return [
        {"score": hit["_score"], **hit["_source"]} for hit in response["hits"]["hits"]
    ]
