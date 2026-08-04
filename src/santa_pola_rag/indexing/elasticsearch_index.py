from functools import lru_cache

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from santa_pola_rag.config import settings
from santa_pola_rag.indexing.chunking import Chunk

INDEX_NAME = "santa_pola_chunks"

# Source documents are in Spanish; the built-in "spanish" analyzer stems
# Spanish text (e.g. "ordenanzas" / "ordenanza") for better BM25 recall.
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
def get_es_client() -> Elasticsearch:
    return Elasticsearch(settings.elasticsearch_url, api_key=settings.elasticsearch_api_key)


def ensure_index(client: Elasticsearch | None = None) -> None:
    client = client or get_es_client()
    if client.indices.exists(index=INDEX_NAME):
        return
    client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)


def reset_index(client: Elasticsearch | None = None) -> None:
    client = client or get_es_client()
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
    ensure_index(client)


def index_chunks(chunks: list[Chunk], client: Elasticsearch | None = None) -> None:
    client = client or get_es_client()
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


def search(
    query: str, top_k: int = 10, client: Elasticsearch | None = None
) -> list[dict]:
    client = client or get_es_client()
    response = client.search(
        index=INDEX_NAME,
        query={"match": {"text": query}},
        size=top_k,
    )
    return [
        {"score": hit["_score"], **hit["_source"]} for hit in response["hits"]["hits"]
    ]
