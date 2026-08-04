import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from santa_pola_rag.config import settings
from santa_pola_rag.indexing.chunking import Chunk
from santa_pola_rag.indexing.embeddings import EMBEDDING_DIM

COLLECTION_NAME = "santa_pola_chunks"

# Qdrant point IDs must be an unsigned int or a UUID; our chunk_id is a sha1
# hex digest (40 chars), too long for a UUID, so we derive one deterministically
# and keep the original chunk_id in the payload to join back with Elasticsearch.
_UUID_NAMESPACE = uuid.NAMESPACE_OID


def chunk_id_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_UUID_NAMESPACE, chunk_id))


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def ensure_collection(client: QdrantClient | None = None) -> None:
    client = client or get_qdrant_client()
    if client.collection_exists(COLLECTION_NAME):
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


def reset_collection(client: QdrantClient | None = None) -> None:
    client = client or get_qdrant_client()
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    ensure_collection(client)


def upsert_chunks(
    chunks: list[Chunk], vectors: list[list[float]], client: QdrantClient | None = None
) -> None:
    client = client or get_qdrant_client()
    ensure_collection(client)

    points = [
        PointStruct(
            id=chunk_id_to_point_id(chunk.chunk_id),
            vector=vector,
            payload={
                "chunk_id": chunk.chunk_id,
                "document_url": chunk.document_url,
                "category_slug": chunk.category_slug,
                "title": chunk.title,
                "page_number": chunk.page_number,
                "page_count": chunk.page_count,
                "source": chunk.source,
                "text": chunk.text,
            },
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def search(
    query_vector: list[float], top_k: int = 10, client: QdrantClient | None = None
) -> list[dict]:
    client = client or get_qdrant_client()
    results = client.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=top_k
    )
    return [{"score": point.score, **point.payload} for point in results.points]
