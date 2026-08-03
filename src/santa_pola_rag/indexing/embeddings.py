import os
from functools import lru_cache

# HF tokenizers' fork-based parallelism deadlocks when combined with the MPS
# backend under concurrent/agent workloads (observed as a hung process with a
# leaked loky semaphore); must be set before the model/tokenizer is loaded.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import SentenceTransformer  # noqa: E402

# Multilingual sentence embedding model: Spanish source documents, questions in
# any language (verified cross-lingual similarity ~0.85 ES<->EN/FR at design time).
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    # Force CPU: the MPS backend is what triggers the deadlock above, and a
    # 384-dim MiniLM model is fast enough on CPU for this workload anyway.
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
