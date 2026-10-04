from functools import lru_cache
import numpy as np


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


def embed(texts):
    """Returns L2-normalised vectors, so dot product = cosine similarity."""
    return np.asarray(_model().encode(list(texts), normalize_embeddings=True, batch_size=64), dtype="float32")
