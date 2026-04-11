"""Tiny embedding helper (384-d MiniLM via fastembed, no PyTorch)."""
from functools import lru_cache

VECTOR_SIZE = 384
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=_MODEL_NAME)


def encode(texts: list[str]) -> list[list[float]]:
    m = _model()
    return [vec.tolist() for vec in m.embed(texts)]
