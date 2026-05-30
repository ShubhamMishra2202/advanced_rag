"""Tiny embedding helper (384-d MiniLM via fastembed, no PyTorch)."""
import logging
from functools import lru_cache
from fastembed import TextEmbedding

log = logging.getLogger(__name__)

VECTOR_SIZE = 768
_MODEL_NAME = "BAAI/bge-base-en-v1.5"


@lru_cache(maxsize=1)
def _model():
    log.info("Loading embedding model: %s", _MODEL_NAME)
    m = TextEmbedding(model_name=_MODEL_NAME)
    log.info("Embedding model loaded (vector size=%d)", VECTOR_SIZE)
    return m


def encode(texts: list[str]) -> list[list[float]]:
    log.debug("Encoding %d text(s)", len(texts))
    vecs = [vec.tolist() for vec in _model().embed(texts)]
    log.debug("Encoding done — %d vector(s) produced", len(vecs))
    return vecs
