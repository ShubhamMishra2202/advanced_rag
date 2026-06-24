"""Tiny embedding helper (384-d MiniLM dense + BM25 sparse via fastembed, no PyTorch)."""
import logging
from functools import lru_cache
from fastembed import TextEmbedding, SparseTextEmbedding

log = logging.getLogger(__name__)

VECTOR_SIZE = 384
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_SPARSE_MODEL_NAME = "Qdrant/bm25"

@lru_cache(maxsize=1)
def _model():
    log.info("Loading embedding model: %s", _MODEL_NAME)
    m = TextEmbedding(model_name=_MODEL_NAME)
    log.info("Embedding model loaded (vector size=%d)", VECTOR_SIZE)
    return m


@lru_cache(maxsize=1)
def _sparse_model():
    log.info("Loading sparse embedding model: %s", _SPARSE_MODEL_NAME)
    m = SparseTextEmbedding(model_name=_SPARSE_MODEL_NAME)
    log.info("Sparse embedding model loaded")
    return m


def encode(texts: list[str]) -> list[list[float]]:
    log.debug("Encoding %d text(s)", len(texts))
    vecs = [vec.tolist() for vec in _model().embed(texts)]
    log.debug("Encoding done — %d vector(s) produced", len(vecs))
    return vecs


def encode_sparse(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    log.debug("Sparse-encoding %d text(s)", len(texts))
    result = [
        (emb.indices.tolist(), emb.values.tolist())
        for emb in _sparse_model().embed(texts)
    ]
    log.debug("Sparse encoding done — %d vector(s) produced", len(result))
    return result
