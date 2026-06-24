"""Embed query and search Qdrant (hybrid dense + sparse BM25 with RRF fusion)."""
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from embed import encode, encode_sparse
from ingestion import COLLECTION

log = logging.getLogger(__name__)
CLIENT = QdrantClient(url="http://localhost:6333")


def search(query: str, *, top_k: int = 5, mode: str = "hybrid") -> list[dict]:
    log.info("Searching (%s) | top_k=%d | query: %s", mode, top_k, query[:120])
    dense_vec = encode([query])[0]

    if mode == "dense":
        hits = CLIENT.query_points(
            collection_name=COLLECTION,
            query=dense_vec,
            using="dense",
            limit=top_k,
            with_payload=True,
        ).points
    else:
        sparse_indices, sparse_values = encode_sparse([query])[0]
        hits = CLIENT.query_points(
            collection_name=COLLECTION,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=20),
                Prefetch(
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using="sparse",
                    limit=20,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        ).points
    results = [
        {
            "score": h.score,
            "text": (h.payload or {}).get("text", ""),
            "doc": (h.payload or {}).get("doc", ""),
            "page": (h.payload or {}).get("page"),
            "paragraph": (h.payload or {}).get("paragraph"),
        }
        for h in hits
    ]
    if not results:
        log.warning("No hits returned for query: %s", query[:120])
    else:
        for i, r in enumerate(results, 1):
            log.info(
                "  [%d] score=%.4f | %s p.%s para=%s | %s",
                i, r["score"], r["doc"], r["page"], r["paragraph"],
                r["text"][:80].replace("\n", " "),
            )
    return results
