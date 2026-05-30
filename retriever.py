"""Embed query and search Qdrant."""
import logging
from qdrant_client import QdrantClient
from embed import encode
from ingestion import COLLECTION

log = logging.getLogger(__name__)
CLIENT = QdrantClient(url="http://localhost:6333")


def search(query: str, *, top_k: int = 5) -> list[dict]:
    log.info("Searching | top_k=%d | query: %s", top_k, query[:120])
    hits = CLIENT.query_points(
        collection_name=COLLECTION,
        query=encode([query])[0],
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
