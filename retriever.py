"""Embed query and search Qdrant."""
from qdrant_client import QdrantClient

from embed import encode
from ingestion import COLLECTION

CLIENT = QdrantClient(url="http://localhost:6333")


def search(query: str, *, top_k: int = 5) -> list[dict]:
    hits = CLIENT.query_points(
        collection_name=COLLECTION,
        query=encode([query])[0],
        limit=top_k,
        with_payload=True,
    ).points
    out = []
    for h in hits:
        p = h.payload or {}
        out.append(
            {
                "score": h.score,
                "text": p.get("text", ""),
                "source": p.get("source", ""),
            }
        )
    return out
