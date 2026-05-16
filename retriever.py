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
    return [
        {
            "score": h.score,
            "text": (h.payload or {}).get("text", ""),
            "doc": (h.payload or {}).get("doc", ""),
            "page": (h.payload or {}).get("page"),
            "paragraph": (h.payload or {}).get("paragraph"),
        }
        for h in hits
    ]
