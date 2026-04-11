"""Load text files, chunk, embed, upsert into Qdrant."""
from pathlib import Path
from uuid import NAMESPACE_DNS, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from embed import VECTOR_SIZE, encode

COLLECTION = "advanced_rag"
CLIENT = QdrantClient(url="http://localhost:6333")


def _ensure_collection(recreate: bool) -> None:
    if recreate and CLIENT.collection_exists(COLLECTION):
        CLIENT.delete_collection(COLLECTION)
    if not CLIENT.collection_exists(COLLECTION):
        CLIENT.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def _chunk(text: str, max_chars: int = 500, overlap: int = 80) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def ingest(data_dir: str | Path, *, recreate: bool = False) -> int:
    """Index all `.txt` files under `data_dir`. Returns number of chunks stored."""
    data_dir = Path(data_dir)
    _ensure_collection(recreate)
    rows: list[tuple[Path, int, str]] = []
    for path in sorted(data_dir.rglob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="replace")
        for i, chunk in enumerate(_chunk(raw)):
            rows.append((path, i, chunk))
    if not rows:
        return 0
    vectors = encode([c for _, _, c in rows])
    points = [
        PointStruct(
            id=str(uuid5(NAMESPACE_DNS, f"{path}:{i}:{chunk[:64]}")),
            vector=vec,
            payload={"text": chunk, "source": str(path.relative_to(data_dir))},
        )
        for (path, i, chunk), vec in zip(rows, vectors)
    ]
    CLIENT.upsert(collection_name=COLLECTION, points=points)
    return len(points)
