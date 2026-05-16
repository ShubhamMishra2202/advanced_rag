"""Load a PDF, chunk by page, embed, and upsert into Qdrant."""
from pathlib import Path
from uuid import NAMESPACE_DNS, uuid5
from pypdf import PdfReader
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
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def ingest(pdf_path: str | Path, *, recreate: bool = False) -> int:
    pdf_path = Path(pdf_path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")
    _ensure_collection(recreate)
    doc = pdf_path.name
    pages = [page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages]
    rows: list[tuple[int, int, str]] = [
        (page_num, para_idx, chunk)
        for page_num, page_text in enumerate(pages, start=1)
        for para_idx, chunk in enumerate(_chunk(page_text), start=1)
    ]
    if not rows:
        return 0
    vectors = encode([chunk for _, _, chunk in rows])
    points = [
        PointStruct(
            id=str(uuid5(NAMESPACE_DNS, f"{pdf_path}:{doc}:{page}:{para}:{chunk[:64]}")),
            vector=vec,
            payload={"text": chunk, "doc": doc, "page": page, "paragraph": para},
        )
        for (page, para, chunk), vec in zip(rows, vectors)
    ]
    CLIENT.upsert(collection_name=COLLECTION, points=points)
    return len(points)
