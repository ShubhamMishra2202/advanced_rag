"""Load a PDF, chunk by page, embed, and upsert into Qdrant."""
import logging
from pathlib import Path
from uuid import NAMESPACE_DNS, uuid5
import pdfplumber
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from embed import VECTOR_SIZE, encode

log = logging.getLogger(__name__)

COLLECTION = "advanced_rag"
CLIENT = QdrantClient(url="http://localhost:6333")


def _ensure_collection(recreate: bool) -> None:
    if recreate and CLIENT.collection_exists(COLLECTION):
        log.warning("recreate=True — dropping existing collection '%s'", COLLECTION)
        CLIENT.delete_collection(COLLECTION)
    if not CLIENT.collection_exists(COLLECTION):
        log.info("Creating collection '%s' (size=%d, distance=COSINE)", COLLECTION, VECTOR_SIZE)
        CLIENT.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    else:
        log.info("Collection '%s' already exists — skipping creation", COLLECTION)


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
    log.info("Ingesting PDF: %s", pdf_path)
    _ensure_collection(recreate)
    doc = pdf_path.name
    with pdfplumber.open(str(pdf_path)) as pdf:
        pages = [
            page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            for page in pdf.pages
        ]
    log.info("Extracted text from %d page(s)", len(pages))
    empty_pages = sum(1 for p in pages if not p.strip())
    if empty_pages:
        log.warning("%d page(s) yielded no text (images/scans?)", empty_pages)

    rows: list[tuple[int, int, str]] = [
        (page_num, para_idx, chunk)
        for page_num, page_text in enumerate(pages, start=1)
        for para_idx, chunk in enumerate(_chunk(page_text), start=1)
    ]
    log.info("Created %d chunk(s) across %d page(s)", len(rows), len(pages))
    if not rows:
        log.warning("No chunks produced — aborting ingest")
        return 0

    log.debug("Embedding %d chunk(s)...", len(rows))
    vectors = encode([chunk for _, _, chunk in rows])
    points = [
        PointStruct(
            id=str(uuid5(NAMESPACE_DNS, f"{pdf_path}:{doc}:{page}:{para}:{chunk[:64]}")),
            vector=vec,
            payload={"text": chunk, "doc": doc, "page": page, "paragraph": para},
        )
        for (page, para, chunk), vec in zip(rows, vectors)
    ]
    log.debug("Upserting %d point(s) into Qdrant...", len(points))
    CLIENT.upsert(collection_name=COLLECTION, points=points)
    log.info("Ingest complete — %d point(s) upserted into '%s'", len(points), COLLECTION)
    return len(points)
