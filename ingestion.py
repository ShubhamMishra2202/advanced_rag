"""Load a PDF, chunk by page, embed, and upsert into Qdrant."""
import logging
import re
from pathlib import Path
from uuid import NAMESPACE_DNS, uuid5
import pdfplumber
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, PointStruct, VectorParams,
    SparseVectorParams, SparseVector,
)
from embed import VECTOR_SIZE, encode, encode_sparse

log = logging.getLogger(__name__)

COLLECTION    = "advanced_rag"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 80
CLIENT = QdrantClient(url="http://localhost:6333")

# Boilerplate running header present on almost every page of this PDF
_RUNNING_HEADER = (
    "Package of essential noncommunicable (PEN) disease interventions "
    "for primary health care in low-resource settings"
)


def _is_toc_page(text: str) -> bool:
    """True when >35% of non-empty lines are bare page numbers — i.e. a TOC page."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 5:
        return False
    digit_lines = sum(1 for l in lines if re.fullmatch(r"\d+", l))
    return digit_lines / len(lines) > 0.35


def _is_bibliography_page(text: str) -> bool:
    """True when the page is a search-strategy or bibliography section."""
    lower = text.lower()
    return "systematic reviews" in lower and any(
        kw in lower for kw in ("english language", "amstar", "search strategy", "mesh")
    )


def _clean_page(text: str) -> str:
    """Strip the running header and collapse excess blank lines."""
    text = text.replace(_RUNNING_HEADER, "")
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ensure_collection(recreate: bool) -> None:
    if recreate and CLIENT.collection_exists(COLLECTION):
        log.warning("recreate=True — dropping existing collection '%s'", COLLECTION)
        CLIENT.delete_collection(COLLECTION)
    if not CLIENT.collection_exists(COLLECTION):
        log.info("Creating collection '%s' (dense=%d-d COSINE, sparse=BM25)", COLLECTION, VECTOR_SIZE)
        CLIENT.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )
    else:
        log.info("Collection '%s' already exists — skipping creation", COLLECTION)


def _printed_page_num(text: str) -> int | None:
    """Return the printed page number from the last line of a page, or None."""
    last_line = text.strip().split("\n")[-1].strip() if text.strip() else ""
    return int(last_line) if re.fullmatch(r"\d+", last_line) else None


def _chunk(text: str, max_chars: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
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

    # Resolve each page's number: prefer the printed footer number, fall back to PDF index.
    page_numbers: list[int] = []
    for pdf_idx, page_text in enumerate(pages, start=1):
        printed = _printed_page_num(page_text)
        if printed is not None:
            page_numbers.append(printed)
        else:
            log.debug("No printed page number on PDF index %d — using PDF index", pdf_idx)
            page_numbers.append(pdf_idx)

    # ── Structural page filtering ────────────────────────────────────────────
    filtered_pages: list[tuple[int, str]] = []
    skipped_toc = skipped_bib = skipped_empty = 0
    for page_num, page_text in zip(page_numbers, pages):
        if not page_text.strip():
            skipped_empty += 1
            continue
        if _is_toc_page(page_text):
            log.debug("Skipping TOC page p.%d", page_num)
            skipped_toc += 1
            continue
        if _is_bibliography_page(page_text):
            log.debug("Skipping bibliography page p.%d", page_num)
            skipped_bib += 1
            continue
        filtered_pages.append((page_num, _clean_page(page_text)))

    log.info(
        "Page filter: kept %d / %d pages | skipped toc=%d bib=%d empty=%d",
        len(filtered_pages), len(pages), skipped_toc, skipped_bib, skipped_empty,
    )

    rows: list[tuple[int, int, str]] = [
        (page_num, para_idx, chunk)
        for page_num, page_text in filtered_pages
        for para_idx, chunk in enumerate(_chunk(page_text), start=1)
    ]
    log.info("Created %d chunk(s) across %d page(s)", len(rows), len(filtered_pages))
    if not rows:
        log.warning("No chunks produced — aborting ingest")
        return 0

    log.debug("Embedding %d chunk(s)...", len(rows))
    texts = [chunk for _, _, chunk in rows]
    vectors = encode(texts)
    sparse_vectors = encode_sparse(texts)
    points = [
        PointStruct(
            id=str(uuid5(NAMESPACE_DNS, f"{pdf_path}:{doc}:{page}:{para}:{chunk[:64]}")),
            vector={
                "dense": dense_vec,
                "sparse": SparseVector(indices=sparse_indices, values=sparse_values),
            },
            payload={"text": chunk, "doc": doc, "page": page, "paragraph": para},
        )
        for (page, para, chunk), dense_vec, (sparse_indices, sparse_values)
        in zip(rows, vectors, sparse_vectors)
    ]
    log.debug("Upserting %d point(s) into Qdrant...", len(points))
    CLIENT.upsert(collection_name=COLLECTION, points=points)
    log.info("Ingest complete — %d point(s) upserted into '%s'", len(points), COLLECTION)
    return len(points)
