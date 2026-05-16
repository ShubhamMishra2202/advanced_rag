#!/usr/bin/env python3
"""Evaluate retrieval against eval/rag_eval_dataset_attention_paper.json.

Metrics (no LLM generation in this project):
  - top1_score: mean Qdrant cosine similarity of the best hit (query↔chunk).
  - max_gt_sim: mean over questions of max cosine(ground_truth, chunk_i) across
    top-k retrieved chunks — how semantically close retrieved text is to the
    reference answer.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embed import encode  # noqa: E402
from retriever import search  # noqa: E402

DATASET = Path(__file__).resolve().parent / "rag_eval_dataset_attention_paper.json"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG retrieval eval (ground-truth similarity).")
    ap.add_argument("--top-k", type=int, default=5, help="Retrieval depth.")
    ap.add_argument("--limit", type=int, default=0, help="Only first N items (0 = all).")
    ap.add_argument("--sample", action="store_true", help="Print one example query + hits.")
    args = ap.parse_args()

    rows = json.loads(DATASET.read_text(encoding="utf-8"))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    top1_scores: list[float] = []
    max_gt_sims: list[float] = []

    for i, row in enumerate(rows):
        q = row["query"]
        gt = row["ground_truth"]
        hits = search(q, top_k=args.top_k)
        if not hits:
            top1_scores.append(0.0)
            max_gt_sims.append(0.0)
            continue

        top1_scores.append(float(hits[0]["score"]))
        texts = [h["text"] for h in hits if h.get("text")]
        if not texts:
            max_gt_sims.append(0.0)
            continue

        gt_vec = encode([gt])[0]
        chunk_vecs = encode(texts)
        best = max(_cosine(gt_vec, cv) for cv in chunk_vecs)
        max_gt_sims.append(best)

        if args.sample and i == 0:
            print("=== sample ===")
            print("query:", q)
            print("top1_score:", round(hits[0]["score"], 4))
            print("max_gt_sim:", round(best, 4))
            print("top1 preview:", (hits[0].get("text") or "")[:400].replace("\n", " "))
            print()

    n = len(rows)
    mean_top1 = sum(top1_scores) / n if n else 0.0
    mean_gt = sum(max_gt_sims) / n if n else 0.0

    print(f"dataset: {DATASET.name}")
    print(f"items: {n}  top_k: {args.top_k}")
    print(f"mean top1_score (query↔chunk, Qdrant): {mean_top1:.4f}")
    print(f"mean max_gt_sim (ground_truth↔best chunk): {mean_gt:.4f}")
    thr = 0.5
    above = sum(1 for s in max_gt_sims if s >= thr)
    print(f"max_gt_sim >= {thr}: {above}/{n} ({100.0 * above / n:.1f}%)")


if __name__ == "__main__":
    main()
