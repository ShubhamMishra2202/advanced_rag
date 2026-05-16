#!/usr/bin/env python3
"""Run LLM-judged evidence recall on the eval JSON (gpt-4o-mini)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
for p in (ROOT, EVAL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from retriever import search  # noqa: E402

from recall_utils import (  # noqa: E402
    aggregate_recall_results,
    judge_evidence_recall,
)

DATASET = Path(__file__).resolve().parent / "rag_eval_dataset_attention_paper.json"
TOP_K = 5
LIMIT = 0
SAMPLE = False


def main() -> None:
    rows = json.loads(DATASET.read_text(encoding="utf-8"))
    if LIMIT > 0:
        rows = rows[:LIMIT]

    results: list[dict] = []
    for i, row in enumerate(rows):
        q = row["query"]
        gt = row["ground_truth"]
        hits = search(q, top_k=TOP_K)
        chunks = [h.get("text") or "" for h in hits]
        out = judge_evidence_recall(q, gt, chunks)
        out["id"] = row.get("id")
        results.append(out)

        if SAMPLE and i == 0:
            print("=== sample ===")
            print("query:", q)
            print("recall:", out["recall"])
            print("reason:", out["reason"])
            print()

    summary = aggregate_recall_results(results)
    print(f"dataset: {DATASET.name}")
    print(f"model: gpt-4o-mini  top_k: {TOP_K}")
    print(f"mean evidence recall: {summary['mean_recall']:.4f} ({summary['recall_count']}/{summary['n']})")


if __name__ == "__main__":
    main()
