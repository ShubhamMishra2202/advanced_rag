"""RAG evaluation metrics (non-LLM).

LLM-judged functions (judge_evidence_recall, judge_retrieval_precision) live in llm.py.

Aggregation helpers:
  aggregate_recall_results    -- summarise a list of judge_evidence_recall outputs
  aggregate_precision_results -- summarise a list of judge_retrieval_precision outputs

Similarity-based (no LLM):
  cosine             -- cosine similarity between two vectors
  run_similarity_eval -- batch eval: top1_score + max_gt_sim over a dataset
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_recall_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    flags = [bool(r.get("recall")) for r in results]
    n = len(flags)
    return {
        "n": n,
        "recall_count": sum(1 for f in flags if f),
        "mean_recall": sum(1 for f in flags if f) / n if n else 0.0,
    }


def aggregate_precision_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    precisions = [float(r.get("precision", 0.0)) for r in results]
    n = len(precisions)
    return {
        "n": n,
        "mean_precision": sum(precisions) / n if n else 0.0,
    }


# ---------------------------------------------------------------------------
# MRR — derived from precision judgments, no extra LLM call
# ---------------------------------------------------------------------------

def reciprocal_rank(judgments: list[dict[str, Any]]) -> float:
    """RR for a single query: 1/rank of the first relevant chunk, or 0.0.

    Expects the ``judgments`` list from ``judge_retrieval_precision``,
    each entry having ``index`` (1-based) and ``relevant`` fields.
    """
    for j in sorted(judgments, key=lambda x: x.get("index", 0)):
        if j.get("relevant", False):
            return 1.0 / j["index"]
    return 0.0


def mean_reciprocal_rank(rr_scores: list[float]) -> float:
    """MRR over a collection of per-query reciprocal rank scores."""
    if not rr_scores:
        return 0.0
    return sum(rr_scores) / len(rr_scores)


# ---------------------------------------------------------------------------
# Similarity-based metrics
# ---------------------------------------------------------------------------

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def run_similarity_eval(
    dataset_path: Path,
    *,
    top_k: int = 5,
    limit: int = 0,
    gt_sim_threshold: float = 0.5,
    mode: str = "hybrid",
) -> dict[str, Any]:
    """Batch similarity eval over a JSON dataset.

    Each row must have ``query`` and ``ground_truth`` fields.
    Requires ``embed.encode`` and ``retriever.search`` on sys.path.
    Returns aggregated metrics dict.
    """
    import sys
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from embed import encode
    from retriever import search

    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if limit > 0:
        rows = rows[:limit]

    top1_scores: list[float] = []
    max_gt_sims: list[float] = []

    for row in rows:
        hits = search(row["query"], top_k=top_k, mode=mode)
        if not hits:
            top1_scores.append(0.0)
            max_gt_sims.append(0.0)
            continue

        top1_scores.append(float(hits[0]["score"]))
        texts = [h["text"] for h in hits if h.get("text")]
        if not texts:
            max_gt_sims.append(0.0)
            continue

        gt_vec = encode([row["ground_truth"]])[0]
        chunk_vecs = encode(texts)
        max_gt_sims.append(max(cosine(gt_vec, cv) for cv in chunk_vecs))

    n = len(rows)
    mean_top1 = sum(top1_scores) / n if n else 0.0
    mean_gt = sum(max_gt_sims) / n if n else 0.0
    above = sum(1 for s in max_gt_sims if s >= gt_sim_threshold)
    return {
        "n": n,
        "top_k": top_k,
        "mean_top1_score": mean_top1,
        "mean_max_gt_sim": mean_gt,
        "gt_sim_threshold": gt_sim_threshold,
        "above_threshold": above,
        "above_threshold_pct": 100.0 * above / n if n else 0.0,
    }
