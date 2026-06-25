"""Batch eval runner — compare dense vs hybrid retrieval across the WHO PEN dataset.

Usage:
    # Compare both modes (default — no LLM cost):
    uv run python eval/run_eval.py

    # Single mode:
    uv run python eval/run_eval.py --retrieval dense
    uv run python eval/run_eval.py --retrieval hybrid

    # Filter by difficulty or category:
    uv run python eval/run_eval.py --difficulty hard
    uv run python eval/run_eval.py --category protocol

    # Enable LLM judging (costs API calls — use --limit to control):
    uv run python eval/run_eval.py --llm --limit 10

Metrics (no-LLM mode):
    page_hit_rate   — % of queries where the expected page appears anywhere in top-K
    top1_accuracy   — % of queries where rank-1 chunk is from the expected page
    page_mrr        — mean 1/rank of the first chunk from the expected page

Metrics (--llm mode, adds):
    recall          — % queries where LLM judge says ground truth is covered
    precision       — mean fraction of retrieved chunks judged relevant
    mrr             — mean 1/rank of first relevant chunk (LLM-judged)
"""
import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import logger as _log_setup
_log_setup.setup()

import mlflow
from retriever import search
from eval.metrics import reciprocal_rank, mean_reciprocal_rank

log = logging.getLogger(__name__)

DATASET   = Path(__file__).resolve().parent / "rag_eval_dataset_who_pen.json"
MLFLOW_EX = "advanced_rag_retrieval"


# ── Core runner ───────────────────────────────────────────────────────────────

def _eval_one(row: dict, *, top_k: int, mode: str, use_llm: bool) -> dict:
    hits = search(row["query"], top_k=top_k, mode=mode)
    result = {
        "id":            row["id"],
        "category":      row["category"],
        "difficulty":    row["difficulty"],
        "expected_page": row["page"],
    }

    if not hits:
        return {**result, "page_hit": False, "top1_hit": False, "page_rr": 0.0}

    pages       = [h["page"] for h in hits]
    expected    = row["page"]
    first_rank  = next((i + 1 for i, p in enumerate(pages) if p == expected), None)

    result["page_hit"]  = first_rank is not None
    result["top1_hit"]  = pages[0] == expected
    result["page_rr"]   = 1.0 / first_rank if first_rank else 0.0
    result["top1_score"] = hits[0]["score"]
    result["retrieved_pages"] = pages

    if use_llm:
        from llm import judge_evidence_recall, judge_retrieval_precision
        chunks   = [h["text"] for h in hits]
        recall_r = judge_evidence_recall(
            row["query"], ground_truth=row["ground_truth"], retrieved_chunks=chunks
        )
        prec_r   = judge_retrieval_precision(row["query"], retrieved_chunks=chunks)
        mrr      = reciprocal_rank(prec_r["judgments"])
        result.update({
            "recall":    recall_r["recall"],
            "precision": prec_r["precision"],
            "mrr":       mrr,
        })

    return result


def run_mode(rows: list[dict], *, top_k: int, mode: str, use_llm: bool) -> list[dict]:
    results = []
    for i, row in enumerate(rows, 1):
        log.info("[%d/%d] %s | id=%d | %s", i, len(rows), mode, row["id"], row["query"][:60])
        results.append(_eval_one(row, top_k=top_k, mode=mode, use_llm=use_llm))
    return results


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(results: list[dict], use_llm: bool) -> dict:
    n = len(results)
    if n == 0:
        return {}
    agg = {
        "n":              n,
        "page_hit_rate":  sum(r["page_hit"] for r in results) / n,
        "top1_accuracy":  sum(r["top1_hit"] for r in results) / n,
        "page_mrr":       mean_reciprocal_rank([r["page_rr"] for r in results]),
    }
    if use_llm:
        agg["recall_rate"] = sum(1 for r in results if r.get("recall")) / n
        agg["mean_precision"] = sum(r.get("precision", 0.0) for r in results) / n
        agg["llm_mrr"] = mean_reciprocal_rank([r.get("mrr", 0.0) for r in results])
    return agg


# ── Display ───────────────────────────────────────────────────────────────────

_DIFF_ICON = {"easy": "·", "medium": "○", "hard": "●"}

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"

def _rr(v: float) -> str:
    return f"{v:.3f}"

def print_per_query(results: list[dict], mode: str, use_llm: bool) -> None:
    header = f"\n{'─'*80}\n  {mode.upper()} — per-query results\n{'─'*80}"
    print(header)
    col = "  {:>3}  {:6}  {:^4}  {:^6}  {:^6}  {}"
    print(col.format("ID", "diff", "p.gt", "hit?", "rank1?", "query"))
    print("  " + "─" * 76)
    for r in results:
        icon   = _DIFF_ICON.get(r["difficulty"], "?")
        hit    = "✓" if r["page_hit"] else "✗"
        top1   = "✓" if r["top1_hit"] else "✗"
        query  = f"[{r['category']}] " + str(r.get("query_short", ""))
        print(col.format(r["id"], icon, r["expected_page"], hit, top1, query))


def print_summary(agg_by_mode: dict[str, dict], use_llm: bool) -> None:
    modes = list(agg_by_mode.keys())
    print(f"\n{'═'*60}")
    print("  SUMMARY")
    print(f"{'═'*60}")

    metrics = [
        ("n",             "Queries",        str),
        ("page_hit_rate", "Page Hit Rate",  _pct),
        ("top1_accuracy", "Top-1 Accuracy", _pct),
        ("page_mrr",      "Page MRR",       _rr),
    ]
    if use_llm:
        metrics += [
            ("recall_rate",    "Recall (LLM)",    _pct),
            ("mean_precision", "Precision (LLM)", _pct),
            ("llm_mrr",        "MRR (LLM)",       _rr),
        ]

    # Header
    label_w = 20
    col_w   = 14
    header  = " " * label_w + "".join(f"{m:>{col_w}}" for m in modes)
    print(header)
    print(" " * label_w + "─" * (col_w * len(modes)))

    for key, label, fmt in metrics:
        row = f"  {label:<{label_w - 2}}"
        for m in modes:
            val = agg_by_mode[m].get(key)
            row += f"{fmt(val):>{col_w}}" if val is not None else f"{'—':>{col_w}}"
        print(row)

    # Delta column when comparing exactly two modes
    if len(modes) == 2:
        m1, m2 = modes
        print(f"\n  Delta ({m2} − {m1}):")
        for key, label, fmt in metrics[1:]:  # skip 'n'
            v1 = agg_by_mode[m1].get(key)
            v2 = agg_by_mode[m2].get(key)
            if v1 is not None and v2 is not None and isinstance(v1, float):
                delta = v2 - v1
                sign  = "+" if delta >= 0 else ""
                print(f"  {label:<{label_w - 2}}{sign}{fmt(delta):>{col_w}}")

    print(f"{'═'*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch retrieval eval: dense vs hybrid")
    parser.add_argument(
        "--retrieval", nargs="+", choices=["dense", "hybrid"],
        default=["dense", "hybrid"],
        help="Retrieval mode(s) to evaluate (default: both)",
    )
    parser.add_argument("--top-k",     type=int, default=5)
    parser.add_argument("--limit",     type=int, default=0,
                        help="Max queries to evaluate (0 = all)")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"],
                        help="Filter by difficulty")
    parser.add_argument("--category",  help="Filter by category (e.g. protocol, diabetes)")
    parser.add_argument("--llm",       action="store_true",
                        help="Enable LLM-judged recall/precision/MRR (costs API calls)")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Skip MLflow logging")
    args = parser.parse_args()

    # Load + filter dataset
    rows = json.loads(DATASET.read_text())
    if args.difficulty:
        rows = [r for r in rows if r["difficulty"] == args.difficulty]
    if args.category:
        rows = [r for r in rows if r["category"] == args.category]
    if args.limit > 0:
        rows = rows[:args.limit]

    print(f"\nDataset: {len(rows)} queries | top_k={args.top_k} | LLM eval: {'yes' if args.llm else 'no (page-based)'}")
    print(f"Modes:   {', '.join(args.retrieval)}\n")

    # Run each mode
    agg_by_mode: dict[str, dict] = {}
    results_by_mode: dict[str, list[dict]] = {}

    for mode in args.retrieval:
        print(f"▶ Running {mode}...")
        raw = run_mode(rows, top_k=args.top_k, mode=mode, use_llm=args.llm)

        # Attach short query string for display
        for r, row in zip(raw, rows):
            r["query_short"] = row["query"][:50]

        results_by_mode[mode] = raw
        agg_by_mode[mode] = aggregate(raw, use_llm=args.llm)
        print(f"  done — page_hit_rate={_pct(agg_by_mode[mode]['page_hit_rate'])}  top1_acc={_pct(agg_by_mode[mode]['top1_accuracy'])}  page_mrr={_rr(agg_by_mode[mode]['page_mrr'])}")

    # Print per-query tables
    for mode in args.retrieval:
        print_per_query(results_by_mode[mode], mode=mode, use_llm=args.llm)

    # Print summary comparison
    print_summary(agg_by_mode, use_llm=args.llm)

    # MLflow logging
    if not args.no_mlflow:
        mlflow.set_experiment(MLFLOW_EX)
        for mode, agg in agg_by_mode.items():
            tag = f"eval | {mode} | n={agg['n']} | top_k={args.top_k}"
            if args.difficulty:
                tag += f" | {args.difficulty}"
            with mlflow.start_run(run_name=tag):
                mlflow.log_params({
                    "retrieval_mode": mode,
                    "top_k":          args.top_k,
                    "n_queries":      agg["n"],
                    "difficulty":     args.difficulty or "all",
                    "category":       args.category  or "all",
                    "llm_eval":       args.llm,
                })
                mlflow.log_metrics({k: v for k, v in agg.items()
                                    if isinstance(v, float)})
