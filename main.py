import logging
import mlflow
from pathlib import Path

import logger as _log_setup
_log_setup.setup()

from embed import _MODEL_NAME, VECTOR_SIZE
from ingestion import ingest, CHUNK_SIZE, CHUNK_OVERLAP
from retriever import search
from llm import judge_evidence_recall, judge_retrieval_precision
from eval.metrics import reciprocal_rank

log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
PDF            = Path(__file__).resolve().parent / "data" / "Medical_document_advance_rag.pdf"
MODE           = "query"          # "ingest" | "query"
TOP_K          = 5
PDF_EXTRACTOR  = "pdfplumber"

# Describe what makes this run different — shows up as the run name in MLflow UI
#
# Format: <embedding_model> | <pdf_extractor> | <page_numbering> | <chunking_strategy> | <search_type> | <notes>
#
# History:
#   Run 1 — MiniLM-L6-v2    | pypdf      | PDF index (wrong offset) | char-500/80  | dense-only | baseline, p.58 was false top hit
#   Run 2 — MiniLM-L6-v2    | pdfplumber | PDF index (wrong offset) | char-500/80  | dense-only | pdfplumber fixed p.58 false hit, p.22→rank1 but page label wrong
#   Run 3 — MiniLM-L6-v2    | pdfplumber | printed footer numbers   | char-500/80  | dense-only | page metadata now correct (p.20 shows correctly)
#   Run 4 — BGE-base-en-v1.5 | pdfplumber | printed footer numbers  | char-500/80  | dense-only | larger 768d model, MRR dropped 1.0→0.5 vs MiniLM on complex query
RUN_DESCRIPTION = "run4 | BGE-base-en-v1.5 | pdfplumber | printed-footer-pages | char-500/80 | dense-only"

QUESTION = (
    "A 45-year-old heavy smoker presents at a primary care clinic with persistent cough, "
    "worsening breathlessness, a blood pressure reading of 165/100 mmHg, and fasting blood "
    "glucose of 7.2 mmol/L. He also reports he cannot stop smoking. According to WHO PEN, "
    "what combination of protocols applies to this patient, what diagnostic steps should be "
    "taken to distinguish his respiratory condition, what cardiovascular risk threshold "
    "determines whether he needs drug treatment, and what is the step-by-step approach to "
    "help him quit tobacco?"
)
GROUND_TRUTH = (
    "This patient requires three WHO PEN protocols: Protocol 1 (CVD/diabetes/hypertension), "
    "Protocol 2 (lifestyle counselling and tobacco cessation), and Protocol 3 (respiratory disease). "
    "For respiratory diagnosis: features favouring COPD over asthma include heavy smoking "
    "(>20 cigarettes/day for >15 years), symptoms starting in middle age, persistent daily cough "
    "and sputum, and worsening breathlessness with little day-to-day variation. "
    "For cardiovascular risk: drug treatment is indicated when blood pressure is ≥160/100 mmHg "
    "or when 10-year cardiovascular risk is ≥30%; this patient's BP of 165/100 meets the threshold. "
    "For tobacco cessation, the 5 A's apply: A1-ASK (confirm tobacco use), A2-ADVISE (quit clearly "
    "and strongly), A3-ASSESS (willingness to quit), A4-ASSIST (set quit date, inform family, remove "
    "cigarettes), A5-ARRANGE (follow-up to congratulate success or address relapse). Since the patient "
    "cannot stop, motivational counselling and health hazard information should also be provided."
)  # set to "" to skip LLM eval

MLFLOW_EXPERIMENT = "advanced_rag_retrieval"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_query() -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=RUN_DESCRIPTION):
        mlflow.log_params({
            "embedding_model": _MODEL_NAME,
            "vector_size":     VECTOR_SIZE,
            "chunk_size":      CHUNK_SIZE,
            "chunk_overlap":   CHUNK_OVERLAP,
            "top_k":           TOP_K,
            "pdf_extractor":   PDF_EXTRACTOR,
        })
        mlflow.set_tag("query", QUESTION[:250])
        mlflow.set_tag("description", RUN_DESCRIPTION)

        log.info("Query: %s", QUESTION)
        hits = search(QUESTION, top_k=TOP_K)

        if not hits:
            log.warning("No results returned")
            print("No results.")
            mlflow.log_metric("recall", 0)
            mlflow.log_metric("precision", 0.0)
            mlflow.log_metric("mrr", 0.0)
            return

        print()
        for i, h in enumerate(hits, 1):
            print(f"[{i}] score={h['score']:.4f} | {h['doc']} p.{h['page']}")
            print(f"    {h['text'][:200].replace(chr(10), ' ')}")
            print()
            mlflow.log_metric(f"score_rank_{i}", h["score"])
            mlflow.log_metric(f"page_rank_{i}", h["page"] or 0)

        mlflow.log_metric("top1_score", hits[0]["score"])
        mlflow.log_metric("mean_score", sum(h["score"] for h in hits) / len(hits))

        if not GROUND_TRUTH:
            log.info("LLM eval skipped (GROUND_TRUTH not set)")
            print("--- LLM eval skipped (set GROUND_TRUTH to enable) ---")
            return

        chunks = [h["text"] for h in hits]
        recall    = judge_evidence_recall(QUESTION, ground_truth=GROUND_TRUTH, retrieved_chunks=chunks)
        precision = judge_retrieval_precision(QUESTION, retrieved_chunks=chunks)
        mrr       = reciprocal_rank(precision["judgments"])

        mlflow.log_metrics({
            "recall":    int(recall["recall"]),
            "precision": precision["precision"],
            "mrr":       mrr,
        })
        mlflow.set_tag("recall_reason", recall["reason"][:250])

        # Store the full retrieved chunks as a readable artifact
        artifact = "\n\n".join(
            f"[{i}] p.{h['page']} | score={h['score']:.4f}\n{h['text']}"
            for i, h in enumerate(hits, 1)
        )
        mlflow.log_text(artifact, "retrieved_chunks.txt")

        log.info("=== Eval Summary ===")
        log.info("recall    : %s", recall["recall"])
        log.info("reason    : %s", recall["reason"])
        log.info("precision : %.2f (%d/%d relevant)", precision["precision"], precision["relevant_count"], precision["total_count"])
        log.info("MRR       : %.4f", mrr)

        print("--- LLM eval ---")
        print(f"recall    : {recall['recall']}")
        print(f"reason    : {recall['reason']}")
        print(f"precision : {precision['precision']:.2f} ({precision['relevant_count']}/{precision['total_count']} chunks relevant)")
        print(f"MRR       : {mrr:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=== advanced_rag | MODE=%s ===", MODE)
    if MODE == "ingest":
        n = ingest(PDF, recreate=True)
        log.info("Ingest finished — %d chunk(s) indexed", n)
        print(f"Indexed {n} chunk(s)")
    else:
        _run_query()
