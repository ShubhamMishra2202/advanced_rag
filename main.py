import logging
from pathlib import Path
import logger as _log_setup
_log_setup.setup()

from ingestion import ingest
from retriever import search
from llm import format_context, judge_evidence_recall, judge_retrieval_precision
from eval.metrics import reciprocal_rank

log = logging.getLogger(__name__)

PDF = Path(__file__).resolve().parent / "data" / "Medical_document_advance_rag.pdf"
MODE = "query"  # "ingest" | "query"
QUESTION = "What clinical features distinguish asthma from COPD according to Protocol 3?"
GROUND_TRUTH = "Features favouring asthma include: previous diagnosis of asthma, symptoms since childhood or early adulthood, history of hayfever/eczema/allergies, intermittent symptoms with asymptomatic periods, symptoms worse at night or early morning, symptoms triggered by respiratory infection/exercise/weather/stress, and symptoms responding to salbutamol. Features favouring COPD include: previous diagnosis of COPD, history of heavy smoking (>20 cigarettes/day for >15 years), history of heavy and prolonged exposure to burning fossil fuels in enclosed space or occupational dust, symptoms starting in middle age (after 40), symptoms worsening slowly over a long period, and long history of daily cough and sputum production"  # set to "" to skip LLM eval

if __name__ == "__main__":
    log.info("=== advanced_rag | MODE=%s ===", MODE)
    if MODE == "ingest":
        n = ingest(PDF, recreate=True)
        log.info("Ingest finished — %d chunk(s) indexed", n)
        print(f"Indexed {n} chunk(s)")
    else:
        log.info("Query: %s", QUESTION)
        hits = search(QUESTION)
        if not hits:
            log.warning("No results returned")
            print("No results.")
        else:
            for i, h in enumerate(hits, 1):
                print(f"[{i}] score={h['score']:.4f} | {h['doc']} p.{h['page']}")
                print(f"    {h['text'][:200].replace(chr(10), ' ')}")
                print()

            if GROUND_TRUTH:
                chunks = [h["text"] for h in hits]
                recall = judge_evidence_recall(QUESTION, ground_truth=GROUND_TRUTH, retrieved_chunks=chunks)
                precision = judge_retrieval_precision(QUESTION, retrieved_chunks=chunks)
                mrr = reciprocal_rank(precision["judgments"])
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
            else:
                log.info("LLM eval skipped (GROUND_TRUTH not set)")
                print("--- LLM eval skipped (set GROUND_TRUTH to enable) ---")
