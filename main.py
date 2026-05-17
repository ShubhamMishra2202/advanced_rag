from pathlib import Path
from ingestion import ingest
from retriever import search
from llm import format_context, judge_evidence_recall, judge_retrieval_precision
from eval.metrics import reciprocal_rank

PDF = Path(__file__).resolve().parent / "data" / "Medical_document_advance_rag.pdf"
MODE = "query"  # "ingest" | "query"
QUESTION = "What are the five core parameters used in the NCD costing tool to estimate scale-up costs, and how do they interact?"
GROUND_TRUTH = "The five parameters are: (1) population of the country/region, (2) prevalence/incidence of the disease or risk factor, (3) coverage — proportion of population in need receiving the intervention, (4) resource quantities needed to implement the intervention (human resources, medicines, equipment), and (5) prices or unit costs for each resource item. Population times prevalence defines the population at risk/in need; resource use times price provides cost per case; and coverage is the main mechanism by which scale-up takes place over time"  # set to "" to skip LLM eval

if __name__ == "__main__":
    if MODE == "ingest":
        print(f"Indexed {ingest(PDF)} chunk(s)")
    else:
        hits = search(QUESTION)
        if not hits:
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
                print("--- LLM eval ---")
                print(f"recall    : {recall['recall']}")
                print(f"reason    : {recall['reason']}")
                print(f"precision : {precision['precision']:.2f} ({precision['relevant_count']}/{precision['total_count']} chunks relevant)")
                print(f"MRR       : {reciprocal_rank(precision['judgments']):.4f}")
            else:
                print("--- LLM eval skipped (set GROUND_TRUTH to enable) ---")
