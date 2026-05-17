from pathlib import Path
from ingestion import ingest
from retriever import search
from llm import format_context
from eval.recall_utils import judge_evidence_recall

PDF = Path(__file__).resolve().parent / "data" / "attention_is_all_you_need-1.pdf"
MODE = "query"  # "ingest" | "query"
QUESTION = "How does the Transformer's computational cost compare to a restricted self-attention variant for very long sequences?"
GROUND_TRUTH = "Restricted self-attention (neighborhood size r) reduces complexity to O(r·n·d) per layer and O(1) sequential ops, but increases max path length to O(n/r). Standard self-attention costs O(n²·d) — the square term becomes expensive for very long sequences, motivating the restricted variant as future work"  # set to "" to skip LLM eval

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
                verdict = judge_evidence_recall(QUESTION, ground_truth=GROUND_TRUTH, retrieved_chunks=chunks)
                print("--- LLM eval ---")
                print(f"recall : {verdict['recall']}")
                print(f"reason : {verdict['reason']}")
            else:
                print("--- LLM eval skipped (set GROUND_TRUTH to enable) ---")
