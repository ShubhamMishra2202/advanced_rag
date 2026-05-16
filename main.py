from pathlib import Path
from ingestion import ingest
from retriever import search
from llm import format_context

PDF = Path(__file__).resolve().parent / "data" / "attention_is_all_you_need-1.pdf"
MODE = "ingest"  # "ingest" | "query"
QUESTION = "What is attention?"

if __name__ == "__main__":
    if MODE == "ingest":
        print(f"Indexed {ingest(PDF)} chunk(s)")
    else:
        hits = search(QUESTION)
        print(format_context(hits) if hits else "No results.")
