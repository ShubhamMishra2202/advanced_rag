import argparse
from pathlib import Path

from ingestion import ingest
from llm import format_context
from retriever import search

DEFAULT_DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    p = argparse.ArgumentParser(description="Minimal RAG: ingest .txt files, query Qdrant.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="Index all .txt files under data/")
    pi.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Directory of .txt files")
    pi.add_argument("--recreate", action="store_true", help="Drop collection and re-index")

    pq = sub.add_parser("query", help="Semantic search")
    pq.add_argument("question", type=str)
    pq.add_argument("-k", type=int, default=5, help="Top-k results")

    args = p.parse_args()
    if args.cmd == "ingest":
        n = ingest(args.data, recreate=args.recreate)
        print(f"Indexed {n} chunk(s) from {args.data}")
    else:
        hits = search(args.question, top_k=args.k)
        if not hits:
            print("No results (ingest data first, or broaden the question).")
            return
        print(format_context(hits))


if __name__ == "__main__":
    main()
