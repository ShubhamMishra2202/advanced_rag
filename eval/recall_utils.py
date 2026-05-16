"""Retrieval recall via LLM judge (gpt-4o-mini).

Binary **evidence recall**: whether the union of retrieved chunks contains enough
information to support the reference answer for the query. Uses OPENAI_API_KEY
from the project `.env` (loaded with python-dotenv).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"
_ROOT = Path(__file__).resolve().parent.parent


def load_openai_env(*, project_root: Path | None = None) -> Path:
    """Load environment variables from ``<project_root>/.env`` if present.

    Returns the path that was loaded (or would be used).
    """
    root = project_root or _ROOT
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(env_path)
    else:
        load_dotenv()
    return env_path


def get_openai_client(*, project_root: Path | None = None) -> OpenAI:
    """Build an OpenAI client using ``OPENAI_API_KEY`` after loading ``.env``."""
    load_openai_env(project_root=project_root)
    key = os.environ.get("OPENAI_API_KEY")
    if not key or not key.strip():
        raise RuntimeError(
            "OPENAI_API_KEY is missing or empty. Set it in .env or the environment."
        )
    return OpenAI(api_key=key.strip())


_SYSTEM = (
    "You evaluate retrieval for RAG. Given a user query, a reference answer "
    "(gold), and retrieved text passages, decide if those passages TOGETHER "
    "contain the substantive information needed to justify or state that "
    "reference answer. Paraphrases count; exact wording is not required. "
    "Respond only with valid JSON matching the schema."
)


def judge_evidence_recall(
    query: str,
    ground_truth: str,
    retrieved_chunks: list[str],
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Ask the model whether retrieved chunks cover the reference answer.

    Returns a dict with at least:
      - ``recall`` (bool): True if evidence in chunks is sufficient for the gold answer.
      - ``reason`` (str): short justification.
    """
    own_client = client or get_openai_client(project_root=project_root)
    chunks_block = "\n\n".join(
        f"[{i}] {t.strip()}" for i, t in enumerate(retrieved_chunks, 1) if t and t.strip()
    )
    if not chunks_block.strip():
        return {"recall": False, "reason": "No retrieved passages."}

    user = (
        f"Query:\n{query}\n\n"
        f"Reference answer (gold):\n{ground_truth}\n\n"
        f"Retrieved passages:\n{chunks_block}\n\n"
        'Return JSON: {"recall": <true|false>, "reason": "<brief>"}'
    )
    resp = own_client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return {"recall": False, "reason": f"Invalid JSON from model: {raw[:200]}"}

    recall = bool(out.get("recall", False))
    reason = str(out.get("reason", "")).strip() or "(no reason)"
    return {"recall": recall, "reason": reason}


def mean_binary_recall(recalls: list[bool]) -> float:
    """Mean of binary recall flags (0.0–1.0). Empty input returns 0.0."""
    if not recalls:
        return 0.0
    return sum(1 for r in recalls if r) / len(recalls)


def aggregate_recall_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a list of ``judge_evidence_recall`` outputs."""
    flags = [bool(r.get("recall")) for r in results]
    n = len(flags)
    return {
        "n": n,
        "recall_count": sum(1 for f in flags if f),
        "mean_recall": mean_binary_recall(flags),
    }
