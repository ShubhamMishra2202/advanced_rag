from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"
_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

def _load_env(project_root: Path | None = None) -> None:
    root = project_root or _ROOT
    env_path = root / ".env"
    load_dotenv(env_path if env_path.is_file() else None)


def get_openai_client(*, project_root: Path | None = None) -> OpenAI:
    _load_env(project_root=project_root)
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing or empty. Set it in .env or the environment.")
    return OpenAI(api_key=key)


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        header = f"{h.get('doc', '?')} | p.{h.get('page')} | para={h.get('paragraph')}"
        parts.append(f"[{i}] ({header})\n{h.get('text', '')}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM-judged recall
# ---------------------------------------------------------------------------

_RECALL_SYSTEM = (
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
    """Return ``{recall: bool, reason: str}`` — do chunks cover the gold answer?"""
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
        messages=[{"role": "system", "content": _RECALL_SYSTEM}, {"role": "user", "content": user}],
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return {"recall": False, "reason": f"Invalid JSON from model: {raw[:200]}"}

    return {
        "recall": bool(out.get("recall", False)),
        "reason": str(out.get("reason", "")).strip() or "(no reason)",
    }


# ---------------------------------------------------------------------------
# LLM-judged precision
# ---------------------------------------------------------------------------

_PRECISION_SYSTEM = (
    "You evaluate retrieval for RAG. Given a user query and a numbered list of "
    "retrieved text passages, judge each passage individually: is it relevant to "
    "answering the query? A passage is relevant if it contains information directly "
    "useful for answering the query — even partial relevance counts. "
    "Respond only with valid JSON matching the schema."
)


def judge_retrieval_precision(
    query: str,
    retrieved_chunks: list[str],
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Return precision over retrieved chunks: what fraction are relevant to the query?

    Returns ``{precision: float, relevant_count: int, total_count: int, judgments: list}``.
    Each judgment is ``{index: int, relevant: bool, reason: str}``.
    """
    own_client = client or get_openai_client(project_root=project_root)
    chunks = [t.strip() for t in retrieved_chunks if t and t.strip()]
    if not chunks:
        return {"precision": 0.0, "relevant_count": 0, "total_count": 0, "judgments": []}

    chunks_block = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(chunks, 1))
    user = (
        f"Query:\n{query}\n\n"
        f"Retrieved passages:\n{chunks_block}\n\n"
        "For each passage return an entry in a JSON array:\n"
        '{"judgments": [{"index": <int>, "relevant": <true|false>, "reason": "<brief>"}]}'
    )
    resp = own_client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": _PRECISION_SYSTEM}, {"role": "user", "content": user}],
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return {"precision": 0.0, "relevant_count": 0, "total_count": len(chunks), "judgments": []}

    judgments = out.get("judgments", [])
    relevant_count = sum(1 for j in judgments if j.get("relevant", False))
    total_count = len(chunks)
    return {
        "precision": relevant_count / total_count if total_count else 0.0,
        "relevant_count": relevant_count,
        "total_count": total_count,
        "judgments": judgments,
    }
