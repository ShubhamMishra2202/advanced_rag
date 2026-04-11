"""Minimal 'answer' from retrieved chunks (no external LLM)."""


def format_context(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        src = h.get("source") or "?"
        lines.append(f"[{i}] ({src})\n{h.get('text', '')}")
    return "\n\n".join(lines)
