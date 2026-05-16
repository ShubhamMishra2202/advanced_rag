def format_context(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        header = f"{h.get('doc', '?')} | p.{h.get('page')} | para={h.get('paragraph')}"
        parts.append(f"[{i}] ({header})\n{h.get('text', '')}")
    return "\n\n".join(parts)
