"""Rule-driven Markdown target generation.

I/O conventions:
- Input: seed_text (str), spec (dict), cfg (dict)
- Output: target_text (str), meta (dict) including 'variant_family'
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, List
from .segmenter import segment_seed_text


def generate_markdown(seed_text: str, spec: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Generate Markdown target text deterministically.

    NOTE: This stub is for pipeline wiring. Replace with your full family-based generator.
    """
    segs = segment_seed_text(seed_text)

    lines: List[str] = []
    lines.append("# Document")
    lines.append("")
    lines.append("## Anchors")
    for s in segs:
        if s["type"] in ("heading_like_line", "field_line"):
            lines.append(f"- {s['text']}")
    lines.append("")
    lines.append("## Content")
    for s in segs:
        if s["type"] == "paragraph_sentence":
            lines.append(f"- {s['text']}")
    lines.append("")

    if spec.get("require_codeblock"):
        lines.append("```text")
        lines.append("placeholder")
        lines.append("```")
        lines.append("")

    if spec.get("require_table"):
        cols = int(spec.get("table", {}).get("cols", 3))
        header = "| " + " | ".join([f"H{i+1}" for i in range(cols)]) + " |"
        sep = "| " + " | ".join(["---" for _ in range(cols)]) + " |"
        lines.append(header)
        lines.append(sep)
        lines.append("| " + " | ".join(["X" for _ in range(cols)]) + " |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n", {"variant_family": "stub_minimal"}
