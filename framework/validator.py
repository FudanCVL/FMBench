"""Rule-based Markdown validator (v1).

I/O conventions:
- Input: markdown_text (str), spec (dict), cfg (dict)
- Output: result (dict):
  - hard_fail (bool)
  - hard_fail_reasons (list[str])
  - diagnostics (dict)
"""

from __future__ import annotations
import re
from typing import Any, Dict, List


def _has_unclosed_fence(md: str) -> bool:
    fences = re.findall(r"^```", md, flags=re.MULTILINE)
    return (len(fences) % 2) == 1


def _table_column_counts(md: str) -> List[int]:
    counts = []
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            parts = [p for p in s.split("|")[1:-1]]
            counts.append(len(parts))
    return counts


def validate_markdown(md: str, spec: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    diag: Dict[str, Any] = {}

    if _has_unclosed_fence(md):
        reasons.append("unclosed_fenced_code_block")

    col_counts = _table_column_counts(md)
    if col_counts and len(set(col_counts)) != 1:
        reasons.append("inconsistent_table_columns")
    diag["table_col_counts"] = col_counts

    heading_levels = []
    for line in md.splitlines():
        m = re.match(r"^(#{1,6})\s+\S+", line)
        if m:
            heading_levels.append(len(m.group(1)))
    for a, b in zip(heading_levels, heading_levels[1:]):
        if b - a > 1:
            reasons.append("illegal_heading_jump")
            break
    diag["heading_levels"] = heading_levels

    if spec.get("require_codeblock") and "```" not in md:
        reasons.append("spec_violation_missing_required_component:codeblock")
    if spec.get("require_table") and "|" not in md:
        reasons.append("spec_violation_missing_required_component:table")

    hard_fail = len(reasons) > 0
    diag["length_chars"] = len(md)
    return {"hard_fail": hard_fail, "hard_fail_reasons": reasons, "diagnostics": diag}
