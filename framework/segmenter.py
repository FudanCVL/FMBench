"""Seed text segmentation: line-aware + heading/field-aware.

I/O conventions:
- Input: seed_text (str)
- Output: segments (list[dict]) where each segment has:
  - idx (int): monotonic index
  - type (str): 'heading_like_line' | 'field_line' | 'paragraph_sentence'
  - text (str): raw segment text
"""

from __future__ import annotations
import re
from typing import Dict, List, Any


ABBREV = {
    "e.g.", "i.e.", "vs.", "etc.", "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr."
}

FIELD_KEYS = {
    "abstract", "date", "time", "author", "authors", "keywords", "affiliation", "email"
}


def _uppercase_ratio(s: str) -> float:
    letters = [c for c in s if ("A" <= c <= "Z") or ("a" <= c <= "z")]
    if not letters:
        return 0.0
    upp = sum(1 for c in letters if "A" <= c <= "Z")
    return upp / len(letters)


def classify_line(line: str) -> str:
    l = line.strip()
    if not l:
        return "blank"

    m = re.match(r"^\s*([A-Za-z][A-Za-z\s]{0,30})\s*[:=-]\s*(.*)$", line)
    if m:
        key = m.group(1).strip().lower()
        rest = m.group(2).strip()
        if key in FIELD_KEYS:
            return "field_line"
        if rest == "" and key in FIELD_KEYS:
            return "field_line"

    if len(l) <= 80 and not re.search(r"[.!?]\s*$", l):
        if re.match(r"^(\d+(?:\.\d+)*)\s+\S+", l):
            return "heading_like_line"
        if re.match(r"^([IVXLC]+)\.?\s+\S+", l):
            return "heading_like_line"
        if re.match(r"^[-=]{2,}\s*\S.*\S\s*[-=]{2,}$", l):
            return "heading_like_line"
        if _uppercase_ratio(l) >= 0.6:
            return "heading_like_line"

    return "normal"


def split_sentences(paragraph: str) -> List[str]:
    masked = paragraph
    for a in ABBREV:
        masked = masked.replace(a, a.replace(".", "<DOT>"))

    parts = re.split(r"(?<=[.!?])\s+", masked.strip())
    out = []
    for p in parts:
        p = p.replace("<DOT>", ".").strip()
        if p:
            out.append(p)
    return out


def segment_seed_text(seed_text: str) -> List[Dict[str, Any]]:
    lines = seed_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    segments: List[Dict[str, Any]] = []
    idx = 0
    para_buf: List[str] = []

    def flush_paragraph():
        nonlocal idx, para_buf
        if not para_buf:
            return
        paragraph = " ".join([ln.strip() for ln in para_buf if ln.strip()])
        for sent in split_sentences(paragraph):
            segments.append({"idx": idx, "type": "paragraph_sentence", "text": sent})
            idx += 1
        para_buf = []

    for line in lines:
        t = classify_line(line)
        if t == "blank":
            flush_paragraph()
            continue
        if t in ("heading_like_line", "field_line"):
            flush_paragraph()
            segments.append({"idx": idx, "type": t, "text": line.strip()})
            idx += 1
            continue
        para_buf.append(line)

    flush_paragraph()
    return segments
