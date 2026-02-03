#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_variants.py

Input (seeds jsonl):
  {"id": "seed_0001", "type": "academic_paper", "seed_text": "..."}

Output (variants jsonl), one line per variant:
  {
    "seed_id": "...",
    "variant_id": "...",
    "format": "markdown",
    "difficulty_level": "L1|L2|L3",
    "variant_family": "mixed|headings_sections|nested_lists|tables|blockquotes|codeblocks|wrapping_policy",
    "spec": {...},
    "prompt": "...",
    "target_text": "...",
    "validator_spec": {"name":"markdown_ruleset_v1","version":"v1"}
  }

Notes:
- Fully deterministic generation given (seed_id, variant_id) and fixed config.
- Rule-driven pseudo-structure segmentation: heading-like lines and field lines are treated as segments.
- No semantic rephrase is performed (v1 target_text keeps original text fragments).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import re
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ----------------------------
# Config (v1, Markdown-only)
# ----------------------------

V1_KNOBS = {
    "heading_depth_max": [1, 2],
    "section_count": [2, 5],
    "require_nested_list": [True, False],
    "nested_list_depth": [1, 3],
    "list_items": [3, 10],
    "require_table": [True, False],
    "table_rows": [3, 6],
    "table_cols": [3, 5],
    "allow_long_cell": [True, False],
    "require_blockquote": [True, False],
    "blockquote_count": [1, 3],
    "require_codeblock": [True, False],
    "codeblock_count": [1, 1],
    "wrap_mode": ["none", "hard_wrap_80"],
}

DIFFICULTY_PRESETS = {
    "L1": dict(
        heading_depth_max=1,
        section_count_range=(2, 3),
        require_nested_list=False,
        require_table=False,
        require_blockquote=False,
        require_codeblock=False,
        wrap_mode="none",
    ),
    "L2": dict(
        heading_depth_max=2,
        section_count_range=(3, 4),
        require_nested_list=True,
        nested_list_depth_range=(1, 2),
        list_items_range=(3, 8),
        require_table=False,
        require_blockquote=True,
        blockquote_count_range=(1, 2),
        require_codeblock=False,
        wrap_mode="none",
    ),
    "L3": dict(
        heading_depth_max=2,
        section_count_range=(4, 5),
        require_nested_list=True,
        nested_list_depth_range=(2, 3),
        list_items_range=(6, 10),
        require_table=True,
        table_rows_range=(3, 5),
        table_cols_range=(3, 4),
        allow_long_cell=True,
        require_blockquote=True,
        blockquote_count_range=(2, 3),
        require_codeblock=True,
        codeblock_count=1,
        wrap_mode="hard_wrap_80",
    ),
}

FIELD_KEYS = {
    "abstract", "date", "time", "author", "authors", "keywords",
    "affiliation", "email"
}


# ----------------------------
# Utilities
# ----------------------------

def stable_rng(seed_id: str, variant_id: str) -> random.Random:
    h = hashlib.sha256(f"{seed_id}::{variant_id}".encode("utf-8")).hexdigest()
    # Use first 16 hex chars -> 64-bit int
    return random.Random(int(h[:16], 16))


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_json(path: str, row: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False)

def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def uppercase_ratio(s: str) -> float:
    letters = [c for c in s if "A" <= c <= "Z" or "a" <= c <= "z"]
    if not letters:
        return 0.0
    uppers = [c for c in letters if "A" <= c <= "Z"]
    return len(uppers) / len(letters)


# ----------------------------
# Segmentation (line-aware)
# ----------------------------

@dataclass(frozen=True)
class Segment:
    seg_type: str  # heading_like_line | field_line | paragraph_sentence
    text: str


RE_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+\S+")
RE_ROMAN_HEADING = re.compile(r"^\s*([IVXLC]+)\.?\s+\S+", re.IGNORECASE)
RE_DELIMITED_HEADING = re.compile(r"^\s*[-=]{2,}\s*\S.*\S\s*[-=]{2,}\s*$")
RE_KEY_ONLY = re.compile(r"^\s*([A-Za-z][A-Za-z\s]{0,30})\s*[:=-]\s*$")
RE_KEY_VALUE = re.compile(r"^\s*([A-Za-z][A-Za-z\s]{0,30})\s*[:=-]\s+\S+.*$")


def is_heading_like_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) > 80:
        return False
    if s.endswith((".", "!", "?")):
        # heuristic: headings often do not end with sentence punctuation
        pass
    # Accept several patterns
    if RE_NUMBERED_HEADING.match(s):
        return True
    if RE_ROMAN_HEADING.match(s):
        return True
    if RE_DELIMITED_HEADING.match(s):
        return True
    # High uppercase ratio (when letters exist)
    if uppercase_ratio(s) >= 0.6 and len(s.split()) <= 12:
        return True
    # Short non-sentence-like lines
    if not s.endswith((".", "!", "?")) and len(s.split()) <= 10:
        return True
    return False


def is_field_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    m = RE_KEY_ONLY.match(s) or RE_KEY_VALUE.match(s)
    if not m:
        return False
    key = m.group(1).strip().lower()
    return key in FIELD_KEYS


ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "vs.", "mr.", "mrs.", "ms.", "dr.", "prof.",
    "fig.", "eq.", "al.", "jan.", "feb.", "mar.", "apr.", "jun.", "jul.",
    "aug.", "sep.", "sept.", "oct.", "nov.", "dec."
}


def split_sentences(paragraph: str) -> List[str]:
    """
    Rule-based sentence splitter (v2-ish).
    - Protect common abbreviations and decimals.
    - Split on .!? followed by whitespace and a capital/quote.
    This is intentionally conservative to stabilize "no split/merge" evaluation.
    """
    text = " ".join(paragraph.strip().split())
    if not text:
        return []

    # Protect abbreviations by temporarily replacing periods
    protected = text
    for abbr in ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))

    # Protect decimals like 3.14
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)

    # Split heuristic
    parts = re.split(r"(?<=[.!?])\s+(?=(\"|\'|\(|\[)?[A-Z])", protected)
    # re.split with capturing groups yields extra entries; rebuild cleanly
    sentences: List[str] = []
    buf = ""
    for p in parts:
        if p is None:
            continue
        if p in ['"', "'", "(", "[", ""]:
            # capturing group fragments; ignore
            continue
        if not p:
            continue
        # When re.split yields fragments, we treat each as a sentence candidate
        cand = p.strip()
        if cand:
            sentences.append(cand)

    if not sentences:
        sentences = [protected.strip()]

    # Restore protected dots
    restored = []
    for s in sentences:
        s = s.replace("<DOT>", ".")
        restored.append(s.strip())
    return [s for s in restored if s]


def segment_seed_text(seed_text: str) -> List[Segment]:
    text = seed_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    # Classify lines and aggregate paragraphs
    segments: List[Segment] = []
    paragraph_lines: List[str] = []

    def flush_paragraph():
        nonlocal paragraph_lines, segments
        if not paragraph_lines:
            return
        para = " ".join([ln.strip() for ln in paragraph_lines if ln.strip()])
        paragraph_lines = []
        if not para.strip():
            return
        for sent in split_sentences(para):
            segments.append(Segment("paragraph_sentence", sent))

    for ln in lines:
        if ln.strip() == "":
            flush_paragraph()
            continue

        if is_field_line(ln):
            flush_paragraph()
            segments.append(Segment("field_line", ln.strip()))
            continue

        if is_heading_like_line(ln):
            flush_paragraph()
            segments.append(Segment("heading_like_line", ln.strip()))
            continue

        paragraph_lines.append(ln)

    flush_paragraph()
    # Fallback: if everything got eaten, treat entire text as one paragraph sentence
    if not segments:
        for sent in split_sentences(text):
            segments.append(Segment("paragraph_sentence", sent))
    return segments


# ----------------------------
# Spec sampling + prompt rendering
# ----------------------------

def sample_spec(rng: random.Random, difficulty: str) -> Dict[str, Any]:
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    preset = DIFFICULTY_PRESETS[difficulty]
    spec: Dict[str, Any] = {"format": "markdown"}

    # headings/sections
    spec["heading_depth_max"] = preset.get(
        "heading_depth_max",
        rng.choice(V1_KNOBS["heading_depth_max"])
    )
    sc_lo, sc_hi = preset.get("section_count_range", (V1_KNOBS["section_count"][0], V1_KNOBS["section_count"][1]))
    spec["section_count"] = rng.randint(sc_lo, sc_hi)

    # nested list
    spec["require_nested_list"] = preset.get("require_nested_list", rng.choice(V1_KNOBS["require_nested_list"]))
    if spec["require_nested_list"]:
        nd_lo, nd_hi = preset.get("nested_list_depth_range", (V1_KNOBS["nested_list_depth"][0], V1_KNOBS["nested_list_depth"][1]))
        li_lo, li_hi = preset.get("list_items_range", (V1_KNOBS["list_items"][0], V1_KNOBS["list_items"][1]))
        spec["nested_list_depth"] = rng.randint(nd_lo, nd_hi)
        spec["list_items"] = rng.randint(li_lo, li_hi)
    else:
        spec["nested_list_depth"] = 0
        spec["list_items"] = 0

    # table
    spec["require_table"] = preset.get("require_table", rng.choice(V1_KNOBS["require_table"]))
    if spec["require_table"]:
        tr_lo, tr_hi = preset.get("table_rows_range", (V1_KNOBS["table_rows"][0], V1_KNOBS["table_rows"][1]))
        tc_lo, tc_hi = preset.get("table_cols_range", (V1_KNOBS["table_cols"][0], V1_KNOBS["table_cols"][1]))
        spec["table"] = {
            "rows": rng.randint(tr_lo, tr_hi),
            "cols": rng.randint(tc_lo, tc_hi),
            "allow_long_cell": preset.get("allow_long_cell", rng.choice(V1_KNOBS["allow_long_cell"])),
        }
    else:
        spec["table"] = None

    # blockquote
    spec["require_blockquote"] = preset.get("require_blockquote", rng.choice(V1_KNOBS["require_blockquote"]))
    if spec["require_blockquote"]:
        bq_lo, bq_hi = preset.get("blockquote_count_range", (V1_KNOBS["blockquote_count"][0], V1_KNOBS["blockquote_count"][1]))
        spec["blockquote_count"] = rng.randint(bq_lo, bq_hi)
    else:
        spec["blockquote_count"] = 0

    # codeblock
    spec["require_codeblock"] = preset.get("require_codeblock", rng.choice(V1_KNOBS["require_codeblock"]))
    if spec["require_codeblock"]:
        spec["codeblock_count"] = preset.get("codeblock_count", 1)
    else:
        spec["codeblock_count"] = 0

    # wrapping
    spec["wrap_mode"] = preset.get("wrap_mode", rng.choice(V1_KNOBS["wrap_mode"]))

    # constraints (v1)
    spec["ordering_policy"] = "no_reorder"
    spec["content_policy"] = "semantic_preserve"
    spec["forbid_extra_text_outside_structure"] = True
    spec["forbid_merge_split"] = True

    return spec


def render_prompt(spec: Dict[str, Any]) -> str:
    # Deterministic canonical prompt. Keep stable across versions.
    lines = []
    lines.append("You are given a seed text. Produce a Markdown document that follows the specification below.")
    lines.append("")
    lines.append("Hard constraints:")
    lines.append("- Output Markdown only.")
    lines.append("- Preserve the order of content units (no reordering).")
    lines.append("- Do not merge or split content units.")
    lines.append("- You may rephrase units while keeping meaning.")
    lines.append("- Do not add new information or omit information.")
    lines.append("- Do not output any extra commentary outside the Markdown document.")
    lines.append("")
    lines.append("Specification:")
    lines.append(f"- Max heading depth: {spec['heading_depth_max']}")
    lines.append(f"- Number of sections: {spec['section_count']}")
    lines.append(f"- Require nested list: {spec['require_nested_list']}")
    if spec["require_nested_list"]:
        lines.append(f"  - Nested depth: {spec['nested_list_depth']}")
        lines.append(f"  - List items: {spec['list_items']}")
    lines.append(f"- Require table: {spec['require_table']}")
    if spec["require_table"] and spec["table"] is not None:
        lines.append(f"  - Table rows: {spec['table']['rows']}")
        lines.append(f"  - Table cols: {spec['table']['cols']}")
        lines.append(f"  - Allow long cell: {spec['table']['allow_long_cell']}")
    lines.append(f"- Require blockquote: {spec['require_blockquote']}")
    if spec["require_blockquote"]:
        lines.append(f"  - Blockquotes: {spec['blockquote_count']}")
    lines.append(f"- Require code block: {spec['require_codeblock']}")
    if spec["require_codeblock"]:
        lines.append(f"  - Code blocks: {spec['codeblock_count']}")
    lines.append(f"- Wrap mode: {spec['wrap_mode']}")
    return "\n".join(lines).strip() + "\n"


# ----------------------------
# Markdown generation helpers
# ----------------------------

def md_escape_inline(text: str) -> str:
    # Minimal escaping; keep v1 conservative.
    # Escape pipes because we may use tables.
    return text.replace("|", r"\|")


def make_heading(level: int, title: str) -> str:
    lvl = clamp_int(level, 1, 6)
    title = title.strip()
    if not title:
        title = "Section"
    return f"{'#' * lvl} {title}"


def make_blockquote(lines: List[str]) -> str:
    return "\n".join(["> " + ln for ln in lines])


def make_codeblock(content: str, lang: str = "text") -> str:
    # Ensure no triple backticks inside; replace deterministically
    safe = content.replace("```", "`` `")
    return f"```{lang}\n{safe}\n```"


def make_pipe_table(cells: List[str], rows: int, cols: int) -> str:
    # cells length must be rows*cols
    assert len(cells) == rows * cols
    # header row: first row
    header = cells[:cols]
    body = [cells[i:i+cols] for i in range(cols, rows*cols, cols)]

    def row_line(row: List[str]) -> str:
        return "| " + " | ".join(row) + " |"

    out = []
    out.append(row_line(header))
    out.append("| " + " | ".join(["---"] * cols) + " |")
    for r in body:
        out.append(row_line(r))
    return "\n".join(out)


def hard_wrap(text: str, width: int = 80) -> str:
    # Wrap only paragraphs, keep Markdown structural lines intact where possible.
    out_lines = []
    for ln in text.split("\n"):
        stripped = ln.strip()
        if not stripped:
            out_lines.append(ln)
            continue
        # Do not wrap headings, list markers, blockquotes, table lines, code fences
        if stripped.startswith("#") or stripped.startswith(("-", "*", "+", "1.")) or stripped.startswith(">") or stripped.startswith("|") or stripped.startswith("```"):
            out_lines.append(ln)
            continue
        out_lines.extend(textwrap.wrap(ln, width=width, break_long_words=False, break_on_hyphens=False))
    return "\n".join(out_lines)


# ----------------------------
# Variant construction
# ----------------------------

def choose_title_from_segments(segments: List[Segment], fallback: str) -> str:
    for seg in segments:
        if seg.seg_type == "heading_like_line" and seg.text.strip():
            return seg.text.strip()
    return fallback


def field_line_to_kv(line: str) -> Tuple[str, str]:
    s = line.strip()
    m = RE_KEY_VALUE.match(s)
    if m:
        key = m.group(1).strip()
        rest = s[m.end(1):].strip()
        # strip delimiters
        rest = re.sub(r"^[:=-]\s*", "", rest)
        return key, rest
    m2 = RE_KEY_ONLY.match(s)
    if m2:
        key = m2.group(1).strip()
        return key, ""
    # fallback
    return s, ""


def build_markdown_from_spec(segments: List[Segment], spec: Dict[str, Any], rng: random.Random) -> Tuple[str, str]:
    """
    Returns (variant_family, markdown_text).
    variant_family: "mixed" for combined structures.
    """
    # Extract ordered content pool from paragraph sentences (and optionally field values)
    # We preserve segment order by consuming from a queue.
    pool: List[str] = []
    field_segments: List[Segment] = []
    heading_segments: List[Segment] = []
    para_segments: List[Segment] = []

    for seg in segments:
        if seg.seg_type == "field_line":
            field_segments.append(seg)
        elif seg.seg_type == "heading_like_line":
            heading_segments.append(seg)
        else:
            para_segments.append(seg)

    # Content pool uses paragraph sentences in order (escaped)
    pool = [md_escape_inline(s.text) for s in para_segments]

    # Build sections
    section_count = int(spec["section_count"])
    max_depth = int(spec["heading_depth_max"])
    top_level = 1
    sub_level = min(top_level + 1, max_depth)

    doc_lines: List[str] = []

    # Document title (H1)
    title = choose_title_from_segments(segments, fallback="Document")
    doc_lines.append(make_heading(top_level, md_escape_inline(title)))
    doc_lines.append("")

    # Field lines as a fixed "Metadata" section if present
    if field_segments:
        doc_lines.append(make_heading(sub_level, "Metadata"))
        doc_lines.append("")
        for fs in field_segments:
            key, val = field_line_to_kv(fs.text)
            key = key.strip()
            val = val.strip()
            if val:
                doc_lines.append(f"**{md_escape_inline(key)}:** {md_escape_inline(val)}")
            else:
                doc_lines.append(f"**{md_escape_inline(key)}:**")
        doc_lines.append("")

    # Allocate content to sections sequentially
    # Split pool into roughly equal chunks for sections
    chunks: List[List[str]] = [[] for _ in range(section_count)]
    for i, sent in enumerate(pool):
        chunks[i % section_count].append(sent)

    # Structures to insert (in order) - keep deterministic ordering of component insertion
    require_list = bool(spec["require_nested_list"])
    require_table = bool(spec["require_table"])
    require_bq = bool(spec["require_blockquote"])
    require_code = bool(spec["require_codeblock"])

    variant_family = "mixed"

    # Helper: consume k sentences from a mutable list while preserving order
    def consume(src: List[str], k: int) -> List[str]:
        k = max(0, min(k, len(src)))
        taken = src[:k]
        del src[:k]
        return taken

    # For each section, add heading and then content + optional structures
    for si in range(section_count):
        doc_lines.append(make_heading(sub_level, f"Section {si+1}"))
        doc_lines.append("")

        section_pool = list(chunks[si])  # local mutable
        # Insert nested list (once overall, placed in earliest section with enough content)
        if require_list:
            depth = int(spec.get("nested_list_depth", 2))
            items = int(spec.get("list_items", 6))
            if len(section_pool) >= min(items, 3):
                list_sents = consume(section_pool, items)
                doc_lines.extend(make_nested_list(list_sents, depth=depth))
                doc_lines.append("")
                require_list = False  # placed once

        # Insert table (once overall)
        if require_table and spec.get("table"):
            rows = int(spec["table"]["rows"])
            cols = int(spec["table"]["cols"])
            need = rows * cols
            if len(section_pool) >= min(need, cols * 2):  # require at least some content
                cell_sents = consume(section_pool, min(need, len(section_pool)))
                # If not enough, pad deterministically using last element
                if len(cell_sents) < need:
                    pad = cell_sents[-1] if cell_sents else "N/A"
                    cell_sents.extend([pad] * (need - len(cell_sents)))
                # Optionally make a long cell by concatenating
                if spec["table"]["allow_long_cell"] and need >= 4:
                    cell_sents[0] = (cell_sents[0] + " " + cell_sents[1]).strip()
                doc_lines.append(make_pipe_table(cell_sents[:need], rows=rows, cols=cols))
                doc_lines.append("")
                require_table = False

        # Insert blockquote(s) (once overall)
        if require_bq:
            count = int(spec.get("blockquote_count", 1))
            take = min(len(section_pool), max(1, count))
            if take > 0:
                bq_sents = consume(section_pool, take)
                doc_lines.append(make_blockquote(bq_sents))
                doc_lines.append("")
                require_bq = False

        # Insert codeblock (once overall)
        if require_code:
            count = int(spec.get("codeblock_count", 1))
            # take a few lines; ensure codeblock exists even if section_pool is short
            take = min(len(section_pool), max(2, 4))
            code_lines = consume(section_pool, take)
            if not code_lines:
                code_lines = ["(empty)"]
            doc_lines.append(make_codeblock("\n".join(code_lines), lang="text"))
            doc_lines.append("")
            require_code = False

        # Remaining sentences as paragraphs
        for sent in section_pool:
            doc_lines.append(sent)
            doc_lines.append("")

    md = "\n".join(doc_lines).rstrip() + "\n"

    # Apply wrapping if requested
    if spec.get("wrap_mode") == "hard_wrap_80":
        md = hard_wrap(md, width=80).rstrip() + "\n"

    return variant_family, md


def make_nested_list(items: List[str], depth: int = 2, indent: int = 2) -> List[str]:
    """
    Deterministic nested list.
    depth: 1..N. Items are distributed across depths in a simple pattern.
    """
    depth = clamp_int(depth, 1, 6)
    out: List[str] = []
    for i, it in enumerate(items):
        d = 1 + (i % depth)
        prefix = " " * ((d - 1) * indent) + "- "
        out.append(prefix + it)
    return out


# ----------------------------
# Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate rule-driven Markdown variants from seed jsonl.")
    p.add_argument("--input", required=True, help="Path to seeds.jsonl")
    p.add_argument("--output", required=True, help="Path to variants.jsonl")
    p.add_argument("--variants-per-seed", type=int, default=6, help="Number of variants to generate per seed (default: 6)")
    p.add_argument("--difficulty", choices=["L1", "L2", "L3"], default="L2", help="Difficulty preset (default: L2)")
    p.add_argument("--start-idx", type=int, default=0, help="Start index for variant numbering (default: 0)")
    p.add_argument("--variant-prefix", default="v1", help="Variant id prefix (default: v1)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    rows_out: List[Dict[str, Any]] = []
    vps = max(1, args.variants_per_seed)

    for seed_id, seed in enumerate(read_jsonl(args.input)):
        # print(f"Seed {seed_id}")
        # input()
        # seed_id = str(seed.get("id", "")).strip()
        # if not seed_id:
        #     raise ValueError("Seed record missing non-empty 'id'")
        seed_text = seed.get("seed_text", "")
        if not isinstance(seed_text, str) or not seed_text.strip():
            raise ValueError(f"Seed {seed_id} missing non-empty 'seed_text'")

        segments = segment_seed_text(seed_text)

        for j in range(args.start_idx, args.start_idx + vps):
            variant_id = f"{args.variant_prefix}_{j:04d}"
            rng = stable_rng(seed_id, variant_id)

            spec = sample_spec(rng, difficulty=args.difficulty)
            prompt = render_prompt(spec)
            variant_family, target_text = build_markdown_from_spec(segments, spec, rng)

            rows_out.append({
                "seed_id": seed_id,
                "variant_id": variant_id,
                "format": "markdown",
                "difficulty_level": args.difficulty,
                "variant_family": variant_family,
                "spec": spec,
                "prompt": prompt,
                "seed_text": seed_text,
                "target_text": target_text,
                "validator_spec": {"name": "markdown_ruleset_v1", "version": "v1"},
            })

    write_json(args.output, rows_out)
    print(f"Wrote {len(rows_out)} variants to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
