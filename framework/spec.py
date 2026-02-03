"""Spec sampling and canonical prompt rendering (deterministic).

I/O conventions:
- Input: seed_id (str), variant_id (str), config (dict loaded from config/v1.yaml)
- Output: spec (dict), prompt (str)
"""

from __future__ import annotations
import hashlib
import json
from typing import Any, Dict, Tuple


def deterministic_rng_seed(seed_id: str, variant_id: str) -> int:
    h = hashlib.sha256(f"{seed_id}::{variant_id}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def sample_spec(seed_id: str, variant_id: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministically sample a spec from cfg.

    NOTE: This is a stub. Implement sampling from cfg['spec_language']['knobs']
    and optionally cfg['difficulty']['levels'][...].
    """
    return {
        "format": "markdown",
        "heading_depth_max": 2,
        "section_count": 3,
        "require_nested_list": True,
        "nested_list_depth": 2,
        "require_table": True,
        "table": {"rows": 4, "cols": 3, "allow_long_cell": True},
        "require_blockquote": False,
        "blockquote_count": 1,
        "require_codeblock": True,
        "codeblock_count": 1,
        "wrap_mode": "none",
    }


def render_prompt(spec: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """Deterministically render a natural-language prompt from spec.

    Must not add constraints not present in spec/cfg.
    """
    req = cfg.get("spec_language", {}).get("canonical_prompt_template", {}).get("requirements", [])
    struct = cfg.get("spec_language", {}).get("canonical_prompt_template", {}).get("structure_instructions", [])
    parts = []
    if req:
        parts.append("Requirements:")
        parts += [f"- {r}" for r in req]
    if struct:
        parts.append("")
        parts.append("Structure:")
        parts += [f"- {r}" for r in struct]

    parts.append("")
    parts.append("Spec (JSON):")
    parts.append(json.dumps(spec, ensure_ascii=False, indent=2))
    return "\n".join(parts)


def build_spec_and_prompt(seed_id: str, variant_id: str, cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    spec = sample_spec(seed_id, variant_id, cfg)
    prompt = render_prompt(spec, cfg)
    return spec, prompt
