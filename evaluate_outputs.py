"""Evaluate model outputs with rule-based validator.

Expected input jsonl fields:
- seed_id, variant_id, spec, model_output
"""

from __future__ import annotations
import argparse
from collections import Counter
from framework.io import read_jsonl, load_yaml
from framework.validator import validate_markdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/v1.yaml")
    ap.add_argument("--predictions", required=True, help="predictions.jsonl")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    total = 0
    passed = 0
    ctr = Counter()

    for r in read_jsonl(args.predictions):
        total += 1
        res = validate_markdown(r.get("model_output", ""), r["spec"], cfg)
        if not res["hard_fail"]:
            passed += 1
        else:
            for reason in res["hard_fail_reasons"]:
                ctr[reason] += 1
    
    print(f"hard_pass_rate={passed/max(total,1):.4f} ({passed}/{total})")
    if total - passed:
        for k, v in ctr.most_common(20):
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
