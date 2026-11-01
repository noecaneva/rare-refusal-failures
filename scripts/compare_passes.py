"""Compare two complete judging passes item by item (e.g. the archived SDPA pass against the eager
pass) and write results/judge_pass_comparison.json.

    python scripts/compare_passes.py --config configs/experiment.yaml --old scores_sdpa_v1 --new scores
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import REPO_ROOT, dedupe, load_config, read_jsonl, run_dir  # noqa: E402


def load_dir(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "shard_*.jsonl"))):
        rows += read_jsonl(f)
    return {k: float(r["score"]) for k, r in dedupe(rows).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--old", default="scores_sdpa_v1")
    ap.add_argument("--new", default="scores")
    ap.add_argument("--old-orig", action="store_true", help="use the .jsonl.orig files of --old (the uncorrected first pass)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    thr = float(cfg["judge"]["threshold"])
    base = run_dir(cfg)
    old_dir = base / args.old
    if args.old_orig:
        rows = []
        for f in sorted(glob.glob(str(old_dir / "shard_*.jsonl.orig"))) or sorted(glob.glob(str(old_dir / "shard_*.jsonl"))):
            rows += read_jsonl(f)
        old = {k: float(r["score"]) for k, r in dedupe(rows).items()}
    else:
        old = load_dir(str(old_dir))
    new = load_dir(str(base / args.new))
    keys = sorted(set(old) & set(new))
    a = np.array([old[k] for k in keys]); b = np.array([new[k] for k in keys])
    d = np.abs(a - b)
    ha, hb = a >= thr, b >= thr
    by_prompt_old, by_prompt_new = {}, {}
    for k, x, y in zip(keys, ha, hb):
        by_prompt_old[k[0]] = by_prompt_old.get(k[0], 0) + int(x)
        by_prompt_new[k[0]] = by_prompt_new.get(k[0], 0) + int(y)
    rep = {"old": args.old + (" (.orig)" if args.old_orig else ""), "new": args.new, "n_common": len(keys),
           "n_only_old": len(set(old) - set(new)), "n_only_new": len(set(new) - set(old)),
           "mean_abs_diff": float(d.mean()), "median_abs_diff": float(np.median(d)), "max_abs_diff": float(d.max()),
           "n_diff_gt_0.01": int((d > 0.01).sum()), "n_diff_gt_0.05": int((d > 0.05).sum()), "n_diff_gt_0.2": int((d > 0.2).sum()),
           "harmful_old": int(ha.sum()), "harmful_new": int(hb.sum()),
           "flips_to_harmful": int((~ha & hb).sum()), "flips_to_not_harmful": int((ha & ~hb).sum()),
           "prompts_zero_harmful_old": sum(v == 0 for v in by_prompt_old.values()),
           "prompts_zero_harmful_new": sum(v == 0 for v in by_prompt_new.values()),
           "prompts_changed_cracked_status": sum((by_prompt_old[p] == 0) != (by_prompt_new[p] == 0) for p in by_prompt_old),
           "n_old_at_constant": int((np.abs(a - 0.5266) < 1e-3).sum()), "n_new_at_constant": int((np.abs(b - 0.5266) < 1e-3).sum())}
    (REPO_ROOT / "results" / "judge_pass_comparison.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
