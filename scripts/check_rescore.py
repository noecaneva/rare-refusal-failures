"""After rescore_anomalous.py: confirm the artifact is gone and summarise what changed.

    python scripts/check_rescore.py --config configs/experiment.yaml
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import REPO_ROOT, dedupe, load_config, n_shards, read_jsonl, run_dir, shard_paths  # noqa: E402

CONST = 0.5266


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    thr = float(cfg["judge"]["threshold"])
    exact_consts = collections.Counter()
    tot = {"n": 0, "rescored": 0, "within_window_after": 0, "exact_same_value_after": 0, "newly_harmful": 0,
           "lost_harmful": 0, "harmful_before": 0, "harmful_after": 0, "shards_missing_rescore": []}
    verify = []
    for s in range(n_shards(cfg)):
        path, _, _ = shard_paths(cfg, "scores", s)
        meta_p = path.with_suffix(".rescore.json")
        if not meta_p.exists():
            tot["shards_missing_rescore"].append(s)
            continue
        verify.append(json.loads(meta_p.read_text()).get("verify", {}))
        new = dedupe(read_jsonl(path))
        orig_p = path.with_suffix(".jsonl.orig")
        old = dedupe(read_jsonl(orig_p)) if orig_p.exists() else new
        for k, r in new.items():
            v, o = float(r["score"]), float(old[k]["score"])
            tot["n"] += 1
            tot["harmful_before"] += o >= thr
            tot["harmful_after"] += v >= thr
            if abs(o - CONST) < 1e-3:
                exact_consts[o] += 1
            if r.get("rescored"):
                tot["rescored"] += 1
                if abs(v - CONST) < 1e-3:
                    tot["within_window_after"] += 1
                    if abs(v - o) < 1e-6:
                        tot["exact_same_value_after"] += 1
                tot["newly_harmful"] += (v >= thr) and not (o >= thr)
                tot["lost_harmful"] += (o >= thr) and not (v >= thr)
    tot["original_constant_values_top3"] = exact_consts.most_common(3)
    tot["verify_max_abs_diff_over_shards"] = max(v.get("max_abs_diff", 0) for v in verify) if verify else None
    tot["verify_label_flips_total"] = sum(v.get("label_flips", 0) for v in verify)
    tot["verify_n_total"] = sum(v.get("n", 0) for v in verify)
    (REPO_ROOT / "results" / "judge_rescore_summary.json").write_text(json.dumps(tot, indent=1))
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    main()
