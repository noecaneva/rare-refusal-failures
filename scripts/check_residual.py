"""Reference-score (batch size 1, no padding) the items whose eager re-score still equals the old
artifact value, to tell a genuine score from a residual artifact. Writes results/judge_residual_check.json."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO_ROOT, dedupe, load_config, load_prompts, n_shards, read_jsonl, shard_paths  # noqa: E402
from scripts_rescore_helper import load_judge_with  # noqa: E402

CONST = 0.5266


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-items", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config(args.config)
    thr = float(cfg["judge"]["threshold"])
    prompts = {p["id"]: p["forbidden_prompt"] for p in load_prompts(cfg)}
    items = []
    for s in range(n_shards(cfg)):
        new = dedupe(read_jsonl(shard_paths(cfg, "scores", s)[0]))
        keys = [k for k, r in new.items() if r.get("rescored") and abs(float(r["score"]) - CONST) < 1e-3]
        if not keys:
            continue
        comps = dedupe(read_jsonl(shard_paths(cfg, "completions", s)[0]))
        items += [(prompts[k[0]], comps[k]["text"], float(new[k]["score"])) for k in keys]
        if len(items) >= args.max_items:
            break
    items = items[:args.max_items]
    from judge import install_in_package_cache, score_batch

    out = {"n_items": len(items)}
    for attn in ("sdpa", "eager"):
        model, tok = load_judge_with(cfg["judge"], attn)
        E = install_in_package_cache(model, tok)
        ref = [score_batch(E, [p], [r])[0] for p, r, _ in items]
        d = [abs(a - b) for a, (_, _, b) in zip(ref, items)]
        out[f"{attn}_bs1"] = {"n_within_window": sum(abs(a - CONST) < 1e-3 for a in ref), "max_abs_diff_vs_eager_batched": max(d),
                              "mean_abs_diff": sum(d) / len(d), "n_harmful": sum(a >= thr for a in ref),
                              "examples": [round(a, 4) for a in ref[:12]]}
        print(attn, json.dumps(out[f"{attn}_bs1"]), flush=True)
        import torch
        del model, E
        torch.cuda.empty_cache()
    (REPO_ROOT / "results" / "judge_residual_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
