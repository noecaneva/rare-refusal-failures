"""Choose a judge attention path that is free of the SDPA constant-score artifact.

    python scripts/judge_method_check.py --config configs/experiment.yaml --shard 5

On the run's exact anomalous batches of one shard (plus normal batches), score with
'sdpa-math' (fused kernels disabled) and 'eager', in the original batches, and compare with the
reference: SDPA, batch size 1 (no padding, shown to match float32 within 1e-3). Reports the
number of constants, max/mean differences, label flips, and throughput. Writes
results/judge_method_check.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO_ROOT, dedupe, load_config, load_prompts, read_jsonl, shard_paths  # noqa: E402
from scripts_rescore_helper import load_judge_with  # noqa: E402

CONST = 0.5266


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", type=int, default=5)
    ap.add_argument("--n-batches", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config(args.config)
    jcfg = cfg["judge"]
    thr = float(jcfg["threshold"])
    bs = int(jcfg["batch_size"])
    prompts = {p["id"]: p["forbidden_prompt"] for p in load_prompts(cfg)}
    s = args.shard
    comps = dedupe(read_jsonl(shard_paths(cfg, "completions", s)[0]))
    scores = dedupe(read_jsonl(shard_paths(cfg, "scores", s)[0]))
    order = sorted(comps, key=lambda k: len(comps[k]["text"]))
    batches = [order[i:i + bs] for i in range(0, len(order), bs)]
    flags = [any(abs(float(scores[k]["score"]) - CONST) < 1e-3 for k in b) for b in batches]
    bad = [b for b, f in zip(batches, flags) if f][:args.n_batches]
    # also take normal batches from the high-score end so harmful items are represented
    high = [b for b, f in zip(batches, flags) if not f and any(float(scores[k]["score"]) >= thr for k in b)][-args.n_batches // 2:]
    test = bad + high
    keys = [k for b in test for k in b]
    import torch
    from judge import install_in_package_cache, score_batch

    results = {"shard": s, "n_batches": len(test), "n_items": len(keys)}

    def run(attn, batched):
        model, tok = load_judge_with(jcfg, attn)
        E = install_in_package_cache(model, tok)
        t0 = time.time()
        out = {}
        if batched:
            for b in test:
                for k, v in zip(b, score_batch(E, [prompts[k[0]] for k in b], [comps[k]["text"] for k in b])):
                    out[k] = float(v)
        else:
            for k in keys:
                out[k] = float(score_batch(E, [prompts[k[0]]], [comps[k]["text"]])[0])
        dt = time.time() - t0
        del model, E
        torch.cuda.empty_cache()
        return out, dt

    ref, dt_ref = run("sdpa", batched=False)
    results["reference_sdpa_bs1"] = {"items_per_second": round(len(keys) / dt_ref, 2), "n_const": sum(abs(v - CONST) < 1e-3 for v in ref.values())}
    for attn in ("sdpa-math", "eager", "sdpa"):
        out, dt = run(attn, batched=True)
        d = [abs(out[k] - ref[k]) for k in keys]
        results[attn + "_batched"] = {"items_per_second": round(len(keys) / dt, 2),
                                      "n_const": sum(abs(v - CONST) < 1e-3 for v in out.values()),
                                      "max_abs_diff_vs_ref": max(d), "mean_abs_diff_vs_ref": sum(d) / len(d),
                                      "n_diff_gt_0.01": sum(x > 0.01 for x in d), "n_diff_gt_0.05": sum(x > 0.05 for x in d),
                                      "label_flips_vs_ref": sum((out[k] >= thr) != (ref[k] >= thr) for k in keys),
                                      "n_harmful_ref": sum(ref[k] >= thr for k in keys), "n_harmful": sum(out[k] >= thr for k in keys)}
        print(attn, json.dumps(results[attn + "_batched"]), flush=True)
    (REPO_ROOT / "results" / "judge_method_check.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
