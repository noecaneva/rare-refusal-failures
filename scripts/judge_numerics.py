"""How much do float16 judge scores depend on batching, and how far are they from float32?

    python scripts/judge_numerics.py --config configs/experiment.yaml --per-stratum 100 --cpu-per-stratum 30

Takes already-scored completions stratified by their stored score (low < 0.05, mid, high >= 0.75),
re-scores them on the GPU in float16 three ways (batch size 1 = no padding; length-sorted batches as
in the run; random batches) and a subset in float32 on the CPU (reference), and reports score
differences and label flips at the threshold. Writes results/judge_numerics.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import REPO_ROOT, dedupe, load_config, load_prompts, n_shards, read_jsonl, shard_paths  # noqa: E402


def score_in_batches(E, items, order, bs):
    """Score items (list of (prompt, response)) in the given order and batch size; return scores in item order."""
    from judge import score_batch

    out = [None] * len(items)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        s = score_batch(E, [items[j][0] for j in idx], [items[j][1] for j in idx])
        for j, v in zip(idx, s):
            out[j] = float(v)
    return out


def stats(a, b, thr):
    d = [abs(x - y) for x, y in zip(a, b)]
    return {"n": len(d), "max_abs_diff": max(d), "mean_abs_diff": sum(d) / len(d),
            "n_diff_gt_0.05": sum(x > 0.05 for x in d), "n_diff_gt_0.2": sum(x > 0.2 for x in d),
            "label_flips": sum((x >= thr) != (y >= thr) for x, y in zip(a, b))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--per-stratum", type=int, default=100)
    ap.add_argument("--cpu-per-stratum", type=int, default=30)
    ap.add_argument("--n-shards", type=int, default=6)
    args = ap.parse_args()
    cfg = load_config(args.config)
    jcfg = cfg["judge"]
    thr = float(jcfg["threshold"])
    prompts = {p["id"]: p["forbidden_prompt"] for p in load_prompts(cfg)}
    rnd = random.Random(1)
    pool = {"low": [], "mid": [], "high": []}
    for s in rnd.sample(range(n_shards(cfg)), args.n_shards):
        comps = dedupe(read_jsonl(shard_paths(cfg, "completions", s)[0]))
        for k, r in dedupe(read_jsonl(shard_paths(cfg, "scores", s)[0])).items():
            v = float(r["score"])
            pool["low" if v < 0.05 else "high" if v >= thr else "mid"].append((prompts[k[0]], comps[k]["text"], v))
    items = []
    for name in ("low", "mid", "high"):
        items += rnd.sample(pool[name], min(args.per_stratum, len(pool[name])))
    stored = [v for _, _, v in items]
    pairs = [(p, r) for p, r, _ in items]
    report = {"n_items": len(items), "strata": {k: len(v) for k, v in pool.items()}, "threshold": thr}

    import torch
    from judge import LastTokenLogits, install_in_package_cache, load_judge

    model, tok = load_judge(jcfg)
    E = install_in_package_cache(model, tok)
    bs = int(jcfg["batch_size"])
    n = len(items)
    t0 = time.time()
    fp16_bs1 = score_in_batches(E, pairs, list(range(n)), 1)
    report["seconds_fp16_bs1"] = round(time.time() - t0, 1)
    sorted_order = sorted(range(n), key=lambda j: len(pairs[j][1]))
    fp16_sorted = score_in_batches(E, pairs, sorted_order, bs)
    rand_order = list(range(n))
    rnd.shuffle(rand_order)
    fp16_random = score_in_batches(E, pairs, rand_order, bs)
    fp16_random2 = score_in_batches(E, pairs, rand_order[::-1], bs)

    # float32 reference on the CPU for a subset of each stratum
    cpu_idx = []
    per = args.cpu_per_stratum
    for st in range(3):
        cpu_idx += list(range(st * args.per_stratum, min((st + 1) * args.per_stratum, n)))[:per]
    model.to("cpu", dtype=torch.float32)
    E.cached_models["strongreject_finetuned"] = (LastTokenLogits(model), tok)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    t0 = time.time()
    fp32 = score_in_batches(E, [pairs[j] for j in cpu_idx], list(range(len(cpu_idx))), 1)
    report["seconds_fp32_cpu"] = round(time.time() - t0, 1)

    sub = lambda a: [a[j] for j in cpu_idx]  # noqa: E731
    report["comparisons"] = {
        "stored_run_vs_fp16_bs1": stats(stored, fp16_bs1, thr),
        "stored_run_vs_fp16_sorted_batches": stats(stored, fp16_sorted, thr),
        "fp16_bs1_vs_fp16_sorted_batches": stats(fp16_bs1, fp16_sorted, thr),
        "fp16_bs1_vs_fp16_random_batches": stats(fp16_bs1, fp16_random, thr),
        "fp16_random_vs_fp16_random_reversed": stats(fp16_random, fp16_random2, thr),
        "fp32_cpu_vs_fp16_bs1": stats(fp32, sub(fp16_bs1), thr),
        "fp32_cpu_vs_fp16_sorted_batches": stats(fp32, sub(fp16_sorted), thr),
        "fp32_cpu_vs_fp16_random_batches": stats(fp32, sub(fp16_random), thr),
        "fp32_cpu_vs_stored_run": stats(fp32, sub(stored), thr),
    }
    report["per_item_subset"] = [{"stored": round(stored[j], 4), "fp32": round(fp32[i], 4), "fp16_bs1": round(fp16_bs1[j], 4),
                                  "fp16_sorted": round(fp16_sorted[j], 4), "fp16_random": round(fp16_random[j], 4)}
                                 for i, j in enumerate(cpu_idx)]
    worst = sorted(range(n), key=lambda j: -abs(fp16_random[j] - fp16_bs1[j]))[:10]
    report["worst_random_batch_items"] = [{"stored": round(stored[j], 4), "fp16_bs1": round(fp16_bs1[j], 4),
                                           "fp16_sorted": round(fp16_sorted[j], 4), "fp16_random": round(fp16_random[j], 4),
                                           "response_chars": len(pairs[j][1])} for j in worst]
    out = REPO_ROOT / "results" / "judge_numerics.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_item_subset",)}, indent=1))


if __name__ == "__main__":
    main()
