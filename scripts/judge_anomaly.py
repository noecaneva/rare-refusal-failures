"""Diagnose the constant-score artifact seen in length-sorted judge batches.

    python scripts/judge_anomaly.py --config configs/experiment.yaml

A. Scan every scores shard: how many stored scores equal the suspicious constant (to 1e-3), and
   which other values occur far too often to be genuine.
B. Reproduce: take anomalous items, score them (1) one at a time, (2) in length-sorted batches with
   SDPA attention, (3) the same with eager attention, (4) a batch of 32 identical refusals, and
   (5) the sorted batch with one deliberately longer item so that the batch is padded.
Writes results/judge_anomaly.json.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import REPO_ROOT, dedupe, load_config, load_prompts, read_jsonl, run_dir, shard_paths  # noqa: E402

CONST = 0.5266


def scan(cfg):
    counts = collections.Counter()
    per_shard, per_prompt, anomalous = {}, collections.Counter(), []
    for f in sorted(glob.glob(str(run_dir(cfg) / "scores" / "shard_*.jsonl"))):
        shard = int(os.path.basename(f).split("_")[1].split(".")[0])
        n_bad = 0
        for r in read_jsonl(f):
            v = round(float(r["score"]), 3)
            counts[v] += 1
            if abs(float(r["score"]) - CONST) < 1e-3:
                n_bad += 1
                per_prompt[int(r["prompt_id"])] += 1
                anomalous.append((shard, int(r["prompt_id"]), int(r["sample_idx"]), float(r["score"])))
        per_shard[shard] = n_bad
    total = sum(counts.values())
    spikes = [(v, c) for v, c in counts.most_common(15)]
    return {"n_scores": total, "n_anomalous": len(anomalous), "frac_anomalous": len(anomalous) / total,
            "per_shard": per_shard, "n_prompts_affected": len(per_prompt),
            "top_prompts": per_prompt.most_common(10), "most_common_rounded_scores": spikes}, anomalous


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    jcfg = cfg["judge"]
    report = {}
    report["scan"], anomalous = scan(cfg)
    print(json.dumps(report["scan"], indent=1), flush=True)

    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from judge import LastTokenLogits, install_in_package_cache, load_judge, score_batch

    prompts = {p["id"]: p["forbidden_prompt"] for p in load_prompts(cfg)}
    # 64 anomalous items from the first affected shard(s)
    picked = anomalous[:64]
    by_shard = collections.defaultdict(list)
    for s, pid, idx, _ in picked:
        by_shard[s].append((pid, idx))
    items = []
    for s, keys in by_shard.items():
        comps = dedupe(read_jsonl(shard_paths(cfg, "completions", s)[0]))
        for k in keys:
            items.append((prompts[k[0]], comps[k]["text"]))
    report["n_items"] = len(items)
    report["example_responses"] = sorted(collections.Counter(r for _, r in items).most_common(3))

    def run_variants(model, tok, label):
        E = install_in_package_cache(model, tok)
        out = {}
        out["bs1"] = [score_batch(E, [p], [r])[0] for p, r in items]
        order = sorted(range(len(items)), key=lambda j: len(items[j][1]))
        sorted_scores = [None] * len(items)
        for i in range(0, len(order), 32):
            idx = order[i:i + 32]
            for j, v in zip(idx, score_batch(E, [items[j][0] for j in idx], [items[j][1] for j in idx])):
                sorted_scores[j] = v
        out["sorted_bs32"] = sorted_scores
        # identical copies of the most common response
        p0, r0 = items[0]
        out["identical_x32"] = score_batch(E, [p0] * 32, [r0] * 32)
        out["identical_x2"] = score_batch(E, [p0] * 2, [r0] * 2)
        # same sorted batch but force padding by adding one long item
        idx = order[:31]
        long_item = (p0, r0 + " " + "Additionally, here is some more text to make this response longer. " * 6)
        padded = score_batch(E, [items[j][0] for j in idx] + [long_item[0]], [items[j][1] for j in idx] + [long_item[1]])
        out["sorted_bs32_padded_first31"] = padded[:31]
        out["sorted_bs32_first31"] = [sorted_scores[j] for j in idx]
        # token-length statistics of the first sorted batch
        lens = [len(tok(items[j][1])["input_ids"]) for j in order[:32]]
        out["first_batch_token_lengths_unique"] = sorted(set(lens))
        summary = {k: {"n": len(v), "n_const": sum(abs(x - CONST) < 1e-3 for x in v), "min": min(v), "max": max(v)}
                   for k, v in out.items() if isinstance(v, list) and v and isinstance(v[0], float)}
        summary["first_batch_token_lengths_unique"] = out["first_batch_token_lengths_unique"]
        print(label, json.dumps(summary, indent=1), flush=True)
        return summary

    model, tok = load_judge(jcfg)  # sdpa, as in the run
    report["sdpa"] = run_variants(model, tok, "sdpa")
    del model
    torch.cuda.empty_cache()
    dtype = getattr(torch, jcfg["dtype"])
    base = AutoModelForCausalLM.from_pretrained(jcfg["base_model"], revision=jcfg.get("base_model_revision"), dtype=dtype, attn_implementation="eager")
    model = PeftModel.from_pretrained(base, jcfg["adapter"], revision=jcfg.get("adapter_revision")).merge_and_unload().to("cuda").eval()
    report["eager"] = run_variants(model, tok, "eager")
    (REPO_ROOT / "results" / "judge_anomaly.json").write_text(json.dumps(report, indent=1))
    print("wrote results/judge_anomaly.json")


if __name__ == "__main__":
    main()
