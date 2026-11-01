"""Replay the exact judge batches of the full run that produced the constant-score artifact.

    python scripts/judge_anomaly_replay.py --config configs/experiment.yaml --shards 5,39

The run scored each shard's completions in file order, stably sorted by response length, in
batches of judge.batch_size. This reconstructs those batches, reports whether anomalies come in
whole batches, replays up to 20 anomalous batches and 10 normal ones (same items, same order)
with SDPA and eager attention, and writes results/judge_anomaly_replay.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import REPO_ROOT, dedupe, load_config, load_prompts, read_jsonl, shard_paths  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONST = 0.5266


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shards", default="5,39")
    args = ap.parse_args()
    cfg = load_config(args.config)
    jcfg = cfg["judge"]
    bs = int(jcfg["batch_size"])
    prompts = {p["id"]: p["forbidden_prompt"] for p in load_prompts(cfg)}
    report = {}
    import torch
    from judge import install_in_package_cache, score_batch
    from scripts_rescore_helper import load_judge_with  # noqa: F401  (defined below if missing)

    for s in [int(x) for x in args.shards.split(",")]:
        comps = dedupe(read_jsonl(shard_paths(cfg, "completions", s)[0]))
        scores = dedupe(read_jsonl(shard_paths(cfg, "scores", s)[0].with_suffix(".jsonl.orig")
                                   if shard_paths(cfg, "scores", s)[0].with_suffix(".jsonl.orig").exists()
                                   else shard_paths(cfg, "scores", s)[0]))
        order = sorted(comps, key=lambda k: len(comps[k]["text"]))  # stable, as in the run
        batches = [order[i:i + bs] for i in range(0, len(order), bs)]
        flags = [[abs(float(scores[k]["score"]) - CONST) < 1e-3 for k in b] for b in batches]
        n_bad_items = sum(map(sum, flags))
        full = sum(all(f) for f in flags)
        partial = sum(any(f) and not all(f) for f in flags)
        tok_stats = []
        rep = {"shard": s, "n_items": len(order), "n_batches": len(batches), "n_anomalous_items": n_bad_items,
               "batches_fully_anomalous": full, "batches_partially_anomalous": partial,
               "anomalous_batch_indices_first20": [i for i, f in enumerate(flags) if any(f)][:20],
               "partial_batch_examples": [[int(x) for x in f] for f in flags if any(f) and not all(f)][:3]}
        print(json.dumps(rep), flush=True)
        report[str(s)] = rep
        # replay
        from judge import load_judge
        res = {}
        for attn in ("sdpa", "eager"):
            model, tok = load_judge_with(jcfg, attn)
            E = install_in_package_cache(model, tok)
            bad_idx = [i for i, f in enumerate(flags) if any(f)][:20]
            good_idx = [i for i, f in enumerate(flags) if not any(f)][:10]
            out = {}
            for name, idxs in (("anomalous", bad_idx), ("normal", good_idx)):
                n_const, n_items, maxdiff = 0, 0, 0.0
                lens = []
                for i in idxs:
                    b = batches[i]
                    new = score_batch(E, [prompts[k[0]] for k in b], [comps[k]["text"] for k in b])
                    n_items += len(new)
                    n_const += sum(abs(v - CONST) < 1e-3 for v in new)
                    stored = [float(scores[k]["score"]) for k in b]
                    maxdiff = max(maxdiff, max(abs(a - c) for a, c in zip(new, stored) if abs(c - CONST) >= 1e-3) if any(abs(c - CONST) >= 1e-3 for c in stored) else 0)
                    enc = tok([__import__("strong_reject.evaluate", fromlist=["JUDGE_TEMPLATES"]).JUDGE_TEMPLATES["strongreject_finetuned"].format(forbidden_prompt=prompts[k[0]], response=comps[k]["text"]) for k in b])["input_ids"]
                    lens.append((min(map(len, enc)), max(map(len, enc))))
                out[name] = {"n_batches": len(idxs), "n_items": n_items, "n_const_on_replay": n_const,
                             "max_abs_diff_vs_stored_nonconst": maxdiff, "token_len_min_max_per_batch": lens[:8]}
            res[attn] = out
            print(s, attn, json.dumps(out), flush=True)
            del model, E
            torch.cuda.empty_cache()
        report[str(s)]["replay"] = res
    (REPO_ROOT / "results" / "judge_anomaly_replay.json").write_text(json.dumps(report, indent=1))
    print("wrote results/judge_anomaly_replay.json")


if __name__ == "__main__":
    main()
