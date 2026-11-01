"""Re-score judge outputs that carry the constant artifact value, in place, with a safe judge setup.

    python scripts/rescore_anomalous.py --config configs/experiment.yaml --shards 0,4,8 \
        [--attn eager|sdpa] [--batch-size 1] [--verify 200]

For each shard: find stored scores within 1e-3 of the artifact constant, re-score those items with
the chosen attention implementation and batch size, rewrite the shard's scores file (the original
is kept as shard_NNN.jsonl.orig) and write shard_NNN.rescore.json with counts. `--verify N` also
re-scores N random unaffected items and reports how closely they match.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import dedupe, load_config, load_prompts, read_jsonl, shard_paths, write_jsonl  # noqa: E402

CONST = 0.5266


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts_rescore_helper import load_judge_with  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shards", required=True, help="comma-separated shard indices, or 'all'")
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa", "sdpa-math"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--verify", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    jcfg = cfg["judge"]
    thr = float(jcfg["threshold"])
    from common import n_shards
    shards = list(range(n_shards(cfg))) if args.shards == "all" else [int(x) for x in args.shards.split(",")]
    prompts = {p["id"]: p["forbidden_prompt"] for p in load_prompts(cfg)}

    from judge import install_in_package_cache, score_batch

    model, tok = load_judge_with(jcfg, args.attn)
    E = install_in_package_cache(model, tok)
    rnd = random.Random(0)
    for s in shards:
        score_path, done_path, _ = shard_paths(cfg, "scores", s)
        rescore_meta = score_path.with_suffix(".rescore.json")
        if rescore_meta.exists():
            print(f"shard {s}: already re-scored")
            continue
        scores = dedupe(read_jsonl(score_path))
        bad = [k for k, r in scores.items() if abs(float(r["score"]) - CONST) < 1e-3]
        comps = dedupe(read_jsonl(shard_paths(cfg, "completions", s)[0]))
        t0 = time.time()
        new = {}
        bad.sort(key=lambda k: len(comps[k]["text"]))
        for i in range(0, len(bad), args.batch_size):
            keys = bad[i:i + args.batch_size]
            for k, v in zip(keys, score_batch(E, [prompts[k[0]] for k in keys], [comps[k]["text"] for k in keys])):
                new[k] = float(v)
        still_const = sum(abs(v - CONST) < 1e-3 for v in new.values())
        meta = {"shard": s, "n_scores": len(scores), "n_anomalous": len(bad), "n_still_constant_after": still_const,
                "n_newly_harmful": sum(v >= thr for v in new.values()), "attn": args.attn, "batch_size": args.batch_size,
                "seconds": round(time.time() - t0, 1)}
        if args.verify:
            good = [k for k in scores if k not in new]
            sample = rnd.sample(good, min(args.verify, len(good)))
            ver = {}
            for i in range(0, len(sample), args.batch_size):
                keys = sample[i:i + args.batch_size]
                for k, v in zip(keys, score_batch(E, [prompts[k[0]] for k in keys], [comps[k]["text"] for k in keys])):
                    ver[k] = float(v)
            d = [abs(ver[k] - float(scores[k]["score"])) for k in ver]
            meta["verify"] = {"n": len(d), "max_abs_diff": max(d), "mean_abs_diff": sum(d) / len(d),
                              "label_flips": sum((ver[k] >= thr) != (float(scores[k]["score"]) >= thr) for k in ver)}
        if new:
            if not score_path.with_suffix(".jsonl.orig").exists():
                shutil.copy2(score_path, score_path.with_suffix(".jsonl.orig"))
            rows = []
            for k, r in scores.items():
                if k in new:
                    r = dict(r, score=round(new[k], 6), rescored=True)
                rows.append(r)
            write_jsonl(score_path, rows)
        rescore_meta.write_text(json.dumps(meta, indent=1))
        print(f"shard {s}:", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
