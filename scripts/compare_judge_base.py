"""Check whether the judge base weights used in a run match another base repository.

    python scripts/compare_judge_base.py --config configs/experiment.yaml --official google/gemma-2b \
        --official-revision 9cf48e52b224239de00d483ec8eb84fb8d0f3a3a [--rescore 256]

1. Downloads both bases and compares every tensor bit for bit (safetensors, CPU, no model built).
2. Optionally loads the judge (adapter on the official base) on the GPU and re-scores a random
   sample of already-scored completions from the run; reports max |delta| and label agreement.
Writes the report to results/judge_base_check.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import REPO_ROOT, dedupe, load_config, load_prompts, read_jsonl, shard_paths, n_shards  # noqa: E402


def tensor_files(repo: str, revision: str | None) -> list[str]:
    from huggingface_hub import snapshot_download

    d = snapshot_download(repo, revision=revision, allow_patterns=["*.safetensors", "*.json"])
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".safetensors"))


def load_all(files: list[str]):
    from safetensors import safe_open

    out = {}
    for f in files:
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                out[k] = sf.get_tensor(k)
    return out


def compare_weights(a_repo, a_rev, b_repo, b_rev) -> dict:
    import torch

    a, b = load_all(tensor_files(a_repo, a_rev)), load_all(tensor_files(b_repo, b_rev))
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    diff, dtype_diff, n_equal = [], [], 0
    for k in sorted(set(a) & set(b)):
        ta, tb = a[k], b[k]
        if ta.dtype != tb.dtype:
            dtype_diff.append((k, str(ta.dtype), str(tb.dtype)))
            ta, tb = ta.float(), tb.float()
        if ta.shape != tb.shape or not torch.equal(ta, tb):
            diff.append({"tensor": k, "shape_a": list(ta.shape), "shape_b": list(tb.shape),
                         "max_abs": float((ta.float() - tb.float()).abs().max()) if ta.shape == tb.shape else None})
        else:
            n_equal += 1
    return {"n_tensors_a": len(a), "n_tensors_b": len(b), "n_equal": n_equal, "only_in_a": only_a,
            "only_in_b": only_b, "dtype_differences": dtype_diff, "differing_tensors": diff,
            "identical": not (only_a or only_b or diff)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--official", default="google/gemma-2b")
    ap.add_argument("--official-revision", default=None)
    ap.add_argument("--rescore", type=int, default=0, help="re-score N random already-scored completions with the official base")
    args = ap.parse_args()
    cfg = load_config(args.config)
    jcfg = cfg["judge"]
    report = {"run_base": jcfg["base_model"], "run_base_revision": jcfg.get("base_model_revision"),
              "official": args.official, "official_revision": args.official_revision}
    t0 = time.time()
    report["weights"] = compare_weights(jcfg["base_model"], jcfg.get("base_model_revision"), args.official, args.official_revision)
    report["weights"]["seconds"] = round(time.time() - t0, 1)
    print("weights identical:", report["weights"]["identical"], json.dumps({k: v for k, v in report["weights"].items() if k != "differing_tensors"}))
    if report["weights"]["differing_tensors"]:
        print("first differing tensors:", report["weights"]["differing_tensors"][:5])

    if args.rescore:
        import torch
        from judge import LastTokenLogits, load_judge, install_in_package_cache, score_batch

        prompts = {p["id"]: p["forbidden_prompt"] for p in load_prompts(cfg)}
        rnd = random.Random(0)
        shards_ = rnd.sample(range(n_shards(cfg)), min(4, n_shards(cfg)))
        items = []
        for s in shards_:
            comps = dedupe(read_jsonl(shard_paths(cfg, "completions", s)[0]))
            scores = dedupe(read_jsonl(shard_paths(cfg, "scores", s)[0]))
            keys = rnd.sample(sorted(scores), args.rescore // len(shards_))
            # oversample harmful ones so the threshold region is tested too
            harmful = [k for k in scores if float(scores[k]["score"]) >= jcfg["threshold"]]
            keys += rnd.sample(harmful, min(len(harmful), 16))
            items += [(prompts[k[0]], comps[k]["text"], float(scores[k]["score"])) for k in keys]
        official = dict(jcfg, base_model=args.official, base_model_revision=args.official_revision)
        model, tok = load_judge(official)
        E = install_in_package_cache(model, tok)
        new = []
        for i in range(0, len(items), int(jcfg["batch_size"])):
            b = items[i:i + int(jcfg["batch_size"])]
            new += score_batch(E, [p for p, _, _ in b], [r for _, r, _ in b])
        old = [o for _, _, o in items]
        thr = jcfg["threshold"]
        d = [abs(a - b) for a, b in zip(new, old)]
        report["rescore"] = {"n": len(items), "max_abs_diff": max(d), "mean_abs_diff": sum(d) / len(d),
                             "label_agreement": sum((a >= thr) == (b >= thr) for a, b in zip(new, old)) / len(d),
                             "n_harmful_old": sum(b >= thr for b in old), "n_harmful_new": sum(a >= thr for a in new)}
        print("rescore:", json.dumps(report["rescore"]))
    out = REPO_ROOT / "results" / "judge_base_check.json"
    out.write_text(json.dumps(report, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
