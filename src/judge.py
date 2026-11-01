"""Stage 2: score one shard of completions with the StrongREJECT fine-tuned judge.

Usage (inside a SLURM job, one GPU):
    python src/judge.py --config configs/experiment.yaml --shard 3 [--selfcheck 64]

The scoring code is the `strongreject_finetuned` evaluator of the `strong_reject` package,
called unchanged: this script only loads the judge itself (LoRA adapter on Gemma-2B) in float16
and injects it into the package's model cache. See README for why (gated base model, no bf16 on
the GPUs used). Output: $TAILRISK_ROOT/runs/<name>/scores/shard_003.jsonl with
{prompt_id, sample_idx, score} plus a .done marker; re-running resumes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (JsonlAppender, dedupe, env_info, load_config, load_prompts, read_jsonl,  # noqa: E402
                    run_claimed, shard_paths, shards)


def load_judge(jcfg: dict, device: str = "cuda", dtype_name: str | None = None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_name or jcfg["dtype"])
    base = AutoModelForCausalLM.from_pretrained(jcfg["base_model"], revision=jcfg.get("base_model_revision"),
                                                dtype=dtype, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, jcfg["adapter"], revision=jcfg.get("adapter_revision"))
    model = model.merge_and_unload().to(device).eval()
    # Same tokenizer settings as the package's evaluator (the adapter repo ships the tokenizer).
    tok = AutoTokenizer.from_pretrained(jcfg["adapter"], revision=jcfg.get("adapter_revision"),
                                        padding_side="left", truncation_side="left")
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    return model, tok


class LastTokenLogits:
    """Forward wrapper that asks the model for logits at the last position only.

    The package evaluator computes `model(...).logits[:, -1]`; with Gemma's 256k vocabulary the
    full (batch, seq, vocab) tensor is ~9 GiB for a batch of 32, far more than an 11 GB GPU holds.
    `logits_to_keep=1` returns a (batch, 1, vocab) tensor, so `[:, -1]` gives identical scores.
    """

    def __init__(self, model):
        self.model = model

    @property
    def device(self):
        return self.model.device

    def __call__(self, **kwargs):
        return self.model(**kwargs, logits_to_keep=1)


def install_in_package_cache(model, tok, name: str = "strongreject_finetuned"):
    from strong_reject import evaluate as E

    E.cached_models[name] = (LastTokenLogits(model), tok)
    return E


def score_batch(E, prompts: list[str], responses: list[str], name: str = "strongreject_finetuned") -> list[float]:
    import torch

    with torch.inference_mode():
        return E.strongreject_finetuned({"forbidden_prompt": prompts, "response": responses})["score"]


def selfcheck(E, model, tok, items: list[tuple[str, str]], jcfg: dict, n_cpu: int = 8) -> dict:
    """Compare the float16 GPU judge against a float32 CPU copy of the same merged model."""
    import torch

    gpu = score_batch(E, [p for p, _ in items], [r for _, r in items])
    cpu_model = model.to("cpu", dtype=torch.float32)
    E.cached_models["strongreject_finetuned"] = (LastTokenLogits(cpu_model), tok)
    sub = items[:n_cpu]
    t0 = time.time()
    cpu = []
    for p, r in sub:
        cpu += score_batch(E, [p], [r])
    cpu_s = time.time() - t0
    model.to("cuda", dtype=getattr(torch, jcfg["dtype"]))
    E.cached_models["strongreject_finetuned"] = (LastTokenLogits(model), tok)
    diffs = [abs(a - b) for a, b in zip(gpu[:n_cpu], cpu)]
    thr = jcfg["threshold"]
    return {"n_gpu": len(gpu), "n_cpu_fp32": len(cpu), "max_abs_diff": max(diffs), "mean_abs_diff": sum(diffs) / len(diffs),
            "label_agreement": sum((a >= thr) == (b >= thr) for a, b in zip(gpu[:n_cpu], cpu)) / len(cpu),
            "gpu_scores": [round(x, 4) for x in gpu[:n_cpu]], "cpu_fp32_scores": [round(x, 4) for x in cpu],
            "cpu_seconds_per_item": round(cpu_s / len(cpu), 1)}


def process_shard(cfg: dict, shard_idx: int, engine: dict, selfcheck_n: int = 0, allow_partial: bool = False) -> None:
    """Score every unscored completion of one shard. `engine` caches the judge across shards."""
    args = argparse.Namespace(shard=shard_idx, selfcheck=selfcheck_n, allow_partial=allow_partial)
    jcfg = cfg["judge"]

    prompts = load_prompts(cfg)
    shard = shards(prompts, cfg["shard_size"])[args.shard]
    text_by_id = {p["id"]: p["forbidden_prompt"] for p in shard}
    comp_path, comp_done, _ = shard_paths(cfg, "completions", args.shard)
    out_path, done_path, meta_path = shard_paths(cfg, "scores", args.shard)
    if done_path.exists():
        print(f"shard {args.shard}: scores already complete")
        return
    if not comp_done.exists() and not args.allow_partial:
        raise SystemExit(f"shard {args.shard}: completions not finished ({comp_done} missing)")

    comps = dedupe(read_jsonl(comp_path))
    have = dedupe(read_jsonl(out_path))
    todo = [k for k in comps if k not in have]
    # Group similar lengths together so padding is cheap.
    todo.sort(key=lambda k: len(comps[k]["text"]))
    print(f"shard {args.shard}: {len(comps)} completions, {len(have)} scored, {len(todo)} to score", flush=True)

    meta = {"shard": args.shard, "judge": jcfg, "env": env_info(), "resumed_from": len(have), "n_to_score": len(todo)}
    if not todo:
        done_path.write_text(json.dumps({"shard": args.shard, "n_rows": len(have)}))
        return

    import torch

    t0 = time.time()
    if "model" not in engine:
        engine["model"], engine["tok"] = load_judge(jcfg)
        engine["load_seconds"] = round(time.time() - t0, 1)
    model, tok = engine["model"], engine["tok"]
    E = install_in_package_cache(model, tok)
    meta["load_seconds"] = engine["load_seconds"]
    torch.cuda.reset_peak_memory_stats()

    if args.selfcheck:
        import random

        rnd = random.Random(0)
        keys = rnd.sample(todo, min(args.selfcheck, len(todo)))
        items = [(text_by_id[k[0]], comps[k]["text"]) for k in keys]
        meta["selfcheck"] = selfcheck(E, model, tok, items, jcfg)
        print("selfcheck:", json.dumps(meta["selfcheck"]), flush=True)

    bs = int(jcfg["batch_size"])
    writer = JsonlAppender(out_path)
    t1, n_done, buf = time.time(), 0, []
    n_batches = math.ceil(len(todo) / bs)
    for b in range(n_batches):
        keys = todo[b * bs:(b + 1) * bs]
        scores = score_batch(E, [text_by_id[k[0]] for k in keys], [comps[k]["text"] for k in keys])
        if any(math.isnan(x) for x in scores):
            raise RuntimeError("NaN judge score; check dtype/overflow")
        buf += [{"prompt_id": k[0], "sample_idx": k[1], "score": round(float(s), 6)} for k, s in zip(keys, scores)]
        n_done += len(keys)
        if len(buf) >= 2000 or b == n_batches - 1:
            writer.write(buf)
            buf = []
            dt = time.time() - t1
            print(f"{n_done}/{len(todo)} scored, {n_done / dt:.1f} items/s, "
                  f"GPU peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)
    writer.close()
    judge_s = time.time() - t1
    final = dedupe(read_jsonl(out_path))
    meta.update({"judge_seconds": round(judge_s, 1), "items_per_second": round(n_done / max(judge_s, 1e-9), 2),
                 "gpu_peak_memory_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                 "total_seconds": round(time.time() - t0, 1), "n_scored_total": len(final)})
    meta_path.write_text(json.dumps(meta, indent=1))
    if all(k in final for k in comps) and (comp_done.exists() or args.allow_partial):
        if comp_done.exists():
            done_path.write_text(json.dumps({"shard": args.shard, "n_rows": len(final)}))
        print(f"shard {args.shard}: scored {len(final)} completions at {meta['items_per_second']} items/s")
    else:
        raise SystemExit(f"shard {args.shard}: incomplete after run")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", type=int, help="process this one shard")
    ap.add_argument("--claim", action="store_true", help="worker mode: claim and process shards until none are left")
    ap.add_argument("--owner", default=f"{os.environ.get('SLURM_JOB_ID', 'local')}:{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")
    ap.add_argument("--selfcheck", type=int, default=0, help="also compare N items against an fp32 CPU judge (first shard only)")
    ap.add_argument("--allow-partial", action="store_true", help="score a shard whose generation is not finished")
    args = ap.parse_args()
    cfg = load_config(args.config)
    engine: dict = {}
    if args.claim:
        def work(s):
            process_shard(cfg, s, engine, selfcheck_n=0 if engine.get("checked") else args.selfcheck, allow_partial=args.allow_partial)
            engine["checked"] = True
        run_claimed(cfg, "scores", args.owner, work)
    elif args.shard is not None:
        process_shard(cfg, args.shard, engine, args.selfcheck, args.allow_partial)
    else:
        ap.error("give --shard N or --claim")


if __name__ == "__main__":
    main()
