"""Stage 1: sample completions for one shard of prompts with vLLM.

Usage (inside a SLURM job, one GPU):
    python src/generate.py --config configs/experiment.yaml --shard 3

Writes $TAILRISK_ROOT/runs/<name>/completions/shard_003.jsonl with one line per completion
({prompt_id, sample_idx, text, finish_reason, n_tokens}) and a .done marker when the shard is
complete. Re-running skips chunks that are already on disk, so a killed job just resumes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (JsonlAppender, dedupe, env_info, gpu_memory_used_mib, load_config,  # noqa: E402
                    load_prompts, read_jsonl, run_claimed, seed_for, shard_paths, shards)


class Shard(argparse.Namespace):
    pass


def process_shard(cfg: dict, shard_idx: int, engine: dict) -> None:
    """Generate every missing chunk of one shard. `engine` caches the vLLM instance across shards."""
    vcfg = cfg["vllm"]
    args = Shard(shard=shard_idx)
    prompts = load_prompts(cfg)
    all_shards = shards(prompts, cfg["shard_size"])
    if not 0 <= args.shard < len(all_shards):
        raise SystemExit(f"shard {args.shard} out of range (0..{len(all_shards) - 1})")
    shard = all_shards[args.shard]
    out_path, done_path, meta_path = shard_paths(cfg, "completions", args.shard)
    if done_path.exists():
        print(f"shard {args.shard}: already complete ({done_path})")
        return

    n_samples, chunk = int(cfg["n_samples"]), int(cfg["chunk_size"])
    n_chunks = math.ceil(n_samples / chunk)
    have = dedupe(read_jsonl(out_path))
    todo: list[tuple[int, int]] = []  # (chunk, prompt_id)
    for c in range(n_chunks):
        idxs = range(c * chunk, min((c + 1) * chunk, n_samples))
        for p in shard:
            if not all((p["id"], i) in have for i in idxs):
                todo.append((c, p["id"]))
    print(f"shard {args.shard}: {len(shard)} prompts, {len(have)} completions on disk, "
          f"{len(todo)} of {n_chunks * len(shard)} chunks to generate", flush=True)

    meta = {"config": {k: v for k, v in cfg.items() if not k.startswith("_")}, "shard": args.shard,
            "prompt_ids": [p["id"] for p in shard], "env": env_info(), "chunks_generated": len(todo),
            "resumed_from": len(have)}
    if not todo:
        meta["note"] = "nothing to do; marker written on resume"
        done_path.write_text(json.dumps(meta, indent=1))
        return

    from vllm import LLM, SamplingParams

    t0 = time.time()
    if "llm" not in engine:
        engine["llm"] = LLM(
            model=cfg["model"], revision=cfg.get("model_revision"), dtype=vcfg["dtype"],
            gpu_memory_utilization=vcfg["gpu_memory_utilization"], max_model_len=vcfg["max_model_len"],
            max_num_seqs=vcfg["max_num_seqs"], enable_prefix_caching=vcfg.get("enable_prefix_caching", True),
            generation_config=vcfg.get("generation_config", "vllm"), seed=int(cfg["seed"]),
            attention_backend=vcfg["attention_backend"],
        )
        engine["load_seconds"] = round(time.time() - t0, 1)
    llm = engine["llm"]
    meta["load_seconds"] = engine["load_seconds"]
    s = cfg["sampling"]
    by_id = {p["id"]: p for p in shard}

    def conv(p):
        msgs = []
        if cfg.get("system_prompt"):
            msgs.append({"role": "system", "content": cfg["system_prompt"]})
        msgs.append({"role": "user", "content": p["forbidden_prompt"]})
        return msgs

    # Record exactly what the model sees for the first prompt of the shard (chat template applied).
    tok = llm.get_tokenizer()
    meta["rendered_prompt_example"] = tok.apply_chat_template(conv(shard[0]), tokenize=False,
                                                              add_generation_prompt=True)

    writer = JsonlAppender(out_path)
    gen_t0, n_tok, n_done = time.time(), 0, 0
    for c in range(n_chunks):
        pids = [pid for (cc, pid) in todo if cc == c]
        if not pids:
            continue
        lo, hi = c * chunk, min((c + 1) * chunk, n_samples)
        params = [SamplingParams(n=hi - lo, temperature=s["temperature"], top_p=s["top_p"],
                                 top_k=s["top_k"], repetition_penalty=s["repetition_penalty"],
                                 max_tokens=s["max_new_tokens"], seed=seed_for(cfg, pid, c))
                  for pid in pids]
        t1 = time.time()
        outs = llm.chat([conv(by_id[pid]) for pid in pids], params, use_tqdm=False)
        rows = []
        for pid, out in zip(pids, outs):
            assert len(out.outputs) == hi - lo, (pid, len(out.outputs))
            for j, o in enumerate(out.outputs):
                rows.append({"prompt_id": pid, "sample_idx": lo + j, "text": o.text,
                             "finish_reason": o.finish_reason, "n_tokens": len(o.token_ids)})
                n_tok += len(o.token_ids)
        writer.write(rows)
        n_done += len(rows)
        dt = time.time() - t1
        print(f"chunk {c + 1}/{n_chunks}: {len(rows)} completions in {dt:.0f}s "
              f"({sum(r['n_tokens'] for r in rows) / max(dt, 1e-9):.0f} tok/s), "
              f"GPU mem {gpu_memory_used_mib()} MiB", flush=True)
    writer.close()

    gen_s = time.time() - gen_t0
    meta.update({"generate_seconds": round(gen_s, 1), "completions_written": n_done,
                 "generated_tokens": n_tok, "tokens_per_second": round(n_tok / max(gen_s, 1e-9)),
                 "gpu_memory_used_mib": gpu_memory_used_mib(), "total_seconds": round(time.time() - t0, 1)})
    final = dedupe(read_jsonl(out_path))
    complete = all((p["id"], i) in final for p in shard for i in range(n_samples))
    meta_path.write_text(json.dumps(meta, indent=1))
    if complete:
        done_path.write_text(json.dumps({"shard": args.shard, "n_rows": len(final)}))
        print(f"shard {args.shard}: complete, {len(final)} completions, {meta['tokens_per_second']} tok/s")
    else:
        raise SystemExit(f"shard {args.shard}: incomplete after run ({len(final)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", type=int, help="process this one shard")
    ap.add_argument("--claim", action="store_true", help="worker mode: claim and process shards until none are left")
    ap.add_argument("--owner", default=f"{os.environ.get('SLURM_JOB_ID', 'local')}:{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")
    args = ap.parse_args()
    cfg = load_config(args.config)
    engine: dict = {}
    if args.claim:
        run_claimed(cfg, "completions", args.owner, lambda s: process_shard(cfg, s, engine))
    elif args.shard is not None:
        process_shard(cfg, args.shard, engine)
    else:
        ap.error("give --shard N or --claim")


if __name__ == "__main__":
    main()
