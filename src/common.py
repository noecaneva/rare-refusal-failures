"""Shared helpers: config, run layout on scratch, prompt loading, sharding, tolerant JSONL I/O."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def scratch_root() -> Path:
    root = os.environ.get("TAILRISK_ROOT")
    if not root:
        scratch = os.environ.get("SCRATCH")
        if not scratch:
            raise SystemExit("Set TAILRISK_ROOT (or SCRATCH); see slurm/env.sh")
        root = os.path.join(scratch, "qwen_tailrisk")
    return Path(root)


def run_dir(cfg: dict) -> Path:
    d = scratch_root() / "runs" / cfg["name"]
    for sub in ("completions", "scores"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------- prompts

def _select_indices(n_total: int, n_prompts: int | None) -> list[int]:
    if n_prompts is None or n_prompts >= n_total:
        return list(range(n_total))
    # evenly spaced through the (category-sorted) dataset so a small pilot covers all categories
    return sorted({(i * n_total) // n_prompts for i in range(n_prompts)})


def load_prompts(cfg: dict) -> list[dict]:
    """The StrongREJECT forbidden prompts (no jailbreak) with id, category, source, forbidden_prompt.

    `id` is the row index in the StrongREJECT dataset CSV (0..312), which has no id column of its
    own. The list is cached in the run directory so every stage sees the same ids.
    """
    cache = run_dir(cfg) / "prompts.jsonl"
    if cache.exists():
        rows = read_jsonl(cache)
        if rows:
            return rows
    from strong_reject.load_datasets import load_strongreject

    ds = load_strongreject()
    all_rows = [
        {"id": i, "category": r["category"], "source": r["source"], "forbidden_prompt": r["forbidden_prompt"]}
        for i, r in enumerate(ds)
    ]
    rows = [all_rows[i] for i in _select_indices(len(all_rows), cfg.get("n_prompts"))]
    write_jsonl(cache, rows)
    return rows


def shards(prompts: list[dict], shard_size: int) -> list[list[dict]]:
    return [prompts[i : i + shard_size] for i in range(0, len(prompts), shard_size)]


def n_shards(cfg: dict) -> int:
    return math.ceil(len(load_prompts(cfg)) / cfg["shard_size"])


def shard_paths(cfg: dict, stage: str, shard: int) -> tuple[Path, Path, Path]:
    """(jsonl, done marker, meta json) for a stage ('completions' or 'scores') and shard."""
    base = run_dir(cfg) / stage / f"shard_{shard:03d}"
    return base.with_suffix(".jsonl"), base.with_suffix(".done"), base.with_suffix(".meta.json")


def seed_for(cfg: dict, prompt_id: int, chunk: int) -> int:
    return (int(cfg["seed"]) * 1_000_003 + prompt_id * 1_009 + chunk) % (2**31 - 1)


# ----------------------------------------------------------------------------- jsonl

def read_jsonl(path: str | os.PathLike) -> list[dict]:
    """Read a JSONL file, tolerating a truncated final line (a job killed mid-write)."""
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path: str | os.PathLike, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


class JsonlAppender:
    """Append-only writer that flushes and fsyncs on demand (resume safety on Lustre)."""

    def __init__(self, path: str | os.PathLike):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a")

    def write(self, rows: list[dict]) -> None:
        for r in rows:
            self.f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self) -> None:
        self.f.close()


def dedupe(rows: list[dict]) -> dict[tuple[int, int], dict]:
    """Index rows by (prompt_id, sample_idx), keeping the first occurrence."""
    out: dict[tuple[int, int], dict] = {}
    for r in rows:
        key = (int(r["prompt_id"]), int(r["sample_idx"]))
        if key not in out:
            out[key] = r
    return out


# ----------------------------------------------------------------------------- environment info

def env_info() -> dict:
    info = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - torch missing in the analysis-only environment
        pass
    for mod in ("vllm", "transformers", "peft"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    return info


def gpu_memory_used_mib() -> int | None:
    """Memory in use on the visible GPU as reported by nvidia-smi (covers vLLM's own allocator)."""
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()
        return int(out[0])
    except Exception:
        return None


# ----------------------------------------------------------------------------- work queue

def claim_next_shard(cfg: dict, stage: str, owner: str) -> int | None:
    """Atomically claim the lowest-numbered shard of `stage` that is neither done nor claimed.

    A claim is a lock file created with O_EXCL (atomic on Lustre); it holds the owner string
    ("<job id>:<gpu>"). For the 'scores' stage only shards whose generation is complete count.
    Returns the shard index, or None when there is nothing left to claim.
    """
    for s in range(n_shards(cfg)):
        _, done, _ = shard_paths(cfg, stage, s)
        if done.exists():
            continue
        if stage == "scores" and not shard_paths(cfg, "completions", s)[1].exists():
            continue
        lock = done.with_suffix(".lock")
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, owner.encode())
        os.close(fd)
        return s
    return None


def release_shard(cfg: dict, stage: str, shard: int) -> None:
    lock = shard_paths(cfg, stage, shard)[1].with_suffix(".lock")
    try:
        lock.unlink()
    except FileNotFoundError:
        pass


def clean_stale_locks(cfg: dict, live_job_ids: set[str]) -> list[str]:
    """Remove locks whose owning SLURM job is no longer in the queue (crashed or timed out)."""
    removed = []
    for stage in ("completions", "scores"):
        for lock in (run_dir(cfg) / stage).glob("shard_*.lock"):
            owner = lock.read_text().strip()
            if owner.split(":")[0] not in live_job_ids:
                lock.unlink()
                removed.append(f"{stage}/{lock.name} ({owner})")
    return removed


def run_claimed(cfg: dict, stage: str, owner: str, process) -> int:
    """Worker loop: claim shards until none are left; `process(shard)` does the work.

    On an exception the lock is released and a `.failed` note is written so another worker can
    retry once, then the error propagates and this worker stops.
    """
    n = 0
    while True:
        s = claim_next_shard(cfg, stage, owner)
        if s is None:
            print(f"[{owner}] no shards left to claim for {stage}; processed {n}", flush=True)
            return n
        print(f"[{owner}] claimed shard {s}", flush=True)
        try:
            process(s)
        except BaseException as e:  # noqa: BLE001 - we re-raise after cleanup
            shard_paths(cfg, stage, s)[1].with_suffix(".failed").write_text(f"{owner}: {type(e).__name__}: {e}\n")
            release_shard(cfg, stage, s)
            raise
        release_shard(cfg, stage, s)
        n += 1
