"""List shards that still need work, for building SLURM array indices.

    python src/shards.py --config configs/experiment.yaml --stage generate   # -> e.g. 0,3,7
    python src/shards.py --config configs/experiment.yaml --stage judge
    python src/shards.py --config configs/experiment.yaml --status           # human-readable table
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import clean_stale_locks, load_config, n_shards, shard_paths  # noqa: E402


def missing(cfg: dict, stage: str) -> list[int]:
    out = []
    for s in range(n_shards(cfg)):
        _, gen_done, _ = shard_paths(cfg, "completions", s)
        _, judge_done, _ = shard_paths(cfg, "scores", s)
        if stage == "generate" and not gen_done.exists():
            out.append(s)
        if stage == "generate-claimable" and not gen_done.exists() and not gen_done.with_suffix(".lock").exists():
            out.append(s)
        # Judging only makes sense for shards whose generation is finished.
        if stage == "judge" and gen_done.exists() and not judge_done.exists():
            out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["generate", "judge", "generate-claimable"])
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--clean-locks", action="store_true", help="drop locks held by SLURM jobs that no longer exist")
    args = ap.parse_args()
    cfg = load_config(args.config)
    n = n_shards(cfg)
    if args.clean_locks:
        import subprocess
        live = set(subprocess.run(["squeue", "--me", "-h", "-o", "%A"], capture_output=True, text=True).stdout.split())
        for r in clean_stale_locks(cfg, live):
            print("removed stale lock", r)
        return
    if args.status:
        g, j = missing(cfg, "generate"), missing(cfg, "judge")
        print(f"{n} shards: {n - len(g)} generated, {n - len(g) - len(j)} judged; "
              f"generate pending {g}; judge pending {j}")
        return
    print(",".join(map(str, missing(cfg, args.stage))))


if __name__ == "__main__":
    main()
