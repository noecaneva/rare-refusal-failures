"""Build the Llama-3.1-8B-Instruct reference curve from the paper's released per-prompt scores.

    python scripts/reference_from_paper.py --config configs/experiment.yaml

Downloads strong_reject/Llama-3.1-8B-Instruct_mc_est_10k.json from the dataset rangell/mc_estimates
(Angell et al.; one JSON line per prompt with 10,000 responses and judge scores) into the scratch
cache, matches prompts to ours by text, applies the same threshold and the same analysis functions,
and writes results/reference_llama31_8b_curve.csv, results/reference_llama31_8b_per_prompt.csv and
updates configs/reference_llama31_8b.yaml (all k values) for the plot overlay.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from analyze import analytic_cracked, empirical_cracked  # noqa: E402
from common import REPO_ROOT, load_config, load_prompts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--repo", default="rangell/mc_estimates")
    ap.add_argument("--file", default="strong_reject/Llama-3.1-8B-Instruct_mc_est_10k.json")
    ap.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    args = ap.parse_args()
    cfg = load_config(args.config)
    thr = float(cfg["judge"]["threshold"])
    acfg = cfg["analysis"]
    k_values = list(acfg["k_values"])

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(args.repo, args.file, repo_type="dataset")
    print("downloaded", path, round(os.path.getsize(path) / 2**20), "MiB")

    ours = {p["forbidden_prompt"].strip(): p for p in load_prompts(cfg)}
    scores: dict[int, np.ndarray] = {}
    keys_seen = None
    unmatched = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if keys_seen is None:
                keys_seen = sorted(k for k in rec if k not in ("response", "score"))
                print("record fields:", keys_seen, "| n responses:", len(rec["response"]), "| n scores:", len(rec["score"]))
            p = ours.get(str(rec["forbidden_prompt"]).strip())
            if p is None:
                unmatched += 1
                continue
            scores[p["id"]] = np.asarray(rec["score"], dtype=float)
    print(f"matched {len(scores)} of {len(ours)} prompts; {unmatched} records unmatched")
    harmful = {pid: s >= thr for pid, s in scores.items()}
    rates = np.array([harmful[pid].mean() for pid in sorted(harmful)])
    n_samples = int(np.median([s.size for s in scores.values()]))
    ana = analytic_cracked(rates, k_values)
    emp_m, emp_s = empirical_cracked(harmful, k_values, int(acfg["n_orderings"]), int(acfg["ordering_seed"]))
    curve = pd.DataFrame({"k": k_values, "empirical_mean": emp_m, "empirical_std": emp_s, "analytic": ana, "n_prompts": len(rates)})
    out = REPO_ROOT / "results"
    curve.to_csv(out / "reference_llama31_8b_curve.csv", index=False)
    rows = []
    by_id = {p["id"]: p for p in ours.values()}
    for pid in sorted(harmful):
        h = harmful[pid]
        rows.append({"prompt_id": pid, "category": by_id[pid]["category"], "source": by_id[pid]["source"],
                     "n_samples": int(h.size), "n_harmful": int(h.sum()), "harm_rate": float(h.mean()),
                     "mean_score": float(scores[pid].mean())})
    pd.DataFrame(rows).to_csv(out / "reference_llama31_8b_per_prompt.csv", index=False)
    all_scores = np.concatenate(list(scores.values()))
    summary = {"model": args.model_name, "source_file": f"{args.repo}/{args.file}", "n_prompts": len(rates),
               "n_samples_per_prompt": n_samples, "n_zero_harmful": int((rates == 0).sum()),
               "overall_harm_rate": float((all_scores >= thr).mean()),
               "score_values_seen": sorted(set(np.round(np.unique(all_scores), 4).tolist()))[:12],
               "cracked_empirical": {int(k): float(v) for k, v in zip(k_values, emp_m)},
               "cracked_analytic": {int(k): float(v) for k, v in zip(k_values, ana)}}
    (out / "reference_llama31_8b_summary.json").write_text(json.dumps(summary, indent=1))
    ref = {"model": args.model_name, "source": f"{args.repo}/{args.file} (Angell et al., arXiv 2604.22167), "
                                                "re-analysed with src/analyze.py at the same threshold",
           "n_prompts": len(rates), "n_zero_harmful": int((rates == 0).sum()),
           "cracked": {int(k): round(float(v), 4) for k, v in zip(k_values, emp_m)}}
    (REPO_ROOT / "configs" / "reference_llama31_8b.yaml").write_text(
        "# Reference curve for Llama-3.1-8B-Instruct, computed by scripts/reference_from_paper.py from the\n"
        "# paper's released per-prompt judge scores (empirical, 20 orderings). Used for the plot overlay.\n"
        + yaml.safe_dump(ref, sort_keys=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
