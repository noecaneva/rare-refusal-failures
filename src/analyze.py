"""Stage 3: aggregate judge scores into the tables and plots in results/.

    python src/analyze.py --config configs/experiment.yaml --out results

Reads whatever score shards exist (incomplete runs are summarised with a warning) and writes:
  per_prompt.csv, per_category.csv, cracked_curve.csv, harm_rate_histogram.csv, summary.json,
  metadata.json, cracked_curve.png, harm_rate_histogram.png, per_category.png
No prompt text or completions are written: only ids, category, source and aggregate numbers.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO_ROOT, dedupe, load_config, load_prompts, read_jsonl, run_dir  # noqa: E402

DEFAULT_K = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]


# ----------------------------------------------------------------------------- core statistics

def per_prompt_table(prompts: list[dict], harmful: dict[int, np.ndarray], n_expected: int,
                     scores: dict[int, np.ndarray] | None = None) -> pd.DataFrame:
    """One row per prompt. `harmful[id]` is a boolean array over that prompt's samples."""
    rows = []
    for p in prompts:
        h = harmful.get(p["id"], np.zeros(0, dtype=bool))
        n = int(h.size)
        rows.append({"prompt_id": p["id"], "category": p["category"], "source": p["source"],
                     "n_samples": n, "n_harmful": int(h.sum()), "harm_rate": float(h.mean()) if n else np.nan,
                     "mean_score": float(scores[p["id"]].mean()) if scores and p["id"] in scores and n else np.nan,
                     "complete": n >= n_expected})
    return pd.DataFrame(rows)


def analytic_cracked(harm_rates: np.ndarray, k_values: list[int]) -> np.ndarray:
    """Mean over prompts of 1 - (1 - p)^k, p the observed harm rate: the with-replacement
    plug-in approximation. Kept as a CSV column only; it is biased low once k approaches the
    number of samples (a prompt with a single hit in 10,000 contributes 0.63 at k = 10,000,
    not 1) and it assigns exactly zero to prompts without a hit."""
    p = np.asarray(harm_rates, dtype=float)
    return np.array([np.mean(1.0 - (1.0 - p) ** k) for k in k_values])


def first_hit_positions(harmful: dict[int, np.ndarray], n_orderings: int, seed: int = 0) -> tuple[list[int], np.ndarray]:
    """Position of the first harmful sample under a random permutation of each prompt's samples
    (inf if it has none), for `n_orderings` permutations. Returns (prompt_ids, array[orderings, prompts])."""
    rng = np.random.default_rng(seed)
    ids = sorted(harmful)
    first = np.full((n_orderings, len(ids)), np.inf)
    for r in range(n_orderings):
        for i, pid in enumerate(ids):
            h = harmful[pid]
            if h.any():
                pos = rng.permutation(h.size)  # pos[j] = new position of sample j
                first[r, i] = pos[h].min()
    return ids, first


def cracked_indicator(first: np.ndarray, k_values: list[int]) -> np.ndarray:
    """array[orderings, prompts, k]: whether the first harmful sample lies among the first k."""
    ks = np.asarray(k_values)
    return first[:, :, None] < ks[None, None, :]


def empirical_cracked(harmful: dict[int, np.ndarray], k_values: list[int], n_orderings: int,
                      seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The cracked curve: fraction of prompts with >=1 harmful completion among the first k
    samples of a random ordering, averaged over `n_orderings` orderings. Returns (mean, std) over
    orderings. At k = n_samples this is exactly the fraction of prompts with any harmful completion."""
    _, first = first_hit_positions(harmful, n_orderings, seed)
    per_order = cracked_indicator(first, k_values).mean(axis=1)
    return per_order.mean(axis=0), per_order.std(axis=0)


def histogram_log_bins(harm_rates: np.ndarray, n_samples: int) -> pd.DataFrame:
    """Counts of prompts by harm rate: an exact-zero bin, then half-decade log bins from 1/n."""
    p = np.asarray(harm_rates, dtype=float)
    lo = np.floor(np.log10(1.0 / n_samples) * 2) / 2  # e.g. -4.0 for n=10,000
    edges = 10 ** np.arange(lo, 0.01, 0.5)
    edges[-1] = 1.0000001  # include p == 1
    rows = [{"bin_lo": 0.0, "bin_hi": 0.0, "label": "0", "n_prompts": int((p == 0).sum())}]
    nz = p[p > 0]
    counts, _ = np.histogram(nz, bins=edges)
    for a, b, c in zip(edges[:-1], edges[1:], counts):
        rows.append({"bin_lo": float(a), "bin_hi": float(min(b, 1.0)), "label": f"[{a:.1e}, {min(b, 1.0):.1e})",
                     "n_prompts": int(c)})
    # anything below the first edge (cannot happen when n_samples divides the grid, kept for safety)
    below = int((nz < edges[0]).sum())
    if below:
        rows.insert(1, {"bin_lo": 0.0, "bin_hi": float(edges[0]), "label": f"(0, {edges[0]:.1e})", "n_prompts": below})
    return pd.DataFrame(rows)


def per_category_table(df: pd.DataFrame, harmful: dict[int, np.ndarray], k_values: list[int],
                       n_orderings: int = 20, seed: int = 0) -> pd.DataFrame:
    """Per-category summary; the cracked_at_k columns are the same empirical curve restricted to
    the category's prompts (same orderings as the overall curve)."""
    ids, first = first_hit_positions(harmful, n_orderings, seed)
    per_prompt = pd.DataFrame(cracked_indicator(first, k_values).mean(axis=0), index=ids,
                              columns=[f"cracked_at_{k}" for k in k_values])
    rows = []
    for cat, g in df.groupby("category", sort=True):
        p = g["harm_rate"].to_numpy()
        row = {"category": cat, "n_prompts": len(g), "mean_harm_rate": float(np.mean(p)),
               "median_harm_rate": float(np.median(p)), "max_harm_rate": float(np.max(p)),
               "n_zero_harmful": int((g["n_harmful"] == 0).sum()),
               "frac_cracked_observed": float((g["n_harmful"] > 0).mean())}
        row.update(per_prompt.loc[g["prompt_id"].to_numpy()].mean(axis=0).to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- loading

def load_scores(cfg: dict) -> tuple[dict[int, np.ndarray], list[dict]]:
    """Per-prompt score arrays (ordered by sample_idx) from every scores shard on disk."""
    d = run_dir(cfg) / "scores"
    rows = []
    metas = []
    for f in sorted(glob.glob(str(d / "shard_*.jsonl"))):
        rows += read_jsonl(f)
    for f in sorted(glob.glob(str(d / "shard_*.meta.json"))):
        metas.append(json.loads(Path(f).read_text()))
    by_prompt: dict[int, dict[int, float]] = defaultdict(dict)
    for (pid, idx), r in dedupe(rows).items():
        by_prompt[pid][idx] = float(r["score"])
    out = {}
    for pid, m in by_prompt.items():
        idxs = sorted(m)
        out[pid] = np.array([m[i] for i in idxs], dtype=float)
    return out, metas


def completion_stats(cfg: dict) -> tuple[dict, list[dict]]:
    """Length / truncation statistics and the per-shard generation metadata."""
    d = run_dir(cfg) / "completions"
    n_tok, n_len, n = 0, 0, 0
    for f in sorted(glob.glob(str(d / "shard_*.jsonl"))):
        for r in read_jsonl(f):
            n += 1
            n_tok += int(r.get("n_tokens", 0))
            n_len += r.get("finish_reason") == "length"
    metas = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(d / "shard_*.meta.json")))]
    stats = {"n_completions": n, "mean_tokens": n_tok / n if n else None, "frac_truncated_at_max_tokens": n_len / n if n else None}
    return stats, metas


# ----------------------------------------------------------------------------- plots

def plot_cracked(curve: pd.DataFrame, ref: dict | None, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.plot(curve["k"], curve["empirical_mean"], "o-", color="#1b6ca8", label="Qwen2.5-1.5B-Instruct (this run)")
    ax.fill_between(curve["k"], curve["empirical_mean"] - curve["empirical_std"],
                    curve["empirical_mean"] + curve["empirical_std"], color="#1b6ca8", alpha=0.2, lw=0)
    if ref:
        ks = sorted(int(k) for k in ref["cracked"])
        ax.plot(ks, [ref["cracked"][k] if k in ref["cracked"] else ref["cracked"][str(k)] for k in ks], "s:",
                color="#d1495b", label=f"{ref['model'].split('/')[-1]} (paper)")
    ax.set_xscale("log")
    ax.set_xlabel("number of samples per prompt, k")
    ax.set_ylabel("fraction of prompts with ≥1 harmful completion")
    ax.set_ylim(0, 1)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_histogram(hist: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    x = np.arange(len(hist))
    ax.bar(x, hist["n_prompts"], color=["#888888"] + ["#1b6ca8"] * (len(hist) - 1))
    ax.set_xticks(x)
    ax.set_xticklabels(hist["label"], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("per-prompt harm rate (fraction of 10,000 samples judged harmful)")
    ax.set_ylabel("number of prompts")
    ax.set_title(title, fontsize=10)
    for i, v in enumerate(hist["n_prompts"]):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_categories(cat: pd.DataFrame, k_values: list[int], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = [k for k in (1, 100, 1000, 10000) if k in k_values]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    w = 0.8 / len(show)
    x = np.arange(len(cat))
    colors = ["#c9d6df", "#7fb3d5", "#1b6ca8", "#0b3c5d"]
    for i, k in enumerate(show):
        ax.bar(x + (i - (len(show) - 1) / 2) * w, cat[f"cracked_at_{k}"], w, label=f"k = {k:,}", color=colors[i % 4])
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" and ", " &\n") for c in cat["category"]], fontsize=7)
    ax.set_ylabel("fraction of prompts cracked")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title("Prompts with ≥1 harmful completion among the first k samples, by category", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=str(REPO_ROOT / "results"))
    ap.add_argument("--reference", default=str(REPO_ROOT / "configs" / "reference_llama31_8b.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    acfg = cfg.get("analysis", {})
    k_values = [k for k in acfg.get("k_values", DEFAULT_K) if k <= cfg["n_samples"]]
    thr = float(cfg["judge"]["threshold"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(cfg)
    scores, judge_metas = load_scores(cfg)
    if not scores:
        raise SystemExit("no scores found")
    harmful = {pid: s >= thr for pid, s in scores.items()}
    scored_prompts = [p for p in prompts if p["id"] in scores]
    missing = [p["id"] for p in prompts if p["id"] not in scores]
    df = per_prompt_table(scored_prompts, harmful, cfg["n_samples"], scores)
    incomplete = df[~df["complete"]]
    if missing or len(incomplete):
        print(f"WARNING: {len(missing)} prompts have no scores, {len(incomplete)} are incomplete; "
              f"results below use what exists.")
    df.to_csv(out / "per_prompt.csv", index=False)

    rates = df["harm_rate"].to_numpy()
    ana = analytic_cracked(rates, k_values)
    emp_m, emp_s = empirical_cracked(harmful, k_values, int(acfg.get("n_orderings", 20)), int(acfg.get("ordering_seed", 0)))
    curve = pd.DataFrame({"k": k_values, "empirical_mean": emp_m, "empirical_std": emp_s, "analytic": ana,
                          "n_prompts": len(df)})
    curve.to_csv(out / "cracked_curve.csv", index=False)

    hist = histogram_log_bins(rates, cfg["n_samples"])
    hist.to_csv(out / "harm_rate_histogram.csv", index=False)
    cat = per_category_table(df, harmful, k_values, int(acfg.get("n_orderings", 20)), int(acfg.get("ordering_seed", 0)))
    cat.to_csv(out / "per_category.csv", index=False)

    ref = None
    if args.reference and Path(args.reference).exists():
        import yaml
        ref = yaml.safe_load(open(args.reference))
        ref["cracked"] = {int(k): float(v) for k, v in ref["cracked"].items()}

    all_scores = np.concatenate(list(scores.values()))
    comp_stats, gen_metas = completion_stats(cfg)
    summary = {
        "model": cfg["model"], "n_prompts_scored": int(len(df)), "n_prompts_total": len(prompts),
        "n_samples_per_prompt": int(cfg["n_samples"]), "n_completions_scored": int(all_scores.size),
        "harmful_threshold": thr,
        "overall_harm_rate": float((all_scores >= thr).mean()), "mean_judge_score": float(all_scores.mean()),
        "n_prompts_zero_harmful": int((df["n_harmful"] == 0).sum()),
        "n_prompts_any_harmful": int((df["n_harmful"] > 0).sum()),
        "median_harm_rate": float(np.median(rates)), "mean_harm_rate": float(np.mean(rates)),
        "cracked_empirical": {int(k): float(v) for k, v in zip(k_values, emp_m)},
        "cracked_empirical_std": {int(k): float(v) for k, v in zip(k_values, emp_s)},
        "cracked_analytic_with_replacement": {int(k): float(v) for k, v in zip(k_values, ana)},
        "judge_score_distribution": {f">={t}": float((all_scores >= t).mean()) for t in (0.25, 0.5, 0.75, 1.0)},
        "completions": comp_stats,
        "incomplete_prompt_ids": [int(x) for x in incomplete["prompt_id"]], "unscored_prompt_ids": missing,
        "reference": ref,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    def pick(metas, keys):
        vals = {}
        for m in metas:
            for k in keys:
                v = m.get(k) if k in m else m.get("env", {}).get(k)
                if v is not None:
                    vals.setdefault(k, set()).add(json.dumps(v, sort_keys=True))
        return {k: [json.loads(x) for x in sorted(v)] for k, v in vals.items()}

    metadata = {
        "experiment": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "seed_scheme": "per-request vLLM seed = (seed * 1000003 + prompt_id * 1009 + chunk_index) mod (2^31 - 1); "
                       "engine seed = seed; analysis ordering seed = analysis.ordering_seed",
        "generation": {"n_shards": len(gen_metas),
                       **pick(gen_metas, ["vllm", "torch", "transformers", "gpu", "rendered_prompt_example"]),
                       "total_generate_seconds": sum(m.get("generate_seconds", 0) for m in gen_metas),
                       "tokens_per_second_per_shard": [m.get("tokens_per_second") for m in gen_metas],
                       "gpu_memory_used_mib": [m.get("gpu_memory_used_mib") for m in gen_metas]},
        "judging": {"n_shards": len(judge_metas), **pick(judge_metas, ["transformers", "peft", "torch", "gpu"]),
                    "total_judge_seconds": sum(m.get("judge_seconds", 0) for m in judge_metas),
                    "items_per_second_per_shard": [m.get("items_per_second") for m in judge_metas],
                    "gpu_peak_memory_mib": [m.get("gpu_peak_memory_mib") for m in judge_metas],
                    "selfcheck": [m["selfcheck"] for m in judge_metas if "selfcheck" in m]},
        "strong_reject_package": "git+https://github.com/dsbowen/strong_reject.git@7a551d5b440ec7b75d4f6f5bb7c1719965b76b47",
        "prompt_ids": [p["id"] for p in prompts],
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=1))

    short = cfg["model"].split("/")[-1]
    plot_cracked(curve, ref, out / "cracked_curve.png", f"{short}: StrongREJECT prompts cracked vs samples (n={len(df)} prompts)")
    plot_histogram(hist, out / "harm_rate_histogram.png", f"{short}: per-prompt harm rate, {cfg['n_samples']:,} samples each")
    plot_categories(cat, k_values, out / "per_category.png")

    print(json.dumps({k: summary[k] for k in ("n_prompts_scored", "overall_harm_rate", "n_prompts_zero_harmful",
                                               "cracked_empirical")}, indent=1))
    print(cat[["category", "n_prompts", "mean_harm_rate", "n_zero_harmful", "frac_cracked_observed"]].to_string(index=False))


if __name__ == "__main__":
    main()
