"""Regenerate the numeric tables of the README's Results section from the files in results/.

    python scripts/render_readme_results.py

Rewrites, in place, the three tables (cracked curve, harm-rate histogram, per category) and the
headline sentences that quote numbers, so the README never drifts from results/*.csv and
results/summary.json. Prose that carries no numbers is left untouched.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> None:
    s = (ROOT / "README.md").read_text()
    summ = json.loads((R / "summary.json").read_text())
    cur = pd.read_csv(R / "cracked_curve.csv").set_index("k")
    ref = pd.read_csv(R / "reference_llama31_8b_curve.csv").set_index("k")
    refsum = json.loads((R / "reference_llama31_8b_summary.json").read_text())
    hist = pd.read_csv(R / "harm_rate_histogram.csv")
    cat = pd.read_csv(R / "per_category.csv")
    pp = pd.read_csv(R / "per_prompt.csv")
    n = summ["n_prompts_scored"]
    n_zero = summ["n_prompts_zero_harmful"]
    n_any = summ["n_prompts_any_harmful"]

    # 1. cracked-curve table
    rows = "\n".join(f"| {k:,} | {pct(cur.loc[k, 'empirical_mean'])} | {pct(cur.loc[k, 'analytic'])} | "
                     f"{pct(ref.loc[k, 'empirical_mean'])} | {pct(ref.loc[k, 'analytic'])} |" for k in cur.index)
    s = re.sub(r"(\| k \| Qwen2\.5-1\.5B empirical \|.*?\n\|---:\|---:\|---:\|---:\|---:\|\n)(?:\|.*\n)+",
               lambda m: m.group(1) + rows + "\n", s)

    # 2. headline sentences
    s = re.sub(r"Prompts with zero harmful completions in 10,000: \*\*\d+ of \d+\*\* \(\d+%\), against \d+ of \d+ for",
               f"Prompts with zero harmful completions in 10,000: **{n_zero} of {n}** ({n_zero / n * 100:.0f}%), against "
               f"{refsum['n_zero_harmful']} of {refsum['n_prompts']} for", s)
    s = re.sub(r"after 1,000 samples it is still climbing \([\d.]+% to [\d.]+%\)",
               f"after 1,000 samples it is still climbing ({pct(cur.loc[1000, 'empirical_mean'])} to {pct(cur.loc[10000, 'empirical_mean'])})", s)
    tot = summ["n_completions_scored"]
    n_h = round(summ["overall_harm_rate"] * tot)
    sd = summ["judge_score_distribution"]
    s = re.sub(r"Overall, [\d.]+% of completions are harmful \([\d,]+ of [\d,]+\); [\d.]+% score >= 0\.25 and [\d.]+%\nscore >= 0\.5",
               f"Overall, {summ['overall_harm_rate'] * 100:.2f}% of completions are harmful ({n_h:,} of {tot:,}); "
               f"{sd['>=0.25'] * 100:.1f}% score >= 0.25 and {sd['>=0.5'] * 100:.1f}%\nscore >= 0.5", s)

    # 3. histogram table
    labels = {"0": "0 (no harmful completion in 10,000)"}
    hrows = []
    for _, r in hist.iterrows():
        lab = labels.get(r["label"], r["label"].replace("e-0", "e-").replace("1.0e-1", "0.1").replace("3.2e-1", "0.32").replace("1.0e+00", "1"))
        if r["label"].startswith("[3.2e-01"):
            lab = ">= 0.32"
        hrows.append(f"| {lab} | {int(r['n_prompts'])} |")
    s = re.sub(r"(\| per-prompt harm rate \| prompts \|\n\|---\|---:\|\n)(?:\|.*\n)+", lambda m: m.group(1) + "\n".join(hrows) + "\n", s)
    one = int((pp["n_harmful"] == 1).sum())
    many = int((pp["n_harmful"] >= 100).sum())
    top = pp.sort_values("n_harmful", ascending=False).iloc[0]
    over10 = pp[pp["harm_rate"] > 0.1]
    cats_over10 = over10["category"].value_counts()
    s = re.sub(r"(?:Five|\w+|\d+) prompts produced exactly one harmful completion in 10,000, and \d+ prompts produced 100 or\nmore\. The highest per-prompt rate is [\d.]+% \(an? \"[^\"]+\" prompt\); .*?\n.*?exceed 10%\.",
               f"{one} prompts produced exactly one harmful completion in 10,000, and {many} prompts produced 100 or\n"
               f"more. The highest per-prompt rate is {top['harm_rate'] * 100:.1f}% (a \"{top['category']}\" prompt); "
               f"{len(over10)} prompts exceed 10%, {cats_over10.iloc[0]} of them in \"{cats_over10.index[0]}\".", s, flags=re.S)

    # 4. per-category table
    crows = "\n".join(f"| {r['category']} | {int(r['n_prompts'])} | {int(r['n_zero_harmful'])} | {pct(r['cracked_at_1_analytic'])} | "
                      f"{pct(r['cracked_at_100_analytic'])} | {pct(r['cracked_at_1000_analytic'])} | {pct(r['cracked_at_10000_analytic'])} |"
                      for _, r in cat.iterrows())
    s = re.sub(r"(\| category \| prompts \| zero harmful \| cracked at k=1 \| k=100 \| k=1,000 \| k=10,000 \|\n\|---\|---:\|---:\|---:\|---:\|---:\|---:\|\n)(?:\|.*\n)+",
               lambda m: m.group(1) + crows + "\n", s)
    nv = cat.set_index("category").loc["Non-violent crimes"]
    s = re.sub(r"drugs, theft\) is the weak spot: \d+% of its prompts are cracked within 10,000 samples and half\nwithin 1,000\.",
               f"drugs, theft) is the weak spot: {nv['cracked_at_10000_analytic'] * 100:.0f}% of its prompts are cracked within 10,000 samples and "
               f"{nv['cracked_at_1000_analytic'] * 100:.0f}%\nwithin 1,000.", s)
    (ROOT / "README.md").write_text(s)
    print("README results refreshed:", n_zero, "zero-harmful;", pct(cur.loc[10000, "empirical_mean"]), "cracked at 10k")


if __name__ == "__main__":
    main()
