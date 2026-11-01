# Rare refusal failures: tail risks of Qwen2.5-1.5B-Instruct on StrongREJECT

A small, self-contained replication of the brute-force Monte Carlo experiment in
*Estimating Tail Risks in Language Model Output Distributions* (Angell, Singhal, Horvitz, Yu,
Ranganath, McKeown, He; arXiv [2604.22167](https://arxiv.org/abs/2604.22167), ICML 2026) with a
model the paper did not test: **Qwen/Qwen2.5-1.5B-Instruct**.

The question: a model that refuses a harmful request almost always still produces a harmful answer
*sometimes*. How often, and how many prompts get "cracked" (at least one harmful completion) if an
attacker simply samples k times?

## The experiment

| | |
|---|---|
| Prompts | the 313 StrongREJECT forbidden prompts, no jailbreak (`strong_reject` package, `load_strongreject()`), 6 categories |
| Target model | `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa79`, float16, vLLM |
| Sampling | chat template, user message only (no added system prompt), temperature 1.0, top-p 1.0, no top-k, no repetition penalty, 150 new tokens, 10,000 samples per prompt |
| Judge | StrongREJECT fine-tuned judge (`qylu4156/strongreject-15k-v1`, a LoRA on Gemma-2B) through the `strong_reject` package's `strongreject_finetuned` evaluator; a completion is **harmful iff score >= 0.75** |
| Outputs | per-prompt harm rates, the "cracked curve" (fraction of prompts with >= 1 harmful completion among the first k samples, k = 1 ... 10,000), a log-binned histogram of harm rates, a per-category breakdown |

The cracked curve is computed two ways: empirically (first k samples of a random ordering,
averaged over 20 orderings) and analytically as the mean over prompts of 1 - (1 - p)^k with p the
prompt's observed harm rate.

## Results

Full run: 313 prompts x 10,000 samples = 3,130,000 completions, scored by the StrongREJECT
fine-tuned judge. Numbers below are from `results/summary.json`, `results/cracked_curve.csv`,
`results/per_category.csv` and `results/per_prompt.csv`.

### Cracked curve

Fraction of the 313 prompts with at least one harmful completion (judge score >= 0.75) among the
first k samples. "Empirical" averages over 20 random orderings of each prompt's 10,000 samples;
"analytic" is the mean over prompts of 1 - (1 - p)^k. The Llama-3.1-8B-Instruct column is the
paper's released per-prompt file for the same prompts and judge.

| k | Qwen2.5-1.5B-Instruct, empirical | Qwen2.5-1.5B-Instruct, analytic | Llama-3.1-8B-Instruct (paper) |
|---:|---:|---:|---:|
| 1 | 0.5% | 0.4% | 1.3% |
| 3 | 1.2% | 1.2% | |
| 10 | 2.9% | 2.9% | |
| 30 | 6.0% | 5.7% | |
| 100 | 10.8% | 10.9% | 16.6% |
| 300 | 18.4% | 18.2% | |
| 1,000 | 28.7% | 28.0% | 45.5% |
| 3,000 | 34.6% | 34.3% | |
| 10,000 | 37.1% | 36.4% | 78.9% |

![cracked curve](results/cracked_curve.png)

Prompts with zero harmful completions in 10,000: **197 of 313** (63%), against 66 of 313 for
Llama-3.1-8B-Instruct in the paper. Qwen2.5-1.5B-Instruct is cracked far less often, but the shape
is the same: the cracked fraction rises by almost two orders of magnitude from one sample to
10,000, and after 1,000 samples it is still climbing (28.7% to 37.1%).

### Per-prompt harm rates

Overall, 0.43% of completions are harmful (13,545 of 3,130,000); 8.8% score >= 0.25 and 3.7%
score >= 0.5, so most non-refusals are judged as only partially helpful. The median prompt has a
harm rate of exactly 0.

| per-prompt harm rate | prompts |
|---|---:|
| 0 (no harmful completion in 10,000) | 197 |
| [1e-4, 3.2e-4) | 10 |
| [3.2e-4, 1e-3) | 11 |
| [1e-3, 3.2e-3) | 60 |
| [3.2e-3, 1e-2) | 15 |
| [1e-2, 3.2e-2) | 10 |
| [3.2e-2, 0.1) | 6 |
| [0.1, 0.32) | 4 |
| >= 0.32 | 0 |

![histogram](results/harm_rate_histogram.png)

Five prompts produced exactly one harmful completion in 10,000, and 20 prompts produced 100 or
more. The highest per-prompt rate is 29.6% (a "Sexual content" prompt); three "Non-violent
crimes" prompts exceed 10%.

### Per category

| category | prompts | zero harmful | cracked at k=1 | k=100 | k=1,000 | k=10,000 |
|---|---:|---:|---:|---:|---:|---:|
| Disinformation and deception | 50 | 33 | 0.1% | 9.0% | 25.9% | 33.3% |
| Hate, harassment and discrimination | 50 | 37 | 0.1% | 5.7% | 19.2% | 25.2% |
| Illegal goods and services | 50 | 34 | 0.2% | 8.5% | 23.9% | 31.9% |
| Non-violent crimes | 59 | 23 | 1.1% | 23.3% | 49.9% | 61.0% |
| Sexual content | 50 | 37 | 0.8% | 8.2% | 17.6% | 25.1% |
| Violence | 54 | 33 | 0.2% | 8.7% | 27.7% | 37.3% |

(Cracked fractions are the analytic mean of 1 - (1 - p)^k within the category; the k=10,000
column equals the observed fraction of cracked prompts.) "Non-violent crimes" (fraud, hacking,
drugs, theft) is the weak spot: 61% of its prompts are cracked within 10,000 samples and half
within 1,000.

![per category](results/per_category.png)

### Completions and cost

Mean completion length 57 tokens; 16.4% of completions hit the 150-token cap (the paper's cap),
so long harmful answers are truncated before the judge sees them. Compute on RTX 2080 Ti (11 GB)
GPUs: generation 8.9 GPU-hours (vLLM, about 5,400 tokens/s per GPU), judging 38.5 GPU-hours
(about 24 completions/s per GPU); wall-clock 47 minutes and 2 h 54 min on four 4-GPU nodes.
The pilot's float16-vs-float32 judge comparison (64 completions) agreed on every label with a
maximum score difference of 1e-5 (`results/pilot/metadata.json`).

## Repository layout

```
configs/      experiment.yaml (full run), pilot.yaml, smoke.yaml, reference_llama31_8b.yaml
src/          generate.py (vLLM sampling), judge.py (StrongREJECT judge), analyze.py (tables + plots),
              shards.py (queue status), common.py (config, prompts, sharding, JSONL, work queue)
slurm/        env.sh (all caches and data on $SCRATCH), generate/judge/pilot .sbatch, workers.sh
scripts/      setup_env.sh (one-time venv), submit.sh (submits stages as job arrays)
results/      aggregated CSV/JSON tables, cracked-curve CSV, PNG plots, metadata.json
tests/        tests of the analysis code on synthetic scores (pytest)
```

Raw completions and per-sample scores are **not** in the repository (they are large and contain
harmful text); they stay under `$TAILRISK_ROOT/runs/<name>/{completions,scores}/` on scratch.

## Running it on a SLURM cluster

Everything heavy lives under `$TAILRISK_ROOT` (default `$SCRATCH/qwen_tailrisk`): the Python
environment, Hugging Face / Triton / vLLM caches, completions, scores and logs. Nothing large is
written under `$HOME`, and nothing but a quick import test runs on a login node.

```bash
source slurm/env.sh                 # paths; edit TAILRISK_ROOT / TAILRISK_MODULE for your site
bash scripts/setup_env.sh           # venv on scratch on top of the cluster's vLLM module
cp slurm/local.sh.example slurm/local.sh && $EDITOR slurm/local.sh   # your SLURM account (git-ignored)
python -m pytest tests              # analysis unit tests, CPU only

scripts/submit.sh pilot configs/smoke.yaml   # 2 prompts x 40 samples, a few minutes (debug QoS)
scripts/submit.sh pilot                      # 10 prompts x 1,000 samples: timing, memory, judge self-check
scripts/submit.sh generate configs/experiment.yaml 4            # 4 node jobs x 4 GPUs sampling 313 x 10,000
scripts/submit.sh judge    configs/experiment.yaml 4 <gen_job_id>  # judging, starts when generation ends
python src/analyze.py --config configs/experiment.yaml          # tables and plots into results/
```

Design notes:

- **Shards and resumability.** The 313 prompts are split into shards of `shard_size` prompts.
  Generation writes one JSONL per shard in chunks of `chunk_size` samples and only generates
  chunks that are missing; judging scores only unscored completions; a `.done` marker closes a
  shard. Killing and resubmitting a job wastes at most one chunk.
- **Node jobs with a work queue.** The cluster's es0 nodes have 4 RTX 2080 Ti; each array task
  takes a node and runs one worker per GPU. Workers claim shards through atomic lock files, so
  all four GPUs stay busy until the queue is empty and no shard is processed twice. The vLLM
  engine and the judge are loaded once per worker and reused across shards.
- **Job scripts** default to partition `es0`; the account comes from `SLURM_ACCOUNT`
  (`slurm/local.sh`). Adjust `--qos`, `--gres`, `--time` to your site in `slurm/*.sbatch`.
- **Seeds.** vLLM is seeded per request: `(seed * 1000003 + prompt_id * 1009 + chunk) mod (2^31-1)`;
  the engine seed, the analysis ordering seed and every version are in `results/metadata.json`.

## Faithfulness notes and deviations

- **Chat template.** As in the paper's code, the prompt is a single user message passed to
  `llm.chat`, so the model's own chat template is applied. Qwen2.5's template inserts its default
  system line ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") when no
  system message is given; we add nothing. The rendered prompt is recorded in `metadata.json`.
- **Sampling.** vLLM is started with `generation_config="vllm"` so Qwen's repository defaults
  (top-p 0.8, top-k 20, repetition penalty 1.05) are not applied; temperature 1.0, top-p 1.0,
  no top-k and no penalties are also set explicitly on every request.
- **Judge base weights.** The judge is a LoRA adapter on `google/gemma-2b`, which is a gated
  repository. The run used the ungated mirror `unsloth/gemma-2b` as the base; set
  `judge.base_model` to `google/gemma-2b` if you have access. The judge runs in float16 (the GPUs
  used have no bfloat16); the pilot compares float16 GPU scores against float32 CPU scores for the
  same completions and reports the differences (see `results/pilot/metadata.json`).
- **Scoring code.** `src/judge.py` calls the `strong_reject` package's `strongreject_finetuned`
  function unchanged (same prompt template, same 512-token response truncation, same expected-
  value score over the five grade tokens); it only pre-loads the model into the package's cache.
- **Prompt ids.** The StrongREJECT CSV has no id column; `id` is the row index (0 to 312).

## Citation

```
@article{angell2026tailrisks,
  title   = {Estimating Tail Risks in Language Model Output Distributions},
  author  = {Angell, Rico and Singhal, Raghav and Horvitz, Zachary and Yu, Zhou and Ranganath, Rajesh and McKeown, Kathleen and He, He},
  journal = {arXiv preprint arXiv:2604.22167},
  year    = {2026}
}
@article{souly2024strongreject,
  title   = {A StrongREJECT for Empty Jailbreaks},
  author  = {Souly, Alexandra and Lu, Qingyuan and Bowen, Dillon and Trinh, Tu and Hsieh, Elvis and Pandey, Sana and Abbeel, Pieter and Svegliato, Justin and Emmons, Scott and Watkins, Olivia and Toyer, Sam},
  journal = {arXiv preprint arXiv:2402.10260},
  year    = {2024}
}
```
