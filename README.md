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

RESULTS_PLACEHOLDER

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
