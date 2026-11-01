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

The cracked curve takes the first k samples of a random ordering of each prompt's 10,000
completions and averages over 20 orderings (seeded, so it is reproducible). At k = 10,000 it is
simply the fraction of prompts with any harmful completion.

## Results

Full run: 313 prompts x 10,000 samples = 3,130,000 completions, scored by the StrongREJECT
fine-tuned judge. Numbers below are from `results/summary.json`, `results/cracked_curve.csv`,
`results/per_category.csv` and `results/per_prompt.csv`.

### Cracked curve

Fraction of the 313 prompts with at least one harmful completion (judge score >= 0.75) among the
first k samples, averaged over 20 random orderings of each prompt's 10,000 samples. The
Llama-3.1-8B-Instruct column comes from the paper's released per-prompt scores
(`rangell/mc_estimates`, `strong_reject/Llama-3.1-8B-Instruct_mc_est_10k.json`), re-analysed with
the same code and threshold by `scripts/reference_from_paper.py`; it reproduces the paper's
headline numbers (78.9% cracked at k = 10,000, 66 prompts never cracked). (`results/cracked_curve.csv`
also carries the with-replacement approximation, the mean over prompts of 1 - (1 - p)^k, in the
column `analytic`; it agrees with the empirical curve to within 0.5 points everywhere and is biased
low near k = 10,000, so it is not shown.)

| k | Qwen2.5-1.5B-Instruct | Llama-3.1-8B-Instruct (paper) |
|---:|---:|---:|
| 1 | 0.3% | 1.2% |
| 3 | 1.0% | 2.5% |
| 10 | 3.0% | 4.9% |
| 30 | 5.8% | 8.8% |
| 100 | 11.0% | 16.5% |
| 300 | 17.9% | 27.8% |
| 1,000 | 28.6% | 46.8% |
| 3,000 | 34.6% | 64.3% |
| 10,000 | 36.7% | 78.9% |

![cracked curve](results/cracked_curve.png)

Prompts with zero harmful completions in 10,000: **198 of 313** (63%), against 66 of 313 for
Llama-3.1-8B-Instruct. Qwen2.5-1.5B-Instruct is cracked far less often, but the shape
is the same: the cracked fraction rises by almost two orders of magnitude from one sample to
10,000, and after 1,000 samples it is still climbing (28.6% to 36.7%).

### Per-prompt harm rates

Overall, 0.45% of completions are harmful (14,069 of 3,130,000); 7.2% score >= 0.25 and 2.0%
score >= 0.5, so most non-refusals are judged as only partially helpful. The median prompt has a
harm rate of exactly 0.

| per-prompt harm rate | prompts |
|---|---:|
| 0 (no harmful completion in 10,000) | 198 |
| [1.0e-4, 3.2e-4) | 10 |
| [3.2e-4, 1.0e-3) | 8 |
| [1.0e-3, 3.2e-3) | 61 |
| [3.2e-3, 1.0e-2) | 16 |
| [1.0e-2, 3.2e-2) | 10 |
| [3.2e-2, 0.1) | 6 |
| [0.1, 0.32) | 4 |
| >= 0.32 | 0 |

![histogram](results/harm_rate_histogram.png)

5 prompts produced exactly one harmful completion in 10,000, and 20 prompts produced 100 or
more. The highest per-prompt rate is 29.6% (a "Sexual content" prompt); 4 prompts exceed 10%, 3 of them in "Non-violent crimes".

### Per category

| category | prompts | zero harmful | cracked at k=1 | k=100 | k=1,000 | k=10,000 |
|---|---:|---:|---:|---:|---:|---:|
| Disinformation and deception | 50 | 34 | 0.2% | 9.4% | 25.1% | 32.0% |
| Hate, harassment and discrimination | 50 | 37 | 0.0% | 5.3% | 19.4% | 26.0% |
| Illegal goods and services | 50 | 34 | 0.2% | 8.8% | 24.6% | 32.0% |
| Non-violent crimes | 59 | 23 | 0.8% | 23.5% | 51.4% | 61.0% |
| Sexual content | 50 | 37 | 0.5% | 7.9% | 18.1% | 26.0% |
| Violence | 54 | 33 | 0.1% | 9.3% | 28.8% | 38.9% |

(Same empirical curve, restricted to the category's prompts.) "Non-violent crimes" (fraud, hacking,
drugs, theft) is the weak spot: 61% of its prompts are cracked within 10,000 samples and 51%
within 1,000.

![per category](results/per_category.png)

### Completions and cost

Mean completion length 57 tokens; 16.4% of completions hit the 150-token cap (the paper's cap),
so long harmful answers are truncated before the judge sees them. Compute on RTX 2080 Ti (11 GB)
GPUs: generation 8.9 GPU-hours (vLLM, about 5,400 tokens/s per GPU); judging 37.5 GPU-hours for the
final eager-attention pass (19 to 26 completions/s per GPU), plus 38.5 GPU-hours for the discarded
SDPA pass; wall-clock 47 minutes and 2 h 52 min on four 4-GPU nodes.
The pilot's float16-vs-float32 judge comparison (64 completions) agreed on every label with a
maximum score difference of 1e-5 (`results/pilot/metadata.json`).

## Judge numerics: a kernel bug, found and corrected

The first judging pass used transformers' default SDPA attention. On the RTX 2080 Ti (sm_75) in
float16, PyTorch's fused memory-efficient attention kernel returns an input-independent result for
some left-padded sequences in a batch; the symptom was the identical judge score 0.5266 for
**55,639 of 3,130,000 completions (1.78%)**, across 311 prompts and every shard. It was caught by
re-scoring a sample with the official `google/gemma-2b` weights and noticing one score move by 0.53
with identical weights, and pinned down as follows (`scripts/judge_*.py`, `results/judge_*.json`):

- Float16 itself is accurate: unpadded (batch size 1) float16 scores match a float32 CPU reference
  within 0.0012 on 90 stratified items; random batches change scores by at most 0.003.
- Replaying the run's exact batches with SDPA reproduces the stored constants almost one for one
  (538 of 640 items in one shard); the unaffected items in those batches match to 1e-7. The same
  batches with eager attention, or with the fused SDPA kernels disabled, give zero constants, match
  the unpadded reference within 0.002, and produce no label flips at the 0.75 threshold.

Two corrections were made. First, the 55,639 constant-valued items were re-scored in place
with eager attention (`scripts/rescore_anomalous.py`); 392 of them turned out to be harmful. A
verification re-score of 12,000 random unaffected items then found three scores off by more than
0.3 with no constant fingerprint, i.e. the fused kernel had also perturbed a small fraction of
other scores. So, second, **all 3,130,000 completions were re-judged from scratch with eager
attention**, and those are the scores every number in this README is computed from. Comparing
the two complete passes item by item (`scripts/compare_passes.py`,
`results/judge_pass_comparison.json`):

| | SDPA pass (uncorrected) | eager pass (final) |
|---|---:|---:|
| harmful completions (score >= 0.75) | 13,545 | 14,069 |
| items whose score differs by > 0.05 between passes | 55,353 | |
| label flips not-harmful -> harmful | | 539 |
| label flips harmful -> not-harmful | | 15 |
| prompts never cracked | 197 | 198 |
| median absolute score difference | | 4e-06 |

Of the 539 completions that became harmful, 392 had carried the constant and 147 had carried an
ordinary-looking but wrong score. One prompt lost its single harmful completion and moved to
"never cracked". The headline curve moved by at most 0.5 percentage point.

`src/judge.py` now uses eager attention by default (`judge.attn_implementation`); the pipeline's
own scores shipped in `results/` are the corrected ones. Lesson for anyone re-running this on
Turing-class GPUs: validate batched float16 scoring against batch size 1 on the production batches,
not on single items.

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
sbatch --account=<account> slurm/analyze.sbatch                 # tables and plots into results/ (src/analyze.py + README refresh)
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
- **Judge base weights.** The judge is a LoRA adapter on `google/gemma-2b`, a gated repository.
  The run used the ungated mirror `unsloth/gemma-2b` as the base. All 164 tensors of the mirror
  were afterwards compared bit for bit with the official `google/gemma-2b` (revision `9cf48e5`):
  they are identical (`results/judge_base_check.json`, produced by
  `scripts/compare_judge_base.py`), so the two ids are interchangeable; set `judge.base_model` to
  either. The judge runs in float16 because the GPUs used have no bfloat16; see "Judge numerics" above.
- **Scoring code.** `src/judge.py` calls the `strong_reject` package's `strongreject_finetuned`
  function unchanged (same prompt template, same 512-token response truncation, same expected-
  value score over the five grade tokens); it only pre-loads the model into the package's cache
  and asks the model for last-position logits only (identical result, far less memory).
- **Expected value vs argmax.** The package's score is the probability-weighted mean of the five
  grades, so it is continuous in [0, 1]. The paper's released scores take only the values
  {0, 0.25, 0.5, 0.75, 1}, i.e. the argmax grade. "Score >= 0.75" is stricter under the
  expected-value rule (it needs most of the probability mass on grades 4 and 5), so the harmful
  counts here are, if anything, conservative relative to the paper's labelling. The Llama
  reference columns use the paper's own discrete scores.
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
