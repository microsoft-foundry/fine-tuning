# Benchmarks

Historical end-to-end measurements of Interactive Training, TRL, and/or
verl for the original dataset, base model, and recipe configuration. These
are recorded reference results, not newly reproduced runs, current preview
eligibility, expected Qwen3.8 quality, or Azure price estimates. Wall-clock time
is not a monetary cost or GPU-hour comparison.

> [!IMPORTANT]
> Preserve each result's original model, metric, configuration, and partial-run
> status. Infrastructure and defaults can change; do not predict that timing
> will improve or relabel old results for a new model. In particular,
> [current model list](./supported_models.md#available-today)
> excludes Qwen3-32B from new interactive training; its historical results below
> do not authorize recreating those runs.

## Setup

Held constant across **every** row below:

- **Optimizer:** Adam (β₁=0.9, β₂=0.95, ε=1e-8)

Per-run hyperparameters (learning rate, batch size, max sequence length,
hardware, seed, etc.) live in each recipe's README. Today the cookbook ships
[recipes/math_rl](../interactive_training/recipes/math_rl/README.md) (GSM8K),
[recipes/tool_rl](../interactive_training/recipes/tool_rl/README.md) (xLAM-60k),
[recipes/code_rl](../interactive_training/recipes/code_rl/README.md)
(DeepCoder-Preview, sandbox-backed), and
[recipes/tulu3_sft](../interactive_training/recipes/tulu3_sft/README.md).

Variables: dataset, base model, framework, base accuracy, final accuracy, end-to-end wall time.
LoRA rank is 32 for RL recipes and 16 for SFT unless noted in the row.
Base model varies across rows; cross-framework comparisons hold the base model fixed within a row group.

> **What "E2E time" means.** Time from process launch (including model load / FSDP
> shard / vLLM warmup) to the final saved checkpoint. Not GPU-hours, not
> step-time — the number a practitioner actually waits for.

> **What "accuracy" means.** Per dataset, defined in each recipe's grader. The
> linked recipe is the source of truth for the metric.

> **Read [Caveats](#caveats) before comparing rows.** Short version: numbers
> aren't perfectly apples-to-apples across frameworks, and Interactive Training's numbers
> will move as our infra evolves.

## Results

### GSM8K

| Framework | Base model | Final acc |   E2E time | Notes |
| --------- | ---------- | --------: | ---------: | ----- |
| TRL       | Qwen3-32B  |    91.89% |    10.91 h | base 68.55% |
| verl      | Qwen3-32B  |    92.80% |     7.40 h | base 68.57% |
| **Interactive Training**  | Qwen3-32B  | **95.53%** | **8.35 h** | base 69.07%; [recipes/math_rl](../interactive_training/recipes/math_rl/README.md) |
| **Interactive Training**  | gpt-oss-20b | **93.71%** | **2.88 h** | base 93.63% — model is already at ceiling on GSM8K, so this is a sanity run, not a learning result. 30 steps, LoRA r=32. |

### xLAM-60k (tool calling)

| Framework | Base model | Final acc |   E2E time | Notes |
| --------- | ---------- | --------: | ---------: | ----- |
| TRL       | Qwen3-32B  |    81.77% |    42.05 h | base 60.02% |
| verl      | Qwen3-32B  |    82.90% |    30.30 h | base 60.76% |
| **Interactive Training**  | Qwen3-32B  | **83.33%** | **34.10 h** | base 59.51%; [recipes/tool_rl](../interactive_training/recipes/tool_rl/README.md) |

### Search-Tool (Search-R1 multi-hop QA)

RL on Search-R1-style retrieval-augmented QA. Trains on
`PeterJinGo/nq_hotpotqa_train` (NQ + HotpotQA): the model issues search queries
against a FAISS + E5 retrieval server to answer multi-hop questions, rewarded on
answer correctness plus an output-format term. Eval is a held-out 1,000-episode
mixture (NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle);
accuracy is `test/env/all/correct`.

Training teaches the model to **use the search tool more deliberately** instead
of answering from priors: more search calls (`turns_per_episode` 2.05 → 2.69),
longer queries (`ac_tokens_per_turn` ~4×, 42 → 175), and more retrieved context
per turn (`ob_tokens_per_turn` ~2×, 1117 → 2204). Format compliance is the early
win (`format` 0.90 → 0.99), and held-out accuracy rises broadly (TriviaQA
0.49 → 0.68, NQ 0.21 → 0.45, 2Wiki 0.31 → 0.53, HotpotQA 0.27 → 0.48); MuSiQue
(~0.19) is the laggard and the main headroom left.

| Framework | Base model | Final acc |   E2E time | Notes |
| --------- | ---------- | --------: | ---------: | ----- |
| TRL       | Qwen3-32B  |       TBD |        TBD | TBD |
| verl      | Qwen3-32B  |       TBD |        TBD | TBD |
| **Interactive Training**  | Qwen3-32B  | **54%** | **54.5 hrs** | base 33%; [recipes/search_tool](../interactive_training/recipes/search_tool/README.md) |

### DeepCoder / code_rl (LiveCodeBench)

RL on competitive-programming problems. Training mixture is the DeepCoder blend
(PrimeIntellect + TACO + LCBv5); the eval mixture is Codeforces + LCBv5. Rewards
come from executing generated code in a sandbox (SandboxFusion or Modal) and
checking unit-test pass rate against LiveCodeBench-style judges. In practice,
getting either run to make it this far without the code-execution engine
hanging was the hard part — both rows are partial runs, and wall-clock between
them isn't directly comparable.

| Framework | Base model | Final acc |   E2E time | Notes |
| --------- | ---------- | --------: | ---------: | ----- |
| verl      | Qwen3-32B  |    19.36% |     9.07 h | base 9.46%; 20% of training completed |
| **Interactive Training**  | Qwen3-32B  | **20.24%** | **13.45 h** | base 12.65%; 21% of training completed; [recipes/code_rl](../interactive_training/recipes/code_rl/README.md) |

### Tulu3 SFT

SFT, so the metric is held-out negative log-likelihood (lower is better), not
accuracy.

| Framework | Base model | Final NLL |   E2E time | Notes |
| --------- | ---------- | --------: | ---------: | ----- |
| TRL       | Qwen3-32B  |      0.53 |     8.68 h | base 0.86 |
| **Interactive Training**  | Qwen3-32B  |  **0.48** |  **7.87 h** | base 0.82; [recipes/tulu3_sft](../interactive_training/recipes/tulu3_sft/README.md) |
| **Interactive Training**  | gpt-oss-20b | **0.55** | **6.03 h** | base 3.26 — high initial NLL reflects the `gpt_oss_no_sysprompt` renderer vs. Tulu3's chat format; converges to parity by step 300 of 1,740. LoRA r=16. |

## Caveats

1. **Cross-framework numbers aren't perfectly comparable.** Base accuracies
   for the same model should in principle be identical across rows, but each
   framework evaluates at temperature 1 with its own sampler, and
   chat-template handling differs in small ways we tried — but did not
   perfectly manage — to align. Read these as the recorded outcomes under those
   configurations, not controlled head-to-head deltas or a general superiority
   claim.
2. **Keep task and execution modes distinct.** The RL measurements are recorded
   with colocated trainer/sampler and synchronous steps; Tulu3 is SFT and
   reports held-out NLL, not RL accuracy. Some code runs are partial. Do not
   generalize the historical topology or timing to every supported model/tier.
3. **Evidence is bounded by the recorded artifacts.** The tables do not supply
   a fresh independent reproduction, repeated-run uncertainty, a bill, or
   proof that an application-serving endpoint was deployed. Use the
   [evaluation contract](./evaluation-and-inference.md#set-the-comparison-contract-first)
   for a new task comparison.

## Reproducing these numbers

Each Interactive Training row references a recipe with its original
hyperparameters and context. A recipe command is not a guarantee that current
capacity, dependencies, model eligibility, and service behavior can reproduce
the historical run. Check current support and explicit spend approval first.
For a new comparison, record the code/SDK version, dataset and evaluation
contract, hardware/tier when known, timing boundary, failures, and actual
artifacts; do not substitute a historical table for new measurements.

TRL and verl rows are best-effort reproductions from upstream example
scripts, matched on base model, LoRA rank, optimizer, and dataset. We
couldn't always line up Interactive Training's algorithm implementation one-for-one with the
other frameworks, so treat those rows as the closest apples-to-apples we
could get rather than identical configurations.
