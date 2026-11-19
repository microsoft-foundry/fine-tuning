# LLM-as-judge RL: BillSum summarization example

This recipe demonstrates how to use a separate LLM as a grader for
reinforcement learning tasks whose quality depends on semantic criteria that
are difficult to express with exact-match rules. The same pattern can support
summarization, open-ended question answering, dialogue quality, instruction
following, and style or policy compliance.

BillSum summarization is the concrete example used here. A Loom policy reads
proposed legislation and writes a concise summary; a separate Azure OpenAI
deployment evaluates the summary against an explicit rubric and supplies the
reinforcement-learning reward.

The recipe demonstrates a single-stage hill-climbing workflow:

1. establish a fixed validation baseline;
2. train with LoRA reinforcement learning;
3. evaluate the same validation rows every 10 steps; and
4. select the checkpoint with the highest validation reward.

An LLM judge is not a substitute for human evaluation in a high-impact
deployment. It works best with an explicit rubric, auditable outputs, and
calibration against human evaluation.

## Task

The policy receives:

- the bill title;
- the complete bill text; and
- an instruction to produce a summary of at most 175 words.

The policy does **not** receive the BillSum reference summary. The source bill
is never silently truncated.

For grading, the judge receives the source bill, the policy summary, the
BillSum reference summary, and the rubric. The source bill is authoritative:
the reference helps assess coverage and omissions, but it is not an
exact-match target and does not override the source.

## Dataset

The default dataset is
[`FiscalNote/billsum`](https://huggingface.co/datasets/FiscalNote/billsum),
pinned to revision:

```text
3d8510441c06a3d9dfb32eb0d7f80151730bcc4f
```

Only the official `train` split is loaded. The official `test` and `ca_test`
splits are left untouched for independent final evaluation.

The extended configuration uses:

| Partition | Rows | Purpose |
| --- | ---: | --- |
| Training | 800 | RL updates |
| Validation | 128 | Fixed evaluation every 10 steps |

The split is deterministic. Each training-split row is ranked by:

```text
SHA256("<seed>:<example_id>")
```

The first `validation_size` rows become validation; training is taken from the
remaining rows. The published configuration uses `seed=20261014`, so the same
dataset revision and parameters reproduce the same split.

Rows with missing fields or prompts that exceed the configured policy or judge
token budgets are rejected and counted. Bill text is not shortened to make a
row fit.

### Synthetic smoke fixture

[`data/synthetic_billsum.jsonl`](data/synthetic_billsum.jsonl) contains small,
original bill-like examples written for this repository. It is intended for
tests and low-cost integration smoke runs; it does not contain copied BillSum
rows and is not used by the extended experiment.

## Evaluation and reward

The Azure OpenAI judge returns integer scores from 0 through 4:

| Dimension | Weight |
| --- | ---: |
| Factual consistency | 0.40 |
| Coverage | 0.30 |
| Relevance | 0.15 |
| Clarity | 0.10 |
| Concision | 0.05 |

The recipe validates the structured response, normalizes each score by four,
and computes the weighted aggregate locally. When normalized factual
consistency is below `0.5`, the aggregate is capped at `0.5`.

A candidate is reward-eligible only when:

- the renderer parses the completion successfully; and
- the summary satisfies the mechanical word limit.

Ineligible candidates receive final reward zero.

The primary metric is:

```text
policy/eligibility_adjusted_reward
```

For fixed validation, use:

```text
test/env/all/policy/eligibility_adjusted_reward
```

Also inspect `policy/reward_eligible` and `judge/aggregate_reward` to distinguish
format/length improvements from improvements in judged summary quality.
Validation is complete-or-fail: a point is not published if any of the 128 rows
is missing.

## Configure the judge

The Loom project and judge are separate endpoints:

- `project_endpoint` identifies the Azure AI Foundry project used for training.
- `AZURE_OPENAI_JUDGE_ENDPOINT` identifies the Azure OpenAI judge.
- `AZURE_OPENAI_JUDGE_DEPLOYMENT` identifies the judge deployment or model.

Supported judge endpoint forms are:

| Endpoint form | Protocol |
| --- | --- |
| `https://<resource>.openai.azure.com` | Azure OpenAI Chat Completions |
| `https://<host>/openai/v1` | OpenAI-compatible Responses API |

Keep `/openai/v1` when it is part of the endpoint. Do not provide a complete
operation URL ending in `/responses`, `/chat/completions`, or a deployment
path.

The judge must support strict JSON-schema structured output. Authenticate with
Microsoft Entra ID when possible:

```bash
az login
export AZURE_OPENAI_JUDGE_ENDPOINT="https://<judge-host>/openai/v1"
export AZURE_OPENAI_JUDGE_DEPLOYMENT="<model-or-deployment>"
```

PowerShell:

```powershell
az login
$env:AZURE_OPENAI_JUDGE_ENDPOINT = "https://<judge-host>/openai/v1"
$env:AZURE_OPENAI_JUDGE_DEPLOYMENT = "<model-or-deployment>"
```

If the resource requires key authentication, place the secret only in the
environment:

```bash
export AZURE_OPENAI_JUDGE_API_KEY="<judge-key>"
```

```powershell
$env:AZURE_OPENAI_JUDGE_API_KEY = "<judge-key>"
```

Do not put keys in commands, configuration files, or source control. Loom
training continues to use its independent `AZURE_AI_API_KEY` or
`DefaultAzureCredential` authentication.

## Run

Run commands from `programmed_training/`.

### Low-cost smoke run

This verifies dataset loading, policy sampling, judge connectivity, reward
calculation, evaluation, and checkpoint creation:

```bash
python -m programmed_training.recipes.llm_judge_rl.train_azure \
  project_endpoint="<your-project-endpoint>" \
  renderer_name=qwen3_disable_thinking \
  dataset_path="programmed_training/recipes/llm_judge_rl/data/synthetic_billsum.jsonl" \
  validation_size=2 max_train_examples=4 max_test_examples=2 \
  group_size=2 groups_per_batch=2 max_steps=1 \
  eval_every=1 save_every=1 behavior_if_log_dir_exists=raise
```

### Extended hill-climbing run

```bash
python -m programmed_training.recipes.llm_judge_rl.train_azure \
  project_endpoint="<your-project-endpoint>" \
  model_name="Qwen/Qwen3-32B" tokenizer_name="Qwen/Qwen3-32B" \
  renderer_name=qwen3_disable_thinking \
  judge_max_concurrency=8 judge_max_response_tokens=2400 \
  learning_rate=5e-6 temperature=0.7 max_tokens=320 lora_rank=16 \
  group_size=8 groups_per_batch=8 max_concurrent_groups=4 \
  seed=20261014 word_limit=175 \
  validation_size=128 max_train_examples=800 max_test_examples=128 \
  max_retries_per_trajectory=4 max_extra_trajectory_attempts_per_group=4 \
  rollout_failure_max_resets=20 rollout_failure_reset_backoff_sec=120 \
  require_full_validation=true \
  max_steps=100 eval_every=10 save_every=4
```

The examples use Bash line continuations. In PowerShell, replace each trailing
`\` with a backtick (`` ` ``), or put the command on one line.

`qwen3_disable_thinking` is intentional for this task. A thinking renderer can
consume the output budget before producing the requested final summary.

## Results

The extended configuration completed 100 optimizer steps and evaluated the
same 128 validation rows every 10 steps.

| Step | Adjusted reward | Eligibility | Raw judge reward |
| ---: | ---: | ---: | ---: |
| 0 | 0.736621 | 88.28% | 0.829395 |
| 30 | 0.779395 | 92.19% | 0.845215 |
| 40 | 0.816797 | 98.44% | 0.827344 |
| 60 | 0.831934 | 99.22% | 0.838379 |
| **70** | **0.848242** | **100.00%** | **0.848242** |
| 80 | 0.835059 | 100.00% | 0.835059 |
| 100 | 0.827344 | 99.22% | 0.833789 |

Checkpoint 70 was selected. Relative to the base model, it improved:

- adjusted reward by **0.111621** (+11.16 points);
- eligibility by **11.72 points**; and
- raw judge reward by **0.018847** (+1.88 points).

The curve demonstrates hill climbing through step 70. Steps 80–100 remained
below the peak, indicating saturation and mild late-run regression rather than
monotonic improvement at every checkpoint.

These results describe one run and one judge rubric, not cross-seed
convergence. Absolute scores from runs using a different rubric should not be
compared directly.

## Outputs

The run directory contains:

| Artifact | Contents |
| --- | --- |
| `config.json` | Resolved run configuration |
| `run_meta.json` | Reproducibility metadata |
| `metrics.jsonl` | Training and validation metrics |
| `checkpoints.jsonl` | Saved model checkpoints |
| `train_iteration_*.html` | Sampled training trajectories |
| `eval_test_iteration_*.html` | Fixed-validation examples |
| `judge_failures.jsonl` | Customer-safe judge failure categories |
| `logs.log` | Runtime and HTTP diagnostics |

The dashboard does not publish a partial validation point. If an evaluation is
still running, its rollouts may appear in `logs.log` before the metric appears.

## Common failures

| Symptom | Check |
| --- | --- |
| HTTP 400 | Confirm the deployment supports strict JSON-schema output and the endpoint protocol is correct. |
| HTTP 401/403 | Check the judge credential and its data-plane role. |
| HTTP 404 | Check the endpoint form and deployment name. |
| HTTP 408/5xx | Treat as transient; retain the published retry settings. |
| HTTP 429 | Reduce `judge_max_concurrency` or increase judge quota. |
| Empty summary | Confirm `renderer_name=qwen3_disable_thinking` and sufficient output tokens. |

Transport failures, refusals, malformed responses, and invalid schemas are not
converted into reward zero. Training retries or drops the affected group;
validation fails rather than silently reporting a biased metric.

## Reproducibility limits

The dataset revision, split, prompts, rubric version, and token budgets are
deterministic for a fixed configuration. Policy sampling remains stochastic,
and changes to the Azure OpenAI deployment can change judge scores. Record the
dataset revision, seed, policy model, renderer, judge deployment, endpoint
protocol, rubric version, and selected checkpoint when comparing runs.
