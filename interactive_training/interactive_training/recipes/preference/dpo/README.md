# Direct Preference Optimization

Train a model to favor a preferred reply over a rejected reply to the same
conversation, using offline preference pairs rather than online rollouts or a
separately trained reward model. This recipe implements Direct Preference
Optimization using the Azure AI Fine-Tuning Sessions SDK.

## Task and data

The measured runs use **Qwen3-32B with rank-16 LoRA** on
[Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf), which
contains human-preferred (`chosen`) and less-preferred (`rejected`) conversation
responses collected for helpfulness and harmlessness. Training and evaluation
use its separate `train` and `test` splits, shuffled with seed 0 before selection.
These runs use small subsets, not the full HH-RLHF corpus.

For a synthetic example, both conversations might end with the user asking
"What is 2 + 2?", with a chosen reply "4" and a rejected reply "5". Each
row becomes an adjacent chosen/rejected pair. DPO increases the chosen reply's
logprob improvement over the frozen reference relative to the rejected reply.

| CLI dataset | Source | Held-out selection |
| --- | --- | --- |
| `hhh` | [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) | First `max_test_examples` shuffled test rows |
| `helpsteer3` | [nvidia/HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3), preference config | First `max_test_examples` shuffled validation rows |
| `ultrafeedback` | [argilla/ultrafeedback-binarized-preferences](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences) | First `max_test_examples` shuffled train rows, excluded from training |

Only HH-RLHF has been exercised in live runs. The other dataset adapters have
offline conversion tests. Datasets download on first use; revisions are not
pinned, so reproducing exact historical row selections requires the same source
revision. Consult the linked dataset cards for provenance and usage terms.

`batch_size` counts preference pairs: eight pairs produce sixteen datums.
The loss uses masked sequence logprob sums, not length averages.
Malformed rows and ties are skipped as whole pairs; incomplete final batches
are dropped. Only text is supported. Rendering follows the renderer's default
assistant-token mask, including earlier assistant turns in shared context; their
identical contributions cancel between pair members when both are retained.
`max_length` truncates each conversation; pairs with no remaining loss tokens
on either side are skipped.

## Smoke example

The current default is `Qwen/Qwen3.8-27B`. The small command below checks the
pipeline on that default; the measured experiments later in this guide retain
their original Qwen3-32B model and are not Qwen3.8 benchmarks.

Follow the cookbook [installation guide](../../../../README.md#install) and
[authentication guide](../../../../docs/auth.md). Run from the cookbook root
with `PROJECT_ENDPOINT` and `AZURE_AI_API_KEY` exported, or use Azure CLI/managed
identity authentication instead of an API key. The smoke command uses the current
model default and the measured run's LoRA rank; availability depends on the
service and region.

```bash
python -m interactive_training.recipes.preference.dpo.train_azure \
  project_endpoint="$PROJECT_ENDPOINT" \
  model_name=Qwen/Qwen3.8-27B lora_rank=16 dataset=hhh \
  batch_size=8 max_train_examples=16 max_test_examples=8 max_length=2048 \
  max_steps=2 learning_rate=1e-5 dpo_beta=0.1 eval_every=1 save_every=1 \
  log_path=./logs/dpo-hhh-smoke behavior_if_log_dir_exists=raise
```

This checks reference scoring, custom-loss gradients, optimizer steps,
evaluation, periodic saves, and final checkpoint export. It is not a quality
benchmark. Use a fresh log directory for each new run.
The historical Qwen3-32B variant passed on 2026-09-22 with bounded reference
scoring and resume guards enabled, including final checkpoint export. The
updated Qwen3.8 command has not been run against the service here.

## Full measured example

The following is the complete **40-update learning-check configuration** that
generated the results below. It deliberately overfits eight fixed training pairs
with an elevated, constant learning rate, while evaluating 32 separate test pairs.
"Full" here means the complete measured experiment, not full-corpus training.

```bash
python -m interactive_training.recipes.preference.dpo.train_azure \
  project_endpoint="$PROJECT_ENDPOINT" \
  model_name=Qwen/Qwen3-32B lora_rank=16 dataset=hhh dpo_beta=0.1 \
  batch_size=8 max_train_examples=8 max_test_examples=32 max_length=2048 \
  num_epochs=40 max_steps=40 learning_rate=0.0001 lr_schedule=constant \
  eval_every=5 save_every=20 log_path=./logs/dpo-hhh-learning-check \
  behavior_if_log_dir_exists=raise
```

## Results

Evaluation compares the same first training batch (`train_probe/*`) and the same
held-out pairs (`test/*`) throughout the run, rather than comparing changing
training batches. Probe reference logprobs are computed once and reused. Initial
and final rows are marked with `evaluation_phase` in `metrics.jsonl`.

The desired plumbing signal is increasing `train_probe/accuracy` and
`train_probe/margin` with falling `train_probe/dpo_loss`. Accuracy compares
chosen/rejected policy-to-reference logprob ratios, not raw response likelihood.
An exact tie counts as incorrect. Held-out improvement is not required for this
intentional overfit test, and training-probe success does not establish
generalization.

Observed on Qwen3-32B with the command above (2026-09-21):

| Completed updates | Fixed train preference accuracy | Fixed train DPO loss | Fixed train reward margin | Held-out preference accuracy |
| --- | --- | --- | --- | --- |
| 0 | 25% | 0.703756 | -0.020493 | 62.5% |
| 5 | 100% | 0.009084 | 5.290776 | 50.0% |
| 10 | 100% | 0.000152 | 9.980913 | 43.75% |
| 15 | 100% | 0.000043 | 11.909700 | 43.75% |
| 40 | 100% | 0.00000858 | 14.304611 | 40.625% |

These measurements demonstrate successful preference optimization on the fixed
training probe. They do not demonstrate improved held-out quality; the tiny
training set and elevated learning rate intentionally favor memorization.
The 40-update learning check exited successfully and saved the final checkpoint.

Held-out DPO loss rose from **0.685220 to 0.843573**, consistent with the
accuracy regression. `test/nll` includes both chosen and rejected replies and
is not itself a preference-quality metric. These are single-run observations,
not a benchmark of generated-response helpfulness or harmlessness.

## Training and validation

A fresh run freezes the initial policy weights as a reference
sampler, then trains the policy using `forward_backward_custom_async`. The
custom loss runs locally over per-token logprobs; Interactive Training's existing adapter applies
the corresponding surrogate gradients on the service. No service or SDK changes
are required. `reference_concurrency=32` bounds reference-scoring requests,
including the held-out probe.

An exactly equal policy/reference gives `dpo_loss=log(2)` before training;
sampler/trainer numerical differences may move the initial value slightly.
`eval_every=0` disables evaluation; `save_every=0` disables periodic saves but
retains the final checkpoint. `max_steps=0` or `None` trains all epochs.
`max_steps` caps completed optimizer updates, not raw dataset batches. Empty
batches do not consume that budget or advance the learning-rate schedule;
training still stops when `num_epochs` is exhausted. The schedule horizon is
the smaller of the update cap (if set) and the raw batch count across epochs,
so skipped batches can leave a decaying schedule unfinished.

A separate 400-row, 50-update pipeline run and a one-update checkpoint resume
also completed, including final state and sampler checkpoint export:

```bash
python -m interactive_training.recipes.preference.dpo.train_azure \
  project_endpoint="$PROJECT_ENDPOINT" \
  model_name=Qwen/Qwen3-32B lora_rank=16 dataset=hhh \
  batch_size=8 max_train_examples=400 max_test_examples=64 \
  max_length=2048 max_steps=50 learning_rate=1e-5 lr_schedule=linear \
  dpo_beta=0.1 eval_every=10 save_every=10 \
  log_path=./logs/dpo-hhh-test behavior_if_log_dir_exists=raise
```

That run consumed 396 valid pairs and exported a final checkpoint.
To reproduce the one-update continuation,
repeat the command with `num_epochs=2 max_steps=51` and
`behavior_if_log_dir_exists=resume`. Held-out NLL was essentially flat during
the original run (2.359421 to 2.359037); no quality improvement is claimed.

## Checkpoints

Set `load_checkpoint_path` to the `state_path` recorded in your SFT run's
`checkpoints.jsonl` to start DPO from that checkpoint. It initializes both
policy and reference. The original
reference is saved durably before the first update and its `reference_state_path`
is retained in every entry of `checkpoints.jsonl`.

To resume, reuse the same `log_path`, dataset settings, model, and LoRA rank with
`behavior_if_log_dir_exists=resume`. The latest policy checkpoint and dataset
cursor win over `load_checkpoint_path`. Resume allocates a separate session
for the original reference, rather than resetting it to the updated policy.
Both sessions are closed on exit. Fresh runs need only one session.
Checkpoints store `completed_updates` separately from the next dataset
`epoch`/`batch` cursor, including trailing skipped batches. Legacy checkpoints
without this count retain the old assumption that every consumed batch produced
an update and emit a warning; their historical count cannot distinguish skips.
Resume validates model, LoRA rank, resolved tokenizer/renderer, dataset, pair
batch size, length/example limits, and DPO beta against `run_meta.json` before
allocating a session. Increasing `num_epochs` or `max_steps` is allowed. Keep
the same dataset revision as well; source content changes are not detected.

Run offline coverage without Azure credentials:

```bash
python -m pytest -q tests/test_dpo_recipe.py
```