# Using Reinforcement Learning to Solve Math Problems

Math problems have been an active testbed for RL with LLMs. This recipe wires GSM8K into the generic train loop with a boxed-answer exact-match grading function.

Each prompt is appended with "Write your answer in `\boxed{}` format." The grading function (`math_grading.py`) extracts the contents of the last `\boxed{}` in the model's response and compares it to the gold answer. This format is convenient because it cleanly separates the reasoning chain from the final answer, and most modern LLMs are already trained to produce `\boxed{}` output for math problems.

## Setup

Authenticate to the Azure AI Fine-Tuning Sessions endpoint with either an API
key (`export AZURE_AI_API_KEY=...`) or `az login` (`DefaultAzureCredential`) —
see [docs/auth.md](../../../docs/auth.md).

Run from the **cookbook directory**, not this recipe directory. Use the
[bounded quickstart](../../../docs/quickstart.md) for a first run; all training
commands here allocate remote compute and may incur charges. The CLI default
`env=arithmetic max_tokens=5` is a toy configuration with **no held-out set**,
not an appropriate default completion budget for GSM8K.

## RL on GSM8K

### Current default: Qwen3.8

The text-recipe default is `Qwen/Qwen3.8-27B`; tokenization follows the model.
The default renderer is `qwen3_8_low_reasoning`, with thinking enabled at low
effort. Start with the bounded [quickstart](../../../docs/quickstart.md); its
token budget is unchanged. Use `qwen3_8_disable_thinking` only when you explicitly
want non-thinking output. The measured Qwen3-32B configuration below is retained
for reproduction, not as the default.

### Historical Qwen3-32B benchmark configuration

```bash
python -m interactive_training.recipes.math_rl.train_azure \
    project_endpoint="<your-project-endpoint>" \
    model_name="Qwen/Qwen3-32B" tokenizer_name="Qwen/Qwen3-32B" \
    env=gsm8k \
    learning_rate=2e-5 temperature=1.0 max_tokens=1200 lora_rank=32 \
    group_size=8 groups_per_batch=256 loss_fn=importance_sampling seed=42 \
    eval_every=999999 save_every=999999
```

`eval_every` and `save_every` are set higher than the total number of steps, so evaluation only runs at the very beginning and end of training. End-to-end training time is ~10 hours, though this number may change significantly as we continue tuning our infrastructure. We observe GSM8K accuracy go from 69% to 95.53%.

Here is an illustrative plain-text answer, not a Qwen3.8 benchmark or token-format example:

```
A plane takes off at 6:00 a.m. and flies for 4 hours from New York City to Chicago. The plane stays at the port in Chicago for 1 hour and then departs for Miami. If the aircraft took three times as many hours to fly to Miami than it took to fly from New York to Chicago, calculate the total time to travel from New York to Miami. Write your answer in \boxed{} format.
Let's break it down step by step:
1. The plane flies from New York City to Chicago for 4 hours. This duration is given.
2. The plane stays at the port in Chicago for 1 hour.
3. The time it takes to fly from Chicago to Miami is three times the time it took to fly from New York to Chicago, which is 4 hours. So, the time to fly from Chicago to Miami is 3 * 4 = 12 hours.
Now, let's calculate the total time:
* Flight from New York City to Chicago: 4 hours
* Stay at the port in Chicago: 1 hour
* Flight from Chicago to Miami: 12 hours
Total time = 4 + 1 + 12 = 17 hours
So, the total time to travel from New York to Miami is 17 hours.
\boxed{17}
```

## Custom losses (illustrative)

Interactive Training supports shipping a **research-grade loss as a Python closure** without
waiting for a backend release: see
[`forward_backward_custom_async`](../../../docs/loss_functions.md) and the
side-by-side correctness proofs in
[`tests/test_surrogate_gradient_equivalence.py`](../../../tests/test_surrogate_gradient_equivalence.py).

This recipe wires up that extension point **purely for illustration** so you
can see what a real integration looks like. The `custom_loss` CLI flag swaps
every `forward_backward` call out for a Python-side loss closure:

```bash
# Re-implementation of importance_sampling, hand-rolled in Python.
# Reward curves should match the builtin `loss_fn=importance_sampling` run
# with the same model/settings (modulo rollout noise).
python -m interactive_training.recipes.math_rl.train_azure \
    project_endpoint="<your-project-endpoint>" \
    model_name="Qwen/Qwen3.8-27B" renderer_name=qwen3_8_low_reasoning \
    env=gsm8k custom_loss=is_py \
    learning_rate=2e-5 temperature=1.0 max_tokens=1200 lora_rank=32 \
    group_size=8 groups_per_batch=256 seed=42

# Reward-agnostic smoke test: confidence-squared regulariser
# (`C = Σ_t logprob_t²`).  This is **not** RL — it ignores advantages
# and reinforces every sampled token equally, much like plain SFT on
# your own rollouts.  Its primary use here is a fast end-to-end check
# that the custom-loss wire path is healthy against your real session;
# whether it incidentally improves accuracy depends on whether your
# base model's rollouts are net-positive on GSM8K.
python -m interactive_training.recipes.math_rl.train_azure \
    project_endpoint="<your-project-endpoint>" \
    model_name="Qwen/Qwen3.8-27B" renderer_name=qwen3_8_low_reasoning \
    env=gsm8k custom_loss=squared_logprob \
    max_steps=3 max_train_examples=24 max_test_examples=8 \
    groups_per_batch=8 group_size=4 max_tokens=256

# DAPO decoupled-clip surrogate (arXiv:2503.14476).  PPO-style clipped IS
# with an *asymmetric* trust region: eps_low=0.20, eps_high=0.28.  The
# wider upper bound ("clip-higher") gives positive-advantage tokens more
# headroom before the clip binds, which the DAPO paper reports reduces
# entropy collapse on reasoning tasks.  This is the canonical example of
# a loss that needs *two* per-token aux quantities (advantages + sampling
# logprobs) -- see docs/loss_functions.md.
python -m interactive_training.recipes.math_rl.train_azure \
    project_endpoint="<your-project-endpoint>" \
    model_name="Qwen/Qwen3.8-27B" renderer_name=qwen3_8_low_reasoning \
    env=gsm8k custom_loss=dapo \
    learning_rate=2e-5 temperature=1.0 max_tokens=1200 lora_rank=32 \
    group_size=8 groups_per_batch=256 seed=42
```

When `custom_loss` is set, `loss_fn` is ignored. Each `forward_backward` call
costs ~1.5× the FLOPs of a builtin loss (one extra forward pass for the
surrogate-gradient trick) — use the builtin whenever it covers your needs;
reach for `custom_loss` when you need to prototype a new objective.

To add your own selector, edit `_CUSTOM_LOSS_REGISTRY` in
[`train_azure.py`](train_azure.py): register a `(repack_fn, loss_fn)` pair,
where `repack_fn(Datum) -> Datum` prepares the per-token auxiliary data and
`loss_fn(data, logprobs_list) -> (loss_tensor, metrics_dict)` is your Python
loss closure. The repack can either (a) precompute everything into a single
`weights` field (see `_repack_for_is` — minimal payload, mathematically
equivalent), or (b) carry **multiple per-token aux quantities under their
own named keys** (see `_repack_for_dapo`, which packs `dapo/mask_adv` and
`dapo/sampling_lp` as separate fields). Option (b) is required whenever
your loss needs more than one per-token raw quantity at the same time —
DAPO is the canonical example, because the clip operates on the raw ratio
`exp(lp - lp_sampling)` and so needs both `advantages` and `lp_sampling`
visible un-mashed inside the closure. The cookbook adapter strips these
extras from the wire payload — they only ever exist in your Python
closure. See [docs/loss_functions.md](../../../docs/loss_functions.md)
constraint 1 for the full contract.

## Adapt, evaluate, and recover

- Replace task data/grading through the existing [RL interfaces](../../../docs/custom-data.md#reinforcement-learning-data-and-rewards), not a fictional custom-file CLI argument.
- Inspect `test/env/all/correct`, `test/env/all/reward/total`, `format`, completion-budget diagnostics, and the scored-episode count; [compare before/after](../../../docs/evaluation-and-inference.md) on a fixed held-out set.
- Preserve the actual completed training checkpoint and use the [post-training sampling helper](../../../docs/evaluation-and-inference.md#use-a-training-checkpoint-after-the-original-session-closes); a sampler ID alone is not a portable deployment.
- For interruption, keep the original run config/ledger and follow [recovery](../../../docs/training.md#resuming-after-an-interruption). For a fresh continual run, use a new directory.
- Normal cleanup is attempted by the shared loop. Inspect an owned leftover session with [session management](../../../docs/session-management.md) if cleanup fails; do not bulk-unload a project.
