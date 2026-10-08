# Continual Fine-Tuning

Continue training from a previously saved checkpoint. This creates a *new* Azure session initialized from the referenced checkpoint: the LoRA adapter weights **and the optimizer's Adam moment buffers** carry over (a warm-start optimizer). What does *not* carry over is the **dataset position** — a continual-FT run points at a fresh `log_path`, so it starts at batch 0 over whatever dataset you give it. The learning rate is whatever you pass on the CLI (it's applied client-side on each step), not restored from the source run.

> [!NOTE]
> If instead you want to **recover an interrupted run** (continue the *same* dataset from where it stopped), see [Resuming after an interruption](./training.md#resuming-after-an-interruption). Both paths warm-start the optimizer from the checkpoint via the same `from_checkpoint` request; the only difference is the **dataset cursor** — crash recovery restores it from the existing `log_path`'s ledger, while continual fine-tuning starts at batch 0 on a fresh `log_path`.

## CLI

Pass `load_checkpoint_path` to the recipe:

```bash
python -m interactive_training.recipes.math_rl.train_azure \
    project_endpoint="<your-project-endpoint>" \
    load_checkpoint_path="session_<session_id>/<checkpoint_name>" \
    model_name="Qwen/Qwen3.8-27B" \
    env=gsm8k \
    learning_rate=2e-5 temperature=1.0 max_tokens=1200 lora_rank=32 \
    group_size=8 groups_per_batch=25 loss_fn=importance_sampling seed=42 \
    max_steps=2 max_train_examples=50 max_test_examples=8 \
    eval_every=1 save_every=1 \
    log_path=./runs/continued-math-check behavior_if_log_dir_exists=raise
```

  This is a bounded **remote** check, not an offline restore or a guarantee of further improvement. Match the source checkpoint's model, renderer/tokenizer, and LoRA configuration before launching. Use a fresh run directory: shared RL auto-resume can override an explicit checkpoint when an existing ledger is present; Tulu3 has a different selection order and can mix an explicit checkpoint with an old cursor if you reuse an unrelated directory.

## Format

The model and LoRA configuration must match the checkpoint's original run.
An older Qwen3-32B checkpoint still requires `model_name="Qwen/Qwen3-32B"`;
do not use the new Qwen3.8 default to resume a different base model.

Compatibility is not eligibility: the [current model list](./supported_models.md#available-today)
excludes Qwen3-32B from new interactive training. Do not launch a restore solely
because a historical checkpoint exists; resolve current project/model support
before allocating a new session.

`<session_id>/<checkpoint_name>`:

- `<session_id>` is the `session_<...>` identifier of the previous run's session. It is printed as `Session ready: session_id=session_<...>` at the start of that run and recorded in its [`run_meta.json`](./storage.md#run_metajson) under `azure.session_id`. The `session_` prefix is added automatically if you omit it; legacy `model_` values are accepted and normalized. To enumerate sessions the service knows about, see [session-management.md](./session-management.md#list-your-sessions).
- `<checkpoint_name>` is the `name` field from a row in that run's [`checkpoints.jsonl`](./storage.md#checkpointsjsonl) — typically `final` or a numbered batch like `4`. You can also list checkpoints for a session via the SDK — see [session-management.md](./session-management.md#list-checkpoints-for-a-session).

## What carries over

| Carries over | Does not carry over |
|---|---|
| LoRA adapter weights | Dataset position (fresh `log_path` → starts at batch 0) |
| Optimizer state (Adam moment buffers) | |

The new run's **effective learning rate** comes from its client config on every optimizer step (a constant for RL, a client-computed schedule for SFT). Restoring optimizer buffers does not override the newly supplied `learning_rate` / `lr_schedule`; do not assume a resumed schedule matches the original without comparing config and step position.

## Where it shows up

When the recipe successfully resumes, it stamps the new run's `run_meta.json`:

```json
{
  "azure": {
    "session_id": "session_<new-session>",
    "from_checkpoint": {
      "source_session_id": "session_<previous-session>",
      "checkpoint_id": "final"
    }
  }
}
```

The dashboard surfaces this as a **Resumed from** row in the Run Overview card.

## Sanity check

A useful check: launch from checkpoint `X` with a fresh run directory and the same held-out dataset, prompts, renderer, sampling budget, grader, and seeds/settings. In a fresh **synchronous** math run with evaluation enabled, compare the initial pre-update evaluation to `X`'s corresponding saved-weight evaluation. Similar results support the restore check, but stochastic outputs and dropped groups can differ; unchanged temperature alone does not guarantee matching scores. Confirm the checkpoint metadata and `azure.from_checkpoint` as well. See [evaluation and inference](./evaluation-and-inference.md) for a controlled comparison and sampling without optimizer updates.
