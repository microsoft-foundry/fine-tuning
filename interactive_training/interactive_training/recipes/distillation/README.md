# On-policy distillation (POC)

A minimal, end-to-end example of **on-policy distillation** on the Azure AI
Fine-Tuning Sessions backend, meant to get you started — not a benchmark.

The student samples completions for prompts, a separate **teacher** model scores
them, and the only training signal is the KL penalty pulling the student toward
the teacher (no correctness/format rewards). There is deliberately no environment
logic (no grading, sandbox, or tools) — the prompt-only env returns reward `0.0`
and the KL penalty does all the work. Prompts come from the Tulu3 mixture
(`allenai/tulu-3-sft-mixture`), streamed.

Authenticate to the endpoint with either an API key (`export AZURE_AI_API_KEY=...`)
or `az login` (`DefaultAzureCredential`) — see [docs/auth.md](../../../docs/auth.md).

## Distillation against a teacher

Point `teacher_checkpoint` at any existing checkpoint (e.g. an SFT'd model) to
distill it into a fresh student:

```bash
python -m interactive_training.recipes.distillation.on_policy_distillation_azure \
    project_endpoint="<your-project-endpoint>" \
    model_name="Qwen/Qwen3.8-27B" \
    teacher_model="Qwen/Qwen3.8-27B" \
    teacher_checkpoint="<session>/final" \
    kl_penalty_coef=1.0 \
    lora_rank=16 group_size=4 groups_per_batch=8 \
    max_train_examples=64 max_tokens=256 max_steps=10 \
    behavior_if_log_dir_exists=raise
```

Two Qwen3.8 sessions run concurrently — the student, plus a separate teacher
session created from `teacher_checkpoint` (so distillation needs quota for two
models). Omit `teacher_checkpoint` and the teacher defaults to the student's base
model; that exercises the full path but there's then nothing to distill (see
below).
The run's ignored `run_meta.json` records the student as `azure.session_id`
and the separate teacher as `azure.reference_session_id` after each is created.
Check both IDs with the SDK after training; a local exit alone does not prove
that either remote session reached `succeeded`.

## Making a teacher first (SFT → distill)

To obtain a teacher that differs from the base student, **SFT one first**, then
distill its checkpoint into a fresh student using the same base model. The
teacher scores the student's token IDs, so teacher and student must use
compatible tokenization; arbitrary cross-model distillation is not implied.
The shared default is Qwen3.8, not a restriction to a single available model.
This mirrors tinker-cookbook's SFT → on-policy distillation flow.

**Stage 1 — SFT a teacher** (reuses [`recipes/tulu3_sft`](../tulu3_sft/README.md)):

```bash
python -m interactive_training.recipes.tulu3_sft.train_azure \
    project_endpoint="<your-project-endpoint>" \
    model_name="Qwen/Qwen3.8-27B" \
    lora_rank=16 batch_size=16 max_length=2048 \
    learning_rate=5e-4 max_train_examples=512 max_steps=15 \
    eval_every=999999 save_every=999999 \
    log_path="$HOME/interactive-post-training-runs/distillation_demo/sft-teacher" \
    behavior_if_log_dir_exists=raise
```

The run saves a `final` checkpoint; grab its `state_path` from the ledger:

```bash
tail -1 "$HOME/interactive-post-training-runs/distillation_demo/sft-teacher/checkpoints.jsonl"
# -> {"name": "final", ..., "state_path": "model_<id>/final", ...}
```

**Stage 2 — distill it into a fresh student** using that `state_path` as
`teacher_checkpoint` in the "Distillation against a teacher" command above, with
`model_name` left as the base Qwen3.8 model used to create the teacher checkpoint.
Do not load a Qwen3-32B checkpoint into a Qwen3.8 session.

## What to watch

The training signal is **`kl_policy_base`** — the mean per-token
`student_logprob - teacher_logprob` on the student's own rollouts (the
student-vs-teacher KL that distillation minimizes).

- If **teacher == student**, it sits at ~0 — nothing to learn.
- With a **teacher that differs** (e.g. an SFT'd checkpoint), a non-zero value
  indicates a teacher/student difference. Evaluate the trend on your own run;
  no measured Qwen3.8 KL or quality result is claimed here. The historical
  Qwen3-32B short run observed ~0.15–0.2; that is not a Qwen3.8 measurement.
- Over a long enough run you'd expect `kl_policy_base` to **trend downward** as
  the student imitates the teacher. This short POC only shows that the pipeline
  runs and computes a real teacher KL — it does not run long enough to show that
  decrease.
