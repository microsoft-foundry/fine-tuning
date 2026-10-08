# Storage

Find a run's local artifacts, interpret its metrics/checkpoint ledger, and distinguish those files from remote model state. Artifact sets vary by recipe and by how far a run progressed.

## Logs root

When `log_path` is not supplied, launchers normally choose `<logs-root>/<recipe>/<run_name>/`. An explicit `log_path`, such as the quickstart's `./runs/first-math-check`, is the run directory itself. Relative paths are resolved from the **cookbook working directory**. The default `<logs-root>` is per-platform:

| Platform | Default logs root |
|---|---|
| Linux / macOS / WSL | `~/interactive-post-training-runs` |
| Windows (native) | `C:\interactive-post-training-runs` |

Override per-run with `log_path=/full/path/to/run-dir`, or globally with the `INTERACTIVE_POST_TRAINING_LOGS_ROOT` environment variable (e.g. `export INTERACTIVE_POST_TRAINING_LOGS_ROOT=/data/interactive-post-training-runs`).

**Rename compatibility:** the outer cookbook and inner Python package now both use `interactive_training`, and the cookbook distribution is `interactive-training`. The legacy `INTERACTIVE_POST_TRAINING_LOGS_ROOT` environment variable and default `interactive-post-training-runs` directories remain unchanged so existing run history stays discoverable. The package rename does not move or rename existing logs.

## Run directory layout

```
<logs-root>/math_rl/<run_name>/
├── config.json                    # shared training configuration (chz-rendered)
├── run_meta.json                  # launch metadata, updated after session creation
├── metrics.jsonl                  # one JSON object per log event; steps can repeat
├── checkpoints.jsonl              # ledger row only after requested saves complete
├── checkpoint_evaluations.jsonl   # evaluation tied to durable state, when emitted
├── code.diff                      # code-change snapshot written by the logger
├── logs.log                       # human-readable log tail
├── train_iteration_*.html         # interactive HTML logs of sampled trajectory groups
└── eval_test_iteration_*.html     # eval-run trajectory groups
```

### `config.json`

The configuration written by the shared logger, including defaults passed into the training loop. It is not necessarily every field on the launcher's CLI. Compare it with `run_meta.json` and retain your exact launch command when reproducing a run. Absence of this file can mean failure before the shared loop configured logging.

### `run_meta.json`

Launch metadata. Math writes an initial version **before** session creation with `azure.session_id=null`, then updates it after creation. A file's existence alone is not proof that a model loaded. Other launchers have their own schema/timing. Common fields include:

- `azure.session_id` — the `session_<...>` identifier of the Azure session. Use this as the `<session_id>` half of `load_checkpoint_path` when starting a new continual-FT run from this run. To discover session IDs without grepping `run_meta.json` files, see [session-management.md](./session-management.md).
- `azure.reference_session_id` — a separate teacher/KL-reference session when one is created; `null` otherwise. Distillation records both its student and teacher sessions here.
- `azure.session_ids` — staged visual-spatial training records its `sft` and `rft` session IDs in the parent run's `run_meta.json`; each stage also writes its own `run_meta.json` under `sft/` or `rft/`.
- `azure.endpoint_host`, `azure.auth` — wire-level info for debugging.
- `azure.from_checkpoint` — `{source_session_id, checkpoint_id}` if this run was launched from a checkpoint, otherwise `null`. See [continual-fine-tuning.md](./continual-fine-tuning.md).
- `model.*` — model name, tokenizer, LoRA rank.

### `metrics.jsonl`

One JSON object per line, written per logging event. Training and evaluation can be logged separately with the **same step**, and not every row has every key. Preserve file order and choose the phase being compared. Keys commonly emitted by shared RL include:

- `train/env/all/correct`, `train/env/all/reward/total`, `train/env/all/format` — graded training rollouts; `correct` and `format` depend on the task. Total reward includes transition and final group rewards.
- `test/env/all/correct`, `test/env/all/reward/total`, etc. — held-out eval rollouts when available; check `test/env/all/total_episodes` and dropped-group warnings.
- `train/env/all/ac_tokens_frac_at_max`, `ac_tokens_p95`, etc. — completion-budget diagnostics, not exact service finish reasons.
- `optim/*` and backend-reported stats — the exact set depends on the service/model.
- `time/*` — wall-clock seconds for each phase of the step (sample, forward_backward, optim_step, …).

SFT instead records `train_mean_nll`, `test/nll`, sequence/loss-token counts, and its learning-rate/timing fields. Standalone NER evaluation writes `metrics.json` and a manifest rather than this training layout. Use the [evaluation guide](./evaluation-and-inference.md#which-metric-means-what) for interpretation; reward is not universally accuracy, and missing results are not zero.

> [!NOTE]
> All keys under the `time/` prefix are wall-clock **seconds**, not steps.

### `checkpoints.jsonl`

One row per checkpoint save. Schema:

```jsonc
{"name": "4", "batch": 4, "state_path": "session_<source>/4", "sampler_path": "4"}
{"name": "final", "batch": 12, "state_path": "session_<source>/final", "sampler_path": "final"}
```

- `name` — checkpoint name (the numbered batch, or `final`).
- `batch` — the resume cursor, not a universal optimizer-step count. SFT also records `epoch`; dynamic-sampling runs can add `prompt_cursor`, and async runs can add rollout-seed cursor fields. Preserve the whole row and original dataset/config.
- `state_path` — SDK-supported `<session>/<checkpoint>` pointer to the server-side **training** checkpoint (LoRA weights + optimizer + step). This is what `load_checkpoint_path` / auto-resume bootstraps from.
- `sampler_path` — **persisted** sampler-format weights saved at this boundary, without optimizer state. It can be a bare identifier bound to the source session, not a portable training path. The separate per-step ephemeral sampler refresh does not write this ledger.

Fields depend on `kind="state"`, `"sampler"`, or `"both"`; a sampler-only row is not resumable training state. The example is schematic: use the actual returned paths and recorded `azure.session_id`, not a constructed filename. Rows are appended after the requested save results finish, so a failure partway through `kind="both"` may leave a remote save without a corresponding local row. Inspect [service checkpoint metadata](./session-management.md#list-checkpoints-for-a-session) if recovery evidence is incomplete.

To resume a crashed or interrupted run from one of these, see [Resuming after an interruption](./training.md#resuming-after-an-interruption). To start a *new* run from a previous run's weights, see [continual-fine-tuning.md](./continual-fine-tuning.md).

### `train_iteration_*.html` / `eval_test_iteration_*.html`

Per-iteration interactive HTML pages showing each sampled trajectory group with its tokens, rewards, and grading verdicts. Open them directly in a browser, or via the dashboard's **Per-iteration artifacts** card.

## Local artifacts versus remote state

| Item | Where it lives | What local deletion does |
|---|---|---|
| Config, metrics, ledger, HTML, and code diff | Client's run directory | Removes evidence/recovery cursors; does not release remote compute |
| Live training/sampling state | Service session / compute | Deleting files or stopping the dashboard does not close it |
| Saved training/sampler checkpoints | Service-managed storage | Removing a ledger does not delete the remote checkpoint |

Preserve the source session ID and completed training-checkpoint name/path before changing machines. A saved training checkpoint can initialize a new compatible session; restoring the same dataset cursor additionally requires the local ledger/config. See [recovery](./training.md#resuming-after-an-interruption), [continual training](./continual-fine-tuning.md), and [checkpoint sampling](./evaluation-and-inference.md).

Persistence is not a retention guarantee. The pinned adapter ignores `ttl_seconds`; confirm service retention/deletion policy before relying on a checkpoint. Closing/unloading an owned session is different from deleting it; do not use a delete operation as routine cleanup because it can remove checkpoints too.

Logs, HTML, manifests, and `code.diff` can contain dataset text, prompts, responses, identifiers, or local changes. Keep the run directory access-controlled, exclude private data from version control, and review/redact artifacts before sharing. The dashboard does not authenticate viewers or redact contents ([security](./dashboard.md#security)).

## Archive and deletion decisions

Use your project's retention/data policy, not an assumed checkpoint TTL or a
successful process exit, to decide what to keep:

| Item | Preserve for | Before deletion or sharing |
|---|---|---|
| Config, dataset manifest/revision, metrics, checkpoint ledger, and source session ID | Reproducibility, evaluation evidence, and recovery cursor | Keep an access-controlled backup if recovery or review is still needed; a remote weight save does not replace these files |
| Prompt/response HTML, logs, `code.diff`, and notebook outputs | Failure analysis and task-error inspection | Review for private text, tokens, local paths, and unrelated code; retain only as policy permits and redact before sharing |
| Completed remote training checkpoint | Compatible restart/recovery | Confirm save completion, intended owner/project, access, retention, and whether it is still the only recoverable state |
| Remote sampler checkpoint | Sampling-format state for its supported session context | Keep its source session association; sampler-only state is not optimizer recovery or a deployment |
| Live session or separate serving resource | Only the approved active workload | Save required state, stop using it, and obtain approval for the specific unload/cancel/delete action; inspect the result |

Unload and deletion are different operations. Do not delete the source
session/checkpoints as a shortcut to releasing compute while you still need
their saved state. Local data removal is likewise separate from service-side
retention. Confirm the owner's deletion procedure and resulting state instead
of assuming the cookbook's ignored files or `ttl_seconds` implement it.
