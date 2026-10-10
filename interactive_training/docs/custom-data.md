# Adapt a recipe to your own data

Use the existing dataset and grading interfaces without changing the shared training algorithms. The data-building and grading checks below are **local** and do not create an Azure session; tokenizer downloads may still require network access. The separately marked [SFT launch](#3-wire-the-builder-into-an-sft-launcher) is an optional paid operation: warn before running it, without a separate spending-approval prompt.

## Choose the smallest change

| Your data | Existing extension point | Start from |
|---|---|---|
| User/assistant conversations | `FromConversationFileBuilder` / `ChatDatasetBuilder` | [Supervised chat data](#supervised-chat-data) |
| Prompt plus verifiable answer | `ProblemEnv` + `ProblemGroupBuilder` + `RLDatasetBuilder` | [RL data and rewards](#reinforcement-learning-data-and-rewards) |
| Multi-turn tool interactions | `Env.initial_observation`, `Env.step`, group grading | [Tool environment](../interactive_training/recipes/tool_rl/env.py) or [grounded NER](../interactive_training/recipes/tool_ner_rl/README.md) |
| Chosen/rejected response pairs | Preference recipe's dataset builder | [DPO data contract](../interactive_training/recipes/preference/dpo/README.md#task-and-data) |
| Images with labeled responses/rewards | Image-capable renderer and dataset builder | [Custom visual-spatial data](../interactive_training/recipes/visual_spatial/README.md) and [vision prerequisites](./supported_models.md#vision-fine-tuning) |

The math CLI's `env` selects built-in builders; the Tulu3 CLI selects the Tulu3 builder. Neither accepts an arbitrary custom data filename just because another recipe does. A custom builder must be wired into your own launcher using the documented interfaces. This page does not add a new CLI or new recipe.

## Supervised chat data

### 1. Prepare conversation JSONL

`FromConversationFileBuilder` reads **one JSON object per line**, each containing a `messages` list. Save UTF-8 JSONL in an access-controlled location outside version control. Do not use one enclosing JSON array or blank lines.

```jsonl
{"messages":[{"role":"system","content":"Classify the request as billing or technical. Reply with one label."},{"role":"user","content":"My invoice was charged twice."},{"role":"assistant","content":"billing"}]}
{"messages":[{"role":"system","content":"Classify the request as billing or technical. Reply with one label."},{"role":"user","content":"The app fails when I sign in."},{"role":"assistant","content":"technical"}]}
{"messages":[{"role":"system","content":"Classify the request as billing or technical. Reply with one label."},{"role":"user","content":"I need a copy of my receipt."},{"role":"assistant","content":"billing"}]}
{"messages":[{"role":"system","content":"Classify the request as billing or technical. Reply with one label."},{"role":"user","content":"I cannot finish a password reset."},{"role":"assistant","content":"technical"}]}
```

These four synthetic rows illustrate the schema, not a useful training corpus or benchmark. Provide correct, representative assistant labels; de-identify private data and check its license/consent before submitting it remotely. Text content can be a string. For images or tool messages, use the renderer's documented content contract instead of inserting arbitrary objects.

### 2. Render and inspect locally

Put the file at `./data/conversations.jsonl` relative to the cookbook directory. The following is a complete **local data-building example** once that file and dependencies exist:

```python
from interactive_training.renderers import TrainOnWhat
from interactive_training.supervised.data import FromConversationFileBuilder
from interactive_training.supervised.types import ChatDatasetBuilderCommonConfig

builder = FromConversationFileBuilder(
    common_config=ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer="Qwen/Qwen3.8-27B",
        renderer_name="qwen3_8_low_reasoning",
        max_length=2048,
        batch_size=2,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        fail_on_truncation=True,
    ),
    file_path="./data/conversations.jsonl",
    test_size=1,
    shuffle_seed=42,
)
training, evaluation = builder.build()
if len(training) == 0 or evaluation is None:
    raise ValueError("Need a full training batch and a non-empty held-out split")
for split in (training, evaluation):
    for index in range(len(split)):
        batch = split.get_batch(index)
        assert batch
        for datum in batch:
            targets = datum.loss_fn_inputs["target_tokens"].data
            weights = datum.loss_fn_inputs["weights"].data
            assert len(targets) == len(weights)
            assert any(weight > 0 for weight in weights), "No assistant training tokens"
print("Full training batches:", len(training), "evaluation batches:", len(evaluation))
```

Expected for the four-row file: **one full training batch and one evaluation batch**. The builder shuffles, reserves one evaluation row, then trains in batches of two. Its dataset length uses **floor division**: one of the three remaining rows is outside the full training batches for that epoch. Choose batch sizes and split sizes deliberately; a dataset smaller than one batch can silently yield no training batches.

Each example here has one assistant response, so `LAST_ASSISTANT_MESSAGE` trains that response. For multi-response conversations, this Qwen3.8 renderer does not guarantee that an earlier assistant turn has the same token prefix as a separately generated response. Split examples deliberately when training earlier turns; do not switch to `ALL_ASSISTANT_MESSAGES` merely to suppress a missing-data problem.

`test_size` is a **row count**, not a ratio. If it is zero or at least the total row count, this builder returns **no evaluation dataset**. A seeded random split is convenient for prototyping but is not a substitute for grouping by customer/document/time to avoid leakage. For explicit train/validation files or group-aware splits, implement a `ChatDatasetBuilder` that returns the two separate datasets.

`fail_on_truncation=True` rejects overlength rendered rows rather than silently truncating them. Set `model_context_length` to the deployment's verified limit when known; do not invent a universal model limit. This example validates the batches it builds, not unused tail rows. Inspect decoded data and masks with the existing [display helpers](../interactive_training/display.py) before a remote run.

For an SFT launcher that exposes `preflight_dataset`, enable it to render its
training and evaluation batches before creating a remote session. The programmatic
example below sets `preflight_dataset=True` and prepares them locally. Preflight
reports the split and failing batch when rendering raises a validation error;
context and truncation checks only run if the builder forwards
`model_context_length` and `fail_on_truncation`. Floor-batched tail rows are not
scanned. The conversation-file and
[Tulu3 builders](../interactive_training/recipes/tulu3_sft/README.md#sft-on-tulu3)
both forward these checks; custom builders must do the same to enforce them.

### 3. Wire the builder into an SFT launcher

The extension is **programmatic**, not an extra flag on the Tulu3 CLI. The
complete example below composes the existing [SFT configuration and loop](../interactive_training/supervised/train.py)
with the [Azure training adapter](../interactive_training/rl/train_azure.py).
It does not modify the shared loop or install a new command. Run the blocks
in order in your own private driver from `fine-tuning/interactive_training`.

Unlike the batch-size-two inspection above, this first-run example uses
**batch size one** so every training row is represented during preflight.
It reserves one evaluation row, checks positive assistant loss weights, and
reuses the prepared datasets after allocation instead of reloading the file.
For the four synthetic rows, expect three training batches and one evaluation
batch; at most two optimizer steps execute. This is a workflow check, not a
recommended production dataset size, batch size, learning rate, or budget.

**Define and prepare locally:** defining either function does not call Azure.
`prepare_custom_sft()` can download public tokenizer files and renders the local
data. Keep private datasets under the ignored `data/` location (or outside the
checkout); private run metadata goes under the ignored `runs/` location.

```python
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path

from azure.ai.finetuningsessions.aio import FineTuningSessionClient
from azure.ai.finetuningsessions.models import LoRAConfig
from azure.identity.aio import DefaultAzureCredential
from interactive_training import checkpoint_utils
from interactive_training.renderers import TrainOnWhat
from interactive_training.rl.train_azure import AzureSDKTrainingClient
from interactive_training.supervised import train
from interactive_training.supervised.data import FromConversationFileBuilder
from interactive_training.supervised.types import ChatDatasetBuilderCommonConfig
from interactive_training.tokenizer_utils import get_tokenizer

def prepare_custom_sft(
    file_path="./data/conversations.jsonl",
    log_path="./runs/own-text-check",
):
    source = Path(file_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    builder = FromConversationFileBuilder(
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer="Qwen/Qwen3.8-27B",
            renderer_name="qwen3_8_low_reasoning",
            max_length=2048,
            batch_size=1,
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            fail_on_truncation=True,
        ),
        file_path=str(source), test_size=1, shuffle_seed=42,
    )
    config = train.Config(
        log_path=log_path, model_name="Qwen/Qwen3.8-27B",
        dataset_builder=builder, lora_rank=32,
        learning_rate=1e-4, num_epochs=1, max_steps=2, seed=42,
        eval_every=1, save_every=1, pipeline_depth=1,
        preflight_dataset=True, sanitize_logs=True,
    )
    prepared = train.prepare_datasets(config)
    training, evaluation = prepared
    if not len(training) or evaluation is None or not len(evaluation):
        raise ValueError("Need training batches and a non-empty evaluation split")
    for split in prepared:
        for index in range(len(split)):
            for datum in split.get_batch(index):
                targets = datum.loss_fn_inputs["target_tokens"].data
                weights = datum.loss_fn_inputs["weights"].data
                if len(targets) != len(weights) or not any(w > 0 for w in weights):
                    raise ValueError("Invalid target/mask lengths or no assistant loss tokens")
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError("Dataset changed during validation; prepare it again")
    manifest = {
        "dataset_sha256": digest, "split_seed": 42,
        "training_batches": len(training), "evaluation_batches": len(evaluation),
        "renderer_name": "qwen3_8_low_reasoning",
    }
    return config, prepared, manifest

async def run_custom_sft(config, prepared, manifest):
    if not config.max_steps or config.max_steps < 1:
        raise ValueError("Use a positive first-run step cap")
    endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "").strip()
    if not endpoint:
        raise ValueError("Set AZURE_AI_PROJECT_ENDPOINT to the intended project URL")
    tokenizer = get_tokenizer(config.model_name)
    run = Path(config.log_path)
    run.mkdir(parents=True, exist_ok=False)  # Never overwrite or silently resume.
    logging.warning("This SFT run uses paid cloud resources and may incur charges.")
    async with DefaultAzureCredential() as credential:
        async with FineTuningSessionClient(
            endpoint=endpoint, credential=credential,
            credential_scopes=["https://ai.azure.com/.default"],
        ) as client:
            session_id = await client.create_session(
                base_model=config.model_name,
                lora_config=LoRAConfig(rank=config.lora_rank),
                type="training", timeout_sec=600.0,
            )
            workload_error = None
            try:
                (run / "session.json").write_text(json.dumps({
                    **manifest, "session_id": session_id,
                    "model_name": config.model_name, "lora_rank": config.lora_rank,
                }, indent=2), encoding="utf-8")
                trainer = AzureSDKTrainingClient(client, session_id, tokenizer)
                await train.main(config, trainer, prepared_datasets=prepared)
                saved = checkpoint_utils.get_last_checkpoint(config.log_path)
                if saved is None or saved.get("name") != "final":
                    raise RuntimeError("No completed final training checkpoint in the ledger")
                return saved["state_path"]
            except BaseException as exc:
                workload_error = exc
                raise
            finally:
                try:
                    await client.close_session(session_id)
                except BaseException:
                    if workload_error is None:
                        raise
                    logging.exception("Session cleanup failed; inspect the owned session")
```

Run only the local preparation first:

```python
config, prepared, manifest = prepare_custom_sft()
print("Validated batches:", manifest["training_batches"], manifest["evaluation_batches"])
```

Before the optional remote launch, confirm [project permissions and model
capacity](./auth.md#confirm-access-before-spending) and review the data and
held-out split. The helper warns before allocating paid compute; no separate
budget confirmation is required. This helper uses
identity authentication intentionally; setting an API-key environment variable
does not change its credential code. It leaves the service's training tier
unspecified rather than assuming a universally available or cheaper tier.

**Optional remote launch after validation:**

```python
checkpoint_path = asyncio.run(run_custom_sft(config, prepared, manifest))
print("Completed training checkpoint:", checkpoint_path)  # Keep this identifier private.
```

Inside an already-running notebook event loop, use `await run_custom_sft(...)`
instead of nesting `asyncio.run`. **SFT `max_steps=0` disables the cap; it is
not evaluation-only.** The step cap does not bound loading, sampler warm-up,
evaluation, checkpoint saves, cleanup time, or price. `sanitize_logs=True`
suppresses the SFT loop's example-content logging, not all private metadata
or the logger's `code.diff`; review artifacts before sharing.

### 4. Verify the result and hand it off

For this fresh run, inspect `config.json`, `metrics.jsonl`, and
`checkpoints.jsonl` under `./runs/own-text-check`. The helper's private
`session.json` records the dataset hash, split seed, model/rank, renderer, and
owned session ID; it is separate from standard launchers' `run_meta.json`.
Initial and final `test/nll` rows measure the same held-out split. The final
row may share a step number with a training row; retain file order. A lower
NLL is not a classification-accuracy or safety guarantee.

Use the returned `state_path` with the [checkpoint response helper](./evaluation-and-inference.md#use-a-training-checkpoint-after-the-original-session-closes)
and the original model/renderer/rank. Supply an unseen task prompt, not a
training label, and compare with a baseline under the same sampling contract.
Verify [remote cleanup](./session-management.md#free-gpu-memory-begin_unload)
separately; successful client closure does not prove compute was released.

If training fails, a prior completed training save may remain in the ledger;
the example does not guarantee a final save on exception. If session creation
times out before returning an ID, inspect the intended project for the partial
owned session before retrying. This helper deliberately rejects an existing
run directory and has **no automatic resume mode**. Preserve the artifacts;
for recovery or continual training, follow the [checkpoint/cursor contract](./training.md#resuming-after-an-interruption)
and create a compatible session from the saved state, not base weights with
an old cursor. See [contributing](./contributing.md#adding-a-recipe) before
promoting a private driver to a maintained recipe.

## Reinforcement learning data and rewards

For a single-turn exact-answer task, reuse [`ProblemEnv`](../interactive_training/rl/problem_env.py) rather than implementing the rollout loop. It handles model-specific rendering, response parsing, logging, and terminal `StepResult` construction.

### 1. Define the task contract

Decide the prompt, acceptable answer format, grader, and failure behavior **before** training. Example local records:

```jsonl
{"prompt":"Reply with one label: Is an invoice billing or technical?","answer":"billing"}
{"prompt":"Reply with one label: Is an app crash billing or technical?","answer":"technical"}
```

Do not reveal the ground-truth answer through the generation prompt or a tool result. Separate held-out records before tuning prompts or rewards. For custom labels, test acceptable variants and deliberately wrong/empty answers before spending compute.

### 2. Reuse the single-turn interfaces

The following local template adapts records to the **existing** RL contracts. It does not start training:

```python
import math
from functools import partial

from interactive_training.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from interactive_training.rl.types import RLDataset

class LabelEnv(ProblemEnv):
    def __init__(self, row, renderer):
        super().__init__(renderer)
        self.row = row

    def get_question(self) -> str:
        return self.row["prompt"]

    def check_answer(self, sample_str: str) -> bool:
        return sample_str.strip().casefold() == self.row["answer"].casefold()

    def check_format(self, sample_str: str) -> bool:
        return sample_str.strip().casefold() in {"billing", "technical"}

    def get_reference_answer(self) -> str:
        return self.row["answer"]

class LabelDataset(RLDataset):
    def __init__(self, rows, renderer, batch_size=8, group_size=4):
        if not rows or batch_size < 1 or group_size < 1:
            raise ValueError("Need rows and positive batch/group sizes")
        self.rows = rows
        self.renderer = renderer
        self.batch_size = batch_size
        self.group_size = group_size

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)

    def get_batch(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        rows = self.rows[index * self.batch_size:(index + 1) * self.batch_size]
        return [
            ProblemGroupBuilder(
                env_thunk=partial(LabelEnv, row, self.renderer),
                num_envs=self.group_size,
                dataset_name="labels",
            )
            for row in rows
        ]
```

`make_envs` creates **one independent environment per completion**, all bound to the same task. The parent `ProblemEnv.step` produces:

$$r = \mathrm{correct} + 0.1\,(\mathrm{format}-1).$$

Thus an invalid-format wrong answer can have reward **−0.1**, not just zero; reward and correctness are not interchangeable. `ProblemGroupBuilder.compute_group_rewards` contributes zero additional final reward because `step` already grades the answer. Do not award the same reward a second time.

Local grader checks can be performed before any SDK session exists:

```python
# Run after the definitions above; no renderer is used by these checks.
env = LabelEnv({"prompt": "Classify an invoice.", "answer": "billing"}, renderer=None)
assert env.check_answer("Billing")
assert not env.check_answer("technical")
assert not env.check_format("")
assert not env.check_format("billing because it is an invoice")
```

### 3. Wire the dataset without modifying the loop

Implement an `RLDatasetBuilder` whose async `__call__` returns `(training_dataset, evaluation_dataset)` using the same selected tokenizer/renderer. Training groups may have four or eight completions; evaluation commonly uses one completion per held-out prompt. Instantiate a fresh group/environment for every rollout; do not share mutable environment state.

Follow the [`ArithmeticDatasetBuilder`](../interactive_training/recipes/math_rl/arithmetic_env.py) for construction and the [math Azure launcher](../interactive_training/recipes/math_rl/train_azure.py) for wiring the builder into `rl.train.Config` and `rl.train_azure.main`. Return an actual held-out dataset: the toy arithmetic builder returns `None` for evaluation and cannot demonstrate held-out improvement.

For multi-turn tasks, [`Env`](../interactive_training/rl/types.py) additionally requires:

- `initial_observation()` → `(ModelInput, StopCondition)`.
- `step(action)` → `StepResult`, including reward, done flag, next observation, and next stop condition. A stop condition is a list of stop strings or token IDs, **not an enum**.
- Optional `compute_group_rewards(trajectories, envs)` → one `(reward, metrics)` pair per trajectory in the same order. Final group rewards are added to transition rewards; avoid double-counting.

Use [code/tool examples](./recipes.md#choose-a-path) only after approving their external execution and security model. Grader exceptions, missing labels, timeouts, and API outages must not be silently converted into trustworthy negative rewards.

## Adaptation checklist

- Keep base model, tokenizer, renderer, and checkpoint LoRA settings compatible.
- Validate schema and decoded prompts locally; check labels, loss masks, format rules, context budget, and tails smaller than a batch.
- Keep training, tuning/validation, and final test data disjoint at the appropriate document/customer/time level.
- Pin dataset revision/split and store an ID/hash manifest; record grader version and sampling settings.
- Confirm data-use terms, tool permissions, and sensitive artifact handling.
- Run a bounded workflow first, then [compare before/after](./evaluation-and-inference.md); do not claim improvement from training reward alone.