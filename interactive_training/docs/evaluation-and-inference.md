# Evaluate improvement and use a checkpoint

Close the loop from **train → measure → save → sample** using the existing cookbook and published SDK. A passing smoke run proves infrastructure worked; it does not establish model quality, checkpoint portability, or production deployment.

## Set the comparison contract first

Before training, fix and record:

- The base model, tokenizer, renderer/reasoning mode, and LoRA configuration.
- A held-out dataset revision, split, and stable example IDs; keep it separate from reward/prompt tuning.
- The prompt template, grader version, tools/retrieval data, and failure handling.
- The completion budget, temperature, stop criteria, samples per task, and seeds where available.
- The primary metric and acceptable quality/cost/regression trade-off.

Evaluate the baseline and checkpoint with **the same contract**. Changing reasoning mode, output length, dataset selection, or tools at the same time confounds the comparison. Match the number of successfully scored examples, report dropped groups, inspect errors, and repeat noisy comparisons. A seed can reduce variation; it is not a guarantee of identical service outputs.

## Read the existing run's measurements

Use the [quickstart](./quickstart.md) with evaluation enabled. For a fresh synchronous math run, evaluation at step 0 measures the starting weights before the first update. Periodic evaluations are pre-update measurements at their labeled iteration; the final evaluation follows the final save. If the run started from a checkpoint, step 0 is that checkpoint—not the original base model.

### Which metric means what?

| Recipe / metric | Interpretation | Limitation |
|---|---|---|
| Math `test/env/all/correct` | Fraction of graded held-out completions with a correct answer | Token-budget/format failures count as wrong; check coverage and format |
| RL `test/env/all/reward/total` | Mean total trajectory reward, including shaping/final rewards | Not necessarily binary accuracy; reward scales vary by task |
| RL `test/env/all/format` | Task/parser format compliance where emitted | A well-formed answer can still be wrong |
| RL `test/env/all/total_episodes` | Completed evaluation trajectories | Compare with the intended number; missing/dropped groups bias a comparison |
| RL `test/env/all/ac_tokens_frac_at_max` | Fraction of turns close to the configured completion cap | A truncation warning signal, not an exact finish-reason counter |
| SFT `test/nll` | Weighted mean held-out assistant-token negative log-likelihood; lower is better | Does not establish task accuracy, safety, or generation quality |
| SFT `test/num_sequences`, `test/num_loss_tokens` | Evaluation rows and supervised-token coverage | Compare the same rendered split/mask; a smaller denominator can invalidate a comparison |
| Standalone NER `base/eval/micro_f1` | Exact-span micro-F1 from the evaluation artifacts | The `base/` prefix is also used when evaluating a loaded checkpoint |

These keys are emitted by the [RL metric implementation](../interactive_training/rl/metric_util.py), [SFT evaluator](../interactive_training/supervised/nll_evaluator.py), and [NER artifacts](../interactive_training/recipes/tool_ner_rl/eval_artifacts.py). A generic `test/env/all/reward` key is **not** the shared RL total-reward key. No particular quality delta is promised for a two-step run.

**Interpret SFT NLL as likelihood of the reference tokens, not a percentage
of good responses.** The evaluator weights selected assistant positions;
changing tokenization, mask, context budget, or the held-out references changes
what is measured. A decrease is useful evidence under a fixed contract, but
generation-based task accuracy, harmful-output regressions, and latency/cost
need their own checks. Do not convert an NLL difference into a claimed
percentage improvement in user-task quality.

### Decide whether the comparison supports improvement

| Check | Acceptable evidence | If missing |
|---|---|---|
| Same task and examples | Matching revision/split/hash or stable document IDs, prompt, grader, renderer, and tools | Treat the runs as different experiments, not a before/after delta |
| Complete coverage | Intended scored examples/loss tokens, with failures and drops reported | Mark the result partial; missing metrics are not zero |
| Comparable sampling | Same completion budget, reasoning mode, temperature, and samples per task | Re-evaluate under a common contract before attributing changes to training |
| Useful outcome | Held-out task outputs/metric plus important negative cases | Training reward or NLL alone does not establish application improvement |
| Credible uncertainty | Enough independent cases for the task, repeated noisy comparisons where needed | Label a tiny sample as a smoke test; do not promise significance |
| Cost and regression boundary | Measured timing/usage when available and explicitly unresolved pricing/cleanup | Do not convert wall-clock time into a bill or assume lower loss saves money |

### Inspect local before/after values

This example reads local files only. Run it from the cookbook directory after the quickstart:

```python
import json
from pathlib import Path

run = Path("./runs/first-math-check")
metric = "test/env/all/correct"
rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
evaluations = [row for row in rows if metric in row]
if len(evaluations) < 2:
    raise ValueError("Need initial and later held-out evaluation rows; check eval settings/failures")
before, after = evaluations[0], evaluations[-1]
expected_episodes = 8  # Requested by the fresh quickstart; change for your approved scope.
if any(row.get("test/env/all/total_episodes") != expected_episodes for row in (before, after)):
    raise ValueError("Incomplete or different evaluation coverage; inspect dropped groups")
print("Initial:", before.get("step"), before[metric], "episodes:", before.get("test/env/all/total_episodes"))
print("Final:", after.get("step"), after[metric], "episodes:", after.get("test/env/all/total_episodes"))
print("Delta (percentage points):", 100 * (after[metric] - before[metric]))
ledger = [json.loads(line) for line in (run / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
final = next(row for row in reversed(ledger) if row.get("name") == "final" and "state_path" in row)
print("Training checkpoint:", final["state_path"])
```

For SFT, select `test/nll`, replace the episode check with matching positive
`test/num_sequences` and `test/num_loss_tokens`, and interpret `after - before`
as an NLL difference, **not percentage points**. A metrics file can have more
than one row per step (including a final evaluation); preserve file order and
don't assume every row includes test results. Runs resumed into the same
directory need careful phase selection: the first row may belong to an earlier
execution. Equal counts alone do not prove equal example IDs or graders.

The dashboard can visualize these curves, but compare the underlying config, example coverage, and outputs as well. Missing evaluation metrics mean "not measured successfully", not zero.

## Standalone before/after evaluation: grounded NER

The existing [NER evaluator](../interactive_training/recipes/tool_ner_rl/eval_azure.py) can evaluate without optimizer updates. It still **creates a remote session, samples, downloads public data, and may incur charges**. Use only for the NER task/compatible checkpoint; it is not a universal evaluation CLI.

For a small OpenPII validation comparison, authenticate and replace the endpoint placeholder. Run from the cookbook directory; commands below use Bash continuation.

**Baseline:**

```bash
python -m interactive_training.recipes.tool_ner_rl.eval_azure \
    project_endpoint="<your-project-endpoint>" model_name="Qwen/Qwen3.8-27B" \
    lora_rank=16 dataset_name=openpii dataset_split=validation \
    max_examples=8 sample_pool_size=200 sample_seed=42 samples_per_task=1 \
    max_tokens=768 second_turn_max_tokens=768 \
    log_path=./runs/ner-before
```

**Checkpoint:** use a training checkpoint from a NER run with the same base model and LoRA rank (replace the placeholder with its actual session/name):

```bash
python -m interactive_training.recipes.tool_ner_rl.eval_azure \
    project_endpoint="<your-project-endpoint>" model_name="Qwen/Qwen3.8-27B" \
    lora_rank=16 load_checkpoint_path="session_<source>/final" \
    dataset_name=openpii dataset_split=validation \
    max_examples=8 sample_pool_size=200 sample_seed=42 samples_per_task=1 \
    max_tokens=768 second_turn_max_tokens=768 \
    log_path=./runs/ner-after
```

After successful evaluation, each directory contains `manifest.json`, `metrics.json`, `run_meta.json`, `per_document.jsonl`, `error_summary.json`, `error_review_sample.json`, `protocol_errors.json`, and `eval_base.html`. A failed run can leave only some artifacts. The CLI refuses to overwrite a non-empty directory. Compare manifest dataset/revision/split, selected document IDs, sampling settings, and errors before reading `base/eval/micro_f1`. Both runs use `base/` keys; the run metadata identifies the loaded checkpoint.

Eight examples test the evaluation pipeline, not statistically credible improvement. Increase the approved evaluation scope for a real comparison, inspect the reported uncertainty/error breakdown, and reserve an untouched final test set. Cleanup is attempted in `finally`; verify failures as in [session management](./session-management.md).

> [!IMPORTANT]
> **Do not use `max_steps=0` on a training launcher as a universal evaluation mode.** SFT interprets zero as no step cap. Math creates a remote session and does not provide a general eval-only CLI. For other RL tasks, compose `RLTestSetEvaluator` with the task's held-out builder and sampling client; for SFT, use `NLLEvaluator` on held-out data plus task-specific generation evaluation. These are programmatic APIs, not new documented CLI flags.

## Persistent versus ephemeral saves

| Purpose | Raw async SDK call | Completion / reuse |
|---|---|---|
| Preserve training state | `await client.save_weights_async(session_id, name)` | Returns a task; await it, keep source session/name, use `FromCheckpoint` for a new session |
| Persist sampler-format weights | `await client.save_weights_for_sampler_async(session_id, name)` | Returns a task; await it, keep **session ID and checkpoint ID** together |
| Per-step rollout refresh | `await client.save_weights_and_get_sampling_client_async(session_id, name)` | Returns a task; ephemeral sync, **not blob persistence**, despite its name |

The cookbook adapter has different signatures/results: `trainer.save_weights_for_sampler_async(name)` returns a future exposing `result_async()`, while its `save_weights_and_get_sampling_client_async()` awaits the sync and returns a wrapper sampling client. See [the exact API boundary](./sdk-reference.md#persistent-versus-ephemeral-checkpoints).

## Sample a persistent checkpoint in its live session

For a raw async SDK client and a **live, compatible session**, a save followed by sampling looks like this. This helper uses remote compute and may create substantial sampler allocations; do not run it just to validate imports, or indiscriminately during an SFT workload.

```python
from azure.ai.finetuningsessions.models import SamplingParams

async def persist_and_sample(client, session_id, renderer):
    save_task = await client.save_weights_for_sampler_async(session_id, "example-sampler")
    saved = await save_task  # Wait until the save completes, not only the POST.
    prompt = renderer.build_generation_prompt([
        {"role": "user", "content": "What is 2 + 2? Reply with a boxed answer."}
    ])
    result = await client.sample(
        session_id,
        prompt,
        SamplingParams(
            max_tokens=1200, temperature=1.0, top_p=1.0, top_k=-1,
            seed=42, stop_criteria=renderer.get_stop_sequences(),
        ),
        checkpoint_id=saved.checkpoint_id,
        num_samples=1,
    )
    return saved.checkpoint_id, result
```

Store the returned checkpoint ID **with its session ID**, not as a standalone portable model path. The local `sampler_path` can be just a checkpoint identifier bound by the adapter to its current session. `create_sampling_client` only constructs a local wrapper; it does not create/reload a remote session or make another session's checkpoint available.

The operation-group alternative is `client.checkpoints.begin_save_sampler_weights` with `SaveSamplerWeightsRequest(path=name, seq_id=0)` and **without** `sampling_session_seq_id`; await the LRO result. Setting `sampling_session_seq_id` requests an ephemeral update. All operation-group calls require the preview and API-version arguments shown in [session management](./session-management.md).

Persistence means the save writes remote storage rather than only refreshing in-memory weights. It does **not** guarantee indefinite retention, a permanent live sampling endpoint, or cross-session loading of arbitrary sampler IDs. The preview cookbook's `ttl_seconds` adapter argument is not forwarded to the pinned SDK; confirm retention with the service owner.

## Use a training checkpoint after the original session closes

The documented restart path is **saved training state → new compatible session → sampler sync → sample**. Use the source session ID/name from the training checkpoint ledger or service checkpoint listing, even if the original session is no longer live. Do not invent a download/export API or attach a bare sampler ID to a different session.

Below is a complete helper using the published async SDK and identity authentication. Set `AZURE_AI_PROJECT_ENDPOINT`, authenticate as in [auth.md](./auth.md), and call it from your script's async entry point. Defaults match the quickstart's Qwen3.8/rank32; override them **only to match the checkpoint's original configuration**. Supply `messages` for your actual held-out task; the default arithmetic prompt checks plumbing, not an own-data model's usefulness. The helper allocates paid remote compute; it is not an offline check.

```python
import os
from azure.ai.finetuningsessions.aio import FineTuningSessionClient
from azure.ai.finetuningsessions.models import LoRAConfig, SamplingParams
from azure.identity.aio import DefaultAzureCredential
from interactive_training.renderers import get_renderer
from interactive_training.tokenizer_utils import get_tokenizer

async def sample_saved_training_state(
    checkpoint_path,
    model_name="Qwen/Qwen3.8-27B",
    renderer_name="qwen3_8_low_reasoning",
    lora_rank=32,
    messages=None,
):
    endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "").strip()
    if not endpoint:
        raise ValueError("Set AZURE_AI_PROJECT_ENDPOINT to your project endpoint")
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer=tokenizer)
    prompt = renderer.build_generation_prompt(messages if messages is not None else [
        {"role": "user", "content": "What is 2 + 2? Reply with a boxed answer."}
    ])
    async with DefaultAzureCredential() as credential:
        async with FineTuningSessionClient(
            endpoint=endpoint,
            credential=credential,
            credential_scopes=["https://ai.azure.com/.default"],
        ) as client:
            if checkpoint_path is None:
                session_id = await client.create_session(
                    base_model=model_name,
                    lora_config=LoRAConfig(rank=lora_rank),
                    type="training",
                )
            else:
                session_id = await client.create_session_from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    base_model=model_name,
                    lora_config=LoRAConfig(rank=lora_rank),
                    type="training",
                )
            try:
                sync_task = await client.save_weights_and_get_sampling_client_async(
                    session_id, "post-training-eval"
                )
                sampler = await sync_task
                result = await client.sample(
                    session_id,
                    prompt,
                    SamplingParams(
                        max_tokens=1200, temperature=1.0, top_p=1.0, top_k=-1,
                        seed=42, stop_criteria=renderer.get_stop_sequences(),
                    ),
                    checkpoint_id=sampler.checkpoint_id,
                    num_samples=1,
                )
                if not result.sequences:
                    raise RuntimeError("No sampled sequences returned")
                message, parsed = renderer.parse_response(result.sequences[0].tokens)
                return message, parsed
            finally:
                await client.close_session(session_id)
```

No optimizer update occurs in this helper. It creates a `type="training"` session to obtain a sampler; that is not the same as deploying an inference endpoint. Close errors propagate rather than reporting successful cleanup. If session creation fails after the service allocated resources, inspect the project for the owned partial session before retrying. This helper uses identity intentionally; an API-key environment variable alone does not change its credential code.

### Print a real response from your saved state

After approving the new session and sampling operation, run this **after the
helper definition** in your private script. Replace the checkpoint placeholder
with the completed training `state_path`; do not invent a path from a failed
save or a sampler-only identifier. This example matches the own-text
walkthrough's model/renderer/rank and uses a new task prompt without supplying
the answer label:

```python
import asyncio
from interactive_training.renderers import get_text_content

message, parsed = asyncio.run(sample_saved_training_state(
    "session_<source>/final",
    model_name="Qwen/Qwen3.8-27B",
    renderer_name="qwen3_8_low_reasoning",
    lora_rank=32,
    messages=[
        {"role": "system", "content": "Classify the request as billing or technical. Reply with one label."},
        {"role": "user", "content": "My subscription shows two renewal charges."},
    ],
))
if not parsed:
    raise ValueError("Renderer could not parse the generated response")
print(get_text_content(message))  # Final text, excluding structured thinking parts.
```

In a notebook's running event loop, call `await sample_saved_training_state(...)`
instead of `asyncio.run`. Parsing proves only that the renderer accepted the
response format. For a quality claim, score a held-out set and a baseline with
the same messages/grader/sampling contract; one plausible label is not evidence
of training improvement. Keep responses private when the prompt is private,
and inspect the new session's cleanup result separately.

### Compare the same own-data prompt before and after

Passing `None` as the helper's checkpoint path creates a fresh baseline
session without loading saved training state. The other arguments, prompt,
renderer, rank, and sampling parameters remain identical. After explicit
approval for **two session allocations and their sampling**, this example
checks one unseen synthetic label task; it is a comparison smoke test, not a
quality benchmark or a statistical claim:

```python
async def compare_one_label(checkpoint_path):
    messages = [
        {"role": "system", "content": "Classify the request as billing or technical. Reply with one label."},
        {"role": "user", "content": "My subscription shows two renewal charges."},
    ]
    expected = "billing"  # Used by the grader only; not added to the prompt.
    results = {}
    for label, path in (("before", None), ("after", checkpoint_path)):
        message, parsed = await sample_saved_training_state(path, messages=messages)
        answer = get_text_content(message).strip().casefold()
        results[label] = {
            "parsed": parsed, "answer": answer,
            "valid_label": parsed and answer in {"billing", "technical"},
            "correct": parsed and answer == expected,
        }
    return results

# Only after approving the new sessions/samples; keep responses private.
comparison = asyncio.run(compare_one_label("session_<source>/final"))
print(comparison)
```

Run after the helper and `get_text_content` import above. The default model,
renderer, and rank match the worked own-text run; for another checkpoint,
pass its original settings to **both** calls. Each call attempts remote
cleanup before returning. A failed request aborts the comparison instead of
being counted as an incorrect model answer; inspect its cleanup before retrying.
Do not generalize one label into a percentage improvement. For a real study,
use a separately held-out, versioned set and an approved task-specific
evaluator that reuses compatible sessions, reports coverage/errors, and follows
the [comparison contract](#set-the-comparison-contract-first). The tiny helper
above creates a session per call and is not a recommended high-volume serving
or evaluation pattern.

## What this does not deploy

The SDK operations above provide **sampling within Fine-Tuning Sessions**. They do not document an adapter-download route, a GGUF/GPTQ conversion, a local Transformers/vLLM serving workflow, or publication to an OpenAI-compatible deployment. Those require a separately supported export/deployment path and its permissions; do not infer them from `state_path` or `sampler_path`.

### Application handoff boundary

| Intended use | What this cookbook supplies | What must be confirmed separately |
|---|---|---|
| Inspect a generated response | Compatible session restore, sampler sync, and SDK sampling above | Training/sampling approval, capacity, prompt privacy, and owned-session cleanup |
| Evaluate a task | Existing NER evaluation CLI, RL/SFT evaluator APIs, and the comparison contract | Task-specific generation/grading, complete coverage, and sufficient held-out evidence |
| Run application inference | No documented permanent inference deployment in these helpers | A supported serving provider/path, compatible model/adapter format, serving access, auth/API contract, capacity, price, and deletion plan |
| Export weights or run locally | No verified export/download/conversion recipe in the pinned public workflow | A supported export route, model/data/license rights, format compatibility, and separately validated conversion/runtime |

Interactive training access is not serving-preview approval. Ask the service or
serving owner for the supported handoff **before** promising an application
endpoint; do not call session sampling an OpenAI-compatible deployment. Keep
training, sampler, and deployed-resource identifiers distinct in the handoff.

For a reviewable result, retain the base/checkpoint identifiers, comparison config, dataset manifest, scored-example counts, error samples, quality delta, completion lengths, and timing. Keep historical results in [benchmarks](./benchmarks.md) labeled by their original model and setup; never use them as evidence for a new default.