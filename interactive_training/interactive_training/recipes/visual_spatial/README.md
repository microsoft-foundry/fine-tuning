# Visual-Spatial Fine-Tuning

This recipe fine-tunes a vision-language model on the `visual_spatial` subset
of [microsoft/Do-You-See-Me](https://huggingface.co/datasets/microsoft/Do-You-See-Me).
The primary entrypoint runs supervised fine-tuning (SFT) followed by
reinforcement fine-tuning (RFT). Objective-specific entrypoints are also
available. During RFT, each rollout contains an image and a question; the reward
is `1` when the model's normalized answer exactly matches the reference answer
and `0` otherwise.

The recipe has its own Azure entrypoint and does not depend on the math or
Tulu3 recipes.

## Prerequisites

- Install the cookbook with image support by following the
  [installation guide](../../../README.md#install).
- Authenticate with an API key (`AZURE_AI_API_KEY`) or Azure credentials. See
  [Authentication](../../../docs/auth.md).
- Select a vision-language model with image fine-tuning enabled for your project.
  See the [image-training requirements](../../../docs/supported_models.md#vision-fine-tuning).
- Ensure the environment can download `microsoft/Do-You-See-Me` from Hugging
  Face.

> [!NOTE]
> Image fine-tuning availability depends on the model, region, and service
> deployment. A vision renderer cannot add image support to a text-only model.

The SFT, RFT, staged training, and sampling entrypoints default to
`meta-models/Muse-Glimmer-30B`. Explicit model overrides remain supported, and
image fine-tuning must still be enabled for the selected model and project.
Set the model and the deployed sampler's actual context limit for the commands
below; the model default does not imply a particular service context limit:

```bash
export VISION_MODEL_NAME="meta-models/Muse-Glimmer-30B"
export MODEL_CONTEXT_LENGTH="<model-context-length>"
```

The recipe uses `model_name` as the tokenizer source and selects the model's
recommended renderer. Set `tokenizer_name` or `renderer_name` only when the
selected model requires an explicit override.


## Validate the dataset

Preflight renders every selected training and evaluation prompt and verifies
that the prompt plus generation budget fits the model context window. It does
not create a training session.

```bash
python -m interactive_training.recipes.visual_spatial.train_rft_azure \
    project_endpoint="<your-project-endpoint>" \
    model_name="$VISION_MODEL_NAME" \
    preflight_only=true \
    max_train_examples=100 max_test_examples=20 \
    model_context_length="$MODEL_CONTEXT_LENGTH" max_tokens=512
```

Run preflight whenever you change the model, renderer, image count, context
length, or generation budget. Preflight also warms the bounded prepared-prompt
cache, so later rollout groups and epochs can reuse image processing and prompt
tokenization while entries remain cached.

## Compare two images

The optional `absolute_difference` task asks the model to answer one counting
question for each of two images and return the absolute difference between the
counts as one integer. Pairs are deterministic and stay within their original
train or test split. This creates a harder compositional task without reducing
the completion budget or changing the model's response protocol.

```bash
python -m interactive_training.recipes.visual_spatial.train_rft_azure \
  project_endpoint="<your-project-endpoint>" \
  model_name="meta-models/Muse-Glimmer-30B" \
  renderer_name=muse_glimmer_low_reasoning \
  task_variant=absolute_difference images_per_example=2 \
  model_context_length="$MODEL_CONTEXT_LENGTH" \
  max_tokens=4096 eval_max_tokens=4096 \
  preflight_only=true
```

For a learning result, compare `test/visual_spatial_accuracy` before and after
training. Also require high `test/visual_spatial_parse_rate` and
`test/visual_spatial_single_integer_answer_rate`, plus near-zero
`test/visual_spatial_frac_at_max_tokens`, so an apparent change is not caused by
response-format failures or truncation. This task is available for the built-in
dataset and requires exactly two images per example.

## Learn a category taxonomy

The `count_category` task models adaptation to a customer-specific label
taxonomy. It keeps the original single-image counting questions but maps the
seven counts to a stable, one-digit category codebook. The prompt asks for the
learned category code without supplying that codebook; training examples and
rewards are the only source of the convention. Evaluation uses unseen images
from the held-out split with the same global codebook.

```bash
python -m interactive_training.recipes.visual_spatial.train_rft_azure \
  project_endpoint="<your-project-endpoint>" \
  model_name="meta-models/Muse-Glimmer-30B" \
  renderer_name=muse_glimmer_low_reasoning \
  task_variant=count_category images_per_example=1 \
  model_context_length="$MODEL_CONTEXT_LENGTH" \
  max_tokens=4096 eval_max_tokens=4096 \
  preflight_only=true
```

This variant demonstrates learning a new output taxonomy, not learning to count
from scratch. It remains a held-out generalization test: train and evaluation
images do not overlap, and the codebook is never inserted into evaluation
prompts. It is available only for the built-in dataset and requires one image
per example.

Exact-match RL cannot bootstrap this task when every rollout has zero reward.
Use the staged entrypoint to run supervised warmup, select its best durable
checkpoint by held-out accuracy, and continue with reinforcement fine-tuning:

```bash
python -m interactive_training.recipes.visual_spatial.train_azure \
  project_endpoint="<your-project-endpoint>" \
  model_name="meta-models/Muse-Glimmer-30B" \
  renderer_name=muse_glimmer_low_reasoning \
  model_context_length="$MODEL_CONTEXT_LENGTH" \
  behavior_if_log_dir_exists=raise
```

The defaults are a bounded, count-category profile: up to 60 SFT steps at a
linearly decayed `3e-5` learning rate, followed by up to 10 RFT steps at
`3e-6`, with 16 rollouts for one prompt from each category per RFT batch. SFT
and RFT have 20- and 25-minute wall-clock budgets, leaving service startup and
shutdown margin around a one-hour customer run. Hosted capacity and model-load
time vary, so the wall-clock target is not a latency SLA.

Use `preflight_only=true` to validate both stages without creating a session.
Override stage parameters with the `sft_` and `rft_` prefixes, for example
`sft_max_steps=40` or `rft_group_size=8`. The command writes separate `sft/`
and `rft/` artifacts under its log directory and a root
`staged_summary.json` containing the selected SFT checkpoint and RFT accuracy
comparison. It creates separate hosted sessions for the two objectives; the
single command coordinates their lifecycle and checkpoint handoff.
The root `run_meta.json` records both IDs under `azure.session_ids.sft` and
`azure.session_ids.rft` as they become available, including if a later stage
fails. Each stage also retains its own `run_meta.json`; preflight-only runs
leave the session IDs null because they do not create sessions.

The complete held-out split is evaluated throughout both stages. Do not use the
training split as the final metric. The SFT and RFT builders balance the seven
category codes deterministically and repeat training rows from rare categories
as needed; they never resample or alter the held-out split. Keep the
response-health checks above alongside accuracy.

The standalone `train_sft_azure` and `train_rft_azure` entrypoints remain
available when only one training objective is needed or stages must be
scheduled separately.

## Start training

```bash
python -m interactive_training.recipes.visual_spatial.train_rft_azure \
    project_endpoint="<your-project-endpoint>" \
    model_name="$VISION_MODEL_NAME" \
    lora_rank=32 \
    learning_rate=1e-5 temperature=1.0 \
    group_size=5 groups_per_batch=3 \
    max_tokens=512 eval_max_tokens=512 \
    max_train_examples=100 max_test_examples=20 \
    model_context_length="$MODEL_CONTEXT_LENGTH" \
    eval_every=5 save_every=5 \
    behavior_if_log_dir_exists=raise
```

Exact-match reward can be sparse. A compatible supervised checkpoint can
provide a better starting policy:

```text
load_checkpoint_path="<session-id>/<checkpoint-id>"
```

Use a new `log_path` when starting from an explicit checkpoint. If the selected
log directory already contains a checkpoint ledger, the recipe resumes that
run instead.

## Choose LoRA placement

Interactive Training always fine-tunes LoRA adapters rather than the base model parameters.
Language-model adapters are included in every training session. For models that
support multimodal LoRA placement, you can also add adapters to the vision tower
and the multimodal projector (sometimes called the connector):

| Configuration | Trainable LoRA adapters |
|---|---|
| `freeze_vision_tower=true freeze_multi_modal_projector=true` | Language model only (default) |
| `freeze_vision_tower=false freeze_multi_modal_projector=true` | Language model and vision tower |
| `freeze_vision_tower=true freeze_multi_modal_projector=false` | Language model and multimodal projector |
| `freeze_vision_tower=false freeze_multi_modal_projector=false` | Language model, vision tower, and multimodal projector |

For example, add adapters to all three components with:

```bash
python -m interactive_training.recipes.visual_spatial.train_rft_azure \
  project_endpoint="<your-project-endpoint>" \
  model_name="meta-models/Muse-Glimmer-30B" \
  model_context_length="$MODEL_CONTEXT_LENGTH" \
  lora_rank=32 \
  freeze_vision_tower=false \
  freeze_multi_modal_projector=false
```

These flags do not fully unfreeze the vision tower or projector's base weights;
`false` means that Interactive Training inserts trainable LoRA adapters into that component.
`lora_rank` applies to all inserted adapters. Additional adapter placement uses
more training memory and may need a different learning rate, so begin with the
frozen defaults and compare against a bounded run.

Currently, component placement outside the language model is supported only for
Muse Glimmer. Session creation rejects these flags for models that have not been
validated for multimodal LoRA. When resuming from `load_checkpoint_path`, use the
same LoRA rank, vision-tower setting, and projector setting as the saved
checkpoint. The service also verifies the recipe's LoRA alpha matches.

## Muse Glimmer reasoning strength

Muse Glimmer supports four customer-facing reasoning-strength renderers. The
default `muse_glimmer` renderer uses high reasoning strength:

| Renderer | Reasoning strength |
|---|---|
| `muse_glimmer_low_reasoning` | Low |
| `muse_glimmer_medium_reasoning` | Medium |
| `muse_glimmer` | High (default) |
| `muse_glimmer_xhigh_reasoning` | Extra high |

Select a lower reasoning strength when a bounded completion budget is more
important:

```bash
# Sampling: high reasoning (the Muse default)
python -m interactive_training.recipes.visual_spatial.sample_azure \
  --project-endpoint "<your-project-endpoint>" \
  --model-name "meta-models/Muse-Glimmer-30B" \
  --renderer-name muse_glimmer \
  --max-tokens 2048

# Training: low reasoning
python -m interactive_training.recipes.visual_spatial.train_rft_azure \
  project_endpoint="<your-project-endpoint>" \
  model_name="meta-models/Muse-Glimmer-30B" \
  renderer_name=muse_glimmer_low_reasoning \
  model_context_length="$MODEL_CONTEXT_LENGTH" \
  max_tokens=4096 eval_max_tokens=4096
```

Higher reasoning strengths can spend substantially more completion tokens before
producing a visible answer. Muse's official chat template does not provide a
disable-thinking option; use `muse_glimmer_low_reasoning` for the smallest
supported reasoning budget. Low reasoning still emits a private reasoning turn,
so short-answer tasks need enough tokens for both that reasoning and the visible
answer. Do not interpret an accuracy increase as learning when responses are
being truncated: require a near-zero `test/visual_spatial_frac_at_max_tokens`
and a high `test/visual_spatial_parse_rate`. Set `model_context_length` to the
deployed sampler's actual limit and leave room for both the rendered image prompt
and `max_tokens`; preflight checks that combined budget.

## How recipe images are encoded

The executable recipe downloads `microsoft/Do-You-See-Me` directly from Hugging
Face and selects rows with `dataset_id="visual_spatial"`. Each row has
`question`, `answer`, `sweep`, `dataset_id`, and `image` columns. The Hugging
Face `image` feature uses `decode=True`, so accessing a row returns a Pillow
image object rather than a URL or base64 data URI.

The recipe constructs this in-memory message:

```python
{
  "role": "user",
  "content": [
    {"type": "image", "image": row["image"]},
    {"type": "text", "text": row["question"]},
  ],
}
```

The vision renderer prepares the Pillow image for sampling and training and
includes its visual-token count in context-length preflight. You do not upload
the `Do-You-See-Me` dataset to Interactive Training first.

## Key parameters

| Parameter | Purpose |
|---|---|
| `model_name` | Identifier of a supported vision-language model. Defaults to `meta-models/Muse-Glimmer-30B`. |
| `tokenizer_name` | Optional tokenizer override. Defaults to `model_name`. |
| `renderer_name` | Optional renderer override. Defaults to the model registry recommendation. |
| `data_path` | Optional local Azure OpenAI conversations JSONL file. Uses `microsoft/Do-You-See-Me` when omitted. The source path is not recorded in run metadata. |
| `task_variant` | `single_image` (default), `absolute_difference` for two-image composition, or `count_category` for learning a stable one-digit label taxonomy. |
| `images_per_example` | Number of ordered images in each prompt. `absolute_difference` requires `2`, `count_category` requires `1`, and otherwise the question refers to the first image. |
| `group_size` | Rollouts sampled for each prompt. Larger groups can improve reward diversity at higher sampling cost. |
| `groups_per_batch` | Prompt groups in each optimizer batch. |
| `num_epochs` | Number of passes over the selected training examples. Increase this to overfit a bounded dataset. |
| `max_tokens` | Maximum completion length for training rollouts. |
| `eval_max_tokens` | Maximum completion length for deterministic evaluation. |
| `model_context_length` | Combined prompt and completion limit checked during preflight. |
| `prompt_cache_max_mb` | Maximum prepared-prompt payload retained in memory for reuse across rollout groups, epochs, and evaluations. Defaults to 128 MiB; set to `0` to disable cross-batch reuse. |
| `eval_concurrency` | Maximum simultaneous evaluation requests and prompt-preparation workers. Defaults to 6. |
| `log_examples` | Includes raw questions, reference answers, and predictions in local logs when `true`. Defaults to `false`; use only in an access-controlled environment. |
| `freeze_vision_tower` | Omits LoRA adapters from the vision tower when `true` (default). |
| `freeze_multi_modal_projector` | Omits LoRA adapters from the multimodal projector when `true` (default). |
| `load_checkpoint_path` | Starts from a compatible Interactive Training checkpoint. |
| `max_steps_off_policy` | Enables asynchronous RL and limits accepted policy staleness. Omit it for synchronous RL. |
| `preflight_only` | Validates all selected prompts without creating a session. |

Use `max_train_examples`, `max_test_examples`, and `max_steps` for a bounded
smoke test before scaling up a run.

## Adapt the recipe to your data

The recipe includes an
[Azure OpenAI conversations JSONL adapter](azure_openai_jsonl.py). The
`data_path` parameter is optional: omit it to use the built-in
`microsoft/Do-You-See-Me` dataset, or set it to a local JSONL file to use your
own data without changing Python code. Each line keeps the familiar `messages`
array. The image may be a public HTTPS URL:

```json
{"messages":[{"role":"user","content":[{"type":"text","text":"What defect is visible?"},{"type":"image_url","image_url":{"url":"https://example.com/part.png"}}]},{"role":"assistant","content":"crack"}]}
```

or a base64 data URI:

```json
{"messages":[{"role":"user","content":[{"type":"text","text":"What defect is visible?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."}}]},{"role":"assistant","content":"crack"}]}
```

Run preflight against your file before starting a session:

```bash
python -m interactive_training.recipes.visual_spatial.train_rft_azure \
  project_endpoint="<your-project-endpoint>" \
  model_name="$VISION_MODEL_NAME" \
  model_context_length="$MODEL_CONTEXT_LENGTH" \
  data_path="data/my_images.jsonl" \
  preflight_only=true
```

Remove `preflight_only=true` and add the training parameters from
[Start training](#start-training) when preflight passes. To sample one example
from the same file:

```bash
python -m interactive_training.recipes.visual_spatial.sample_azure \
  --project-endpoint "<your-project-endpoint>" \
  --model-name "$VISION_MODEL_NAME" \
  --data-path "data/my_images.jsonl"
```

The adapter preserves every message before the final assistant message as the
rollout prompt, including system instructions and earlier turns. It uses the
final assistant message as the reference answer for normalized exact-match
reward. A row may contain multiple `image_url` parts, but keep
`images_per_example=1`; the built-in higher values group separate
`Do-You-See-Me` rows with decoded `image` fields.

For images returned during live tool execution, including MCP `ImageContent`,
embedded resources, and resource links, see
[Image tool results](../mcp_image_tool/README.md#image-tool-results).

### Untrusted image limits

Treat every JSONL image reference as untrusted input. Remote references must use
HTTPS and resolve only to public unicast addresses. The loader rejects HTTP,
embedded credentials, non-default ports, redirects, private or loopback
destinations, DNS changes during a request, and IPv4 destinations embedded in
NAT64 addresses. Production egress policy should independently block access to
private networks and cloud metadata endpoints.

The adapter accepts JPEG, PNG, and WebP images and enforces these limits before
a session is created:

| Resource | Limit |
|---|---:|
| JSONL file | 512,000,000 bytes and 50,000 examples |
| JSONL row | 64,000,000 bytes |
| Messages per example | 128 |
| Text per example | 1,000,000 characters |
| Images per example | 64 |
| Encoded image | 10,000,000 bytes |
| Decoded image | 25,000,000 pixels |
| Aggregate encoded images per example | 32,000,000 bytes |
| Aggregate decoded images per example | 100,000,000 pixels |

Remote fetches have a 10-second deadline per attempt and retry transient
failures up to three times. Redirects are not followed. Error messages omit URL
paths, queries, fragments, and embedded credentials so signed URLs are not
written to logs.

For JSONL runs, validated image bytes are stored under `image_cache/` in the run
directory. Entries are keyed by the SHA-256 hash of the original reference,
written atomically, and readable only by the current user. Resumed runs use the
cached bytes, so a changed or unavailable remote object cannot silently alter a
previously admitted example. Keep the run directory on trusted local storage;
deleting `image_cache/` requires the remote images to pass admission again.

For a different conversation schema, change `adapt_conversation` in the
adapter. For structured output, multiple valid answers, or a model-based
reward, replace its `answers_match` function. The rest of the recipe continues
to own renderer construction, prompt-budget preflight, batching, training, and
evaluation.

## Outputs and metrics

The run directory contains:

- `run_meta.json`, with the resolved model, dataset, training, and endpoint host
  configuration. It excludes the source dataset path and command line.
- `checkpoints.jsonl`, with checkpoints available for resume or downstream use.
- `visual_spatial_eval_<n>.jsonl`, with an opaque example ID and grading results
  for each evaluation. Set `log_examples=true` only when raw questions,
  reference answers, and predictions are required and the run directory is
  appropriately access controlled.
- `visual_spatial_accuracy_summary.json`, with the exact-match accuracy before
  training, after training, and the absolute and percentage-point improvement.
  The recipe also prints this comparison when training completes. For a run
  started with `load_checkpoint_path`, "before training" measures that loaded
  checkpoint. Resumed runs preserve the first evaluation already recorded in
  `metrics.jsonl` as their baseline.

Monitor these evaluation metrics:

- `test/visual_spatial_accuracy`: exact-match accuracy.
- `test/visual_spatial_parse_rate`: fraction of responses parsed successfully.
- `test/visual_spatial_frac_at_max_tokens`: fraction of responses that reached
  the evaluation token limit.
- `test/visual_spatial_prompt_cache_hits`, `_misses`, and `_evictions`: prepared
  prompt cache behavior for evaluation.
- `test/visual_spatial_prompt_cache_bytes`: current evaluation prompt-cache
  payload size.

A low parse rate or a high fraction at the token limit usually means the
renderer or generation budget needs adjustment. If all reward groups are
constant, increase `group_size`, verify answer formatting, or start from a
task-compatible checkpoint before running a long job.

With evaluation enabled (`eval_every > 0`), a fresh run evaluates before the
first optimizer step and again after the final checkpoint. The completion
summary makes the comparison explicit. Setting `eval_every=0` disables both
evaluations, so no comparison artifact is produced.

```text
Visual-spatial accuracy comparison:
  Before training: 20.0% (4/20)
  After training:  65.0% (13/20)
  Improvement:     +45.0 percentage points
  Summary:         <log_path>/visual_spatial_accuracy_summary.json
```
