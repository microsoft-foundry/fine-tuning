# MCP alien-color tool-use SFT + RFT

This recipe trains a vision-language model to memorize an alien vocabulary,
then verify its recalled mappings with an MCP image tool before identifying the
alien words for one, two, or three requested colors.
It uses the official MCP Python SDK over stdio or streamable HTTP and Interactive Training's
standard multi-turn tool environment.

The bundled server is local, deterministic, and network-free. The
`reveal_alien_color` tool accepts one of ten invented words and returns a short
text block followed by a PNG color swatch. The word-to-color mapping is hidden
from evaluation prompts and learned from training demonstrations.

## Setup

Requires Python 3.11+, an Azure AI Foundry project with access to a
vision-language training model, and its tokenizer. Confirm that image fine-tuning
is enabled for your model and project before starting a paid training session;
see the [image-training requirements](../../../docs/supported_models.md#vision-fine-tuning).
A text-only model cannot run this example. No local GPU is required when using
the hosted service.

Complete the [cookbook installation](../../../README.md#install), including
the PyPI SDK, then install the MCP and image dependencies from `interactive_training/`:

```bash
pip install -e '.[mcp]' --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
```

Authenticate with Azure CLI, or set `AZURE_AI_API_KEY` for key-based auth.
See [authentication](../../../docs/auth.md) for the project endpoint and credentials.
After updating the cookbook, follow the
[SDK update instructions](../../../README.md#updating) to keep the installed PyPI release aligned with the cookbook.

## Verify MCP locally

Run the recipe smoke before creating a training session:

```bash
python -m interactive_training.recipes.mcp_image_tool.train_azure \
  mcp_smoke_only=true
```

This does not contact Azure. It starts the bundled server as a child process,
initializes an official `ClientSession`, discovers its tools, invokes
`reveal_alien_color`, and checks that Interactive Training normalized the returned image. For a
server-only protocol smoke, run
`python -m interactive_training.recipes.mcp_image_tool.server --smoke`.

## Train

```bash
python -m interactive_training.recipes.mcp_image_tool.train_azure \
  project_endpoint="<your-project-endpoint>" \
  model_name="meta-models/Muse-Glimmer-30B" \
  renderer_name=muse_glimmer_low_reasoning \
  log_path=./runs/mcp-image-tool-seed0 \
  behavior_if_log_dir_exists=raise
```

Use a new log directory for each fresh run. Keep the seed and evaluation tasks
fixed when measuring learning; do not select the best checkpoint or seed after
seeing evaluation results. The default run pins LoRA initialization with
`lora_seed=552161550` and includes 12 supervised updates,
up to six RL updates with no default time limit, and one evaluation pass per
stage.

The MCP process and session remain open through training and evaluation. Full
credit requires exactly one successful call per requested color and the correct
verified answer. Exploration and recovery earn no reward unless the resulting
trajectory still meets that exact contract. Each task allows one tool call per
requested color plus one exploration call, capped by `max_tool_calls`, followed
by one final answer turn and subject to the trajectory token budget. Using the
exploration call makes strict success impossible for that trajectory, but gives
RL an explicit zero-reward contrast with exact trajectories. The default SFT curriculum contains 20
balanced single-color demonstrations followed by ten two-color and ten
three-color composition demonstrations.
Evaluation uses a held-out prompt template with balanced one-, two-, and
three-color requests. Multi-color answers use an explicit mapping such as
`Answer: green=irevax, blue=nalmek`.

For example, a successful two-color trajectory is:

```text
User: Identify the alien words for green and blue, verifying each with the tool.
Assistant tool call: reveal_alien_color(alien_word="irevax")
Tool: text plus a PNG showing green
Assistant tool call: reveal_alien_color(alien_word="nalmek")
Tool: text plus a PNG showing blue
Assistant: Answer: green=irevax, blue=nalmek
Reward: 1.0
```

This is an illustrative transcript, not an evaluation prompt or measured output.
Looking up a third word or omitting either verification fails exact success,
even if the final answer is correct.

Before RL, the recipe evaluates the fresh adapter, then performs
three supervised passes over 40 training tasks at
`warmup_learning_rate=2e-4` (12 updates at the default batch size). The short
warm-up is intended to leave room for RL; it does not guarantee reward variation.
Demonstrations
execute the MCP tool once per requested color and must pass the exact reward
grader. Only assistant tokens receive supervised loss; prompts, tool results,
and images are masked. The held-out prompt template is never demonstrated.
This is deliberately SFT followed by RL, not evidence of learning solely from
sparse RL rewards. It is the only supported training flow. To continue from
trained weights, set `load_checkpoint_path="model_<id>/<name>"`
in a new log directory. Loading a checkpoint automatically skips SFT and the
fresh-model evaluation; the loaded weights become the before-RL baseline.
This starts a new bounded RL stage, not an exact replay of an interrupted run.
Resumed runs cannot establish a fresh-model baseline or meet the fresh-run
acceptance target.

An exact, efficiently verified answer earns `1.0`; every other trajectory earns
`0.0`. This binary reward is identical to the strict `correct` metric, so RL
cannot trade exact accuracy for partially correct or inefficient behavior.
Proposed calls blocked by the execution budget still prevent exact success.
Wrong, duplicate, exploratory, and post-discovery calls also prevent exact
success. Fabricating a call or skipping tool verification earns no credit. The
`correct`, `verified_answer`, `verified_fraction`, `efficiency`, `wrong_calls`,
and `post_discovery_calls` metrics retain detailed diagnostics.

Every rollout within a GRPO group receives the same deterministically shuffled
tool schema, so centered rewards compare policies under the same context. The
schema can vary across tasks and seeds without confounding members of one group.
The default group size is eight. The default loss is `importance_sampling`
with group-mean-centered advantages, without standard-deviation normalization
or PPO clipping. Set `loss_fn` and `loss_fn_config` to configure the RL loss,
and `num_substeps` to split each RL training batch into optimizer substeps.
`compute_post_kl=true` enables post-update KL diagnostics, not a KL-reference
penalty. The vision tower and multimodal projector default to frozen; use
`freeze_vision_tower` and `freeze_multi_modal_projector` to change that.
The reward formula remains fixed. Set `max_tool_calls` to a positive integer
to change the RL and evaluation call budget, for example `max_tool_calls=10`.
Budgets below the number of requested colors make full verification impossible;
larger budgets may also require increasing `max_trajectory_tokens`. SFT
demonstrations still use exactly one call per requested color.
Format-only variation and parse failures earn
zero reward. Training metrics retain all sampled groups, including groups with
constant rewards.

The default RL stage uses a separate seeded set of 10 single-color, 25
ordered-pair, and 25 ordered-triple tasks. The first three 10-task batches
interleave single, pair, and triple tasks; the final three alternate pair and
triple tasks. Single-color tasks are capped at one per color, and any remaining
task slots are split between pairs and triples. SFT uses 40 tasks, and evaluation
uses 30 held-out prompts. Before each RL update, the recipe records all rollout
rewards. If no group has varying rewards, it skips that update and continues to
the next batch within the fixed budget:
both all-correct and all-failed groups have zero GRPO advantage. This is not a
claim that all tasks are solved. The default budget is six attempted batches
with no wall-clock limit, yielding at most six RL updates. Skipped batches consume that budget;
`max_steps` can lower it, and `mixed_train_examples` controls the available tasks.
`rl_skipped_batches` records how many batches had no training signal. Reward
differences between tasks do not create GRPO advantages: variation must occur
among rollouts of the same task. A longer bounded experiment can explicitly set
`mixed_train_examples=180 max_steps=18`; this does not guarantee 18 updates.
The final checkpoint is saved even when no RL update is possible.

Evaluation runs greedily at temperature zero at stage boundaries, while RL
rollouts retain the configured sampling temperature. Evaluation passes use
the configured seed, incremented for each repetition, so the same checkpoint
and configuration can be compared reproducibly. Each completed RL update saves
`rl_step_<n>` and records it in `rl_checkpoints.jsonl`; the run also saves a
`final` checkpoint. Logging is local JSON/JSONL. There are no curriculum,
RL-only, async, KL-reference penalty, W&B, or custom evaluation/checkpoint-cadence modes.

By default, fresh-model, post-SFT, and final evaluations each run the same
30 tasks once. Incomplete passes abort the comparison. With zero RL updates,
the unchanged post-SFT evaluation is reused and RL gain is exactly zero.

`max_wall_clock_seconds` defaults to `None` (no time limit). An explicit value,
such as `max_wall_clock_seconds=900`, applies only to the RL stage. It is checked between
batches and bounds rollout collection; an expired rollout budget proceeds to
final checkpoint saving and evaluation. In-flight optimizer updates and final
evaluation are allowed to finish. An external timeout must leave room for
warm-up, checkpoint saving, and evaluation.

Training requests are split at the API's 25-million decoded-pixel limit while
retaining one optimizer step per batch. Encoded PNG byte size alone does not
bound decoded image size. Split requests report their count instead of
potentially misleading partial forward/backward metrics.

The CLI retains model/session and MCP connection settings, data sizes, batch
size and concurrency, learning rates, sampling/token budgets, evaluation
repetitions, seed, RL batch/time limits, checkpoint loading, and local logging.
Run the entrypoint with `--help` for all supported options.

## Check the results

Open `learning_report.json` in the run directory after the session completes:

| Field | Interpretation |
|---|---|
| `initial_accuracy` | First evaluated checkpoint's exact success rate; pre-SFT for fresh runs, loaded checkpoint for resumed runs. |
| `post_sft_accuracy` | Exact success rate before RL: after SFT, or from loaded weights. |
| `final_accuracy` | Exact success rate at the end of the run. |
| `rl_accuracy_gain` | Final minus before-RL accuracy; isolates the measured RL change. |
| `before_rl_tool_calls`, `after_rl_tool_calls` | Average calls before and after RL, including exploration. |
| `before_rl_verified_answer`, `after_rl_verified_answer` | Correct image-verified answer rates before and after RL. |
| `rl_efficiency_improved` | Observed fewer calls with no lower verified-answer rate after at least one RL update; not a significance test. |
| `by_color_count` | Initial and final accuracy for one, two, and three colors. |
| `accepted` | Whether this run met the stated learning target with complete evaluations. |
| `final_checkpoint` | Checkpoint URI for the trained adapter. |

Accuracies are fractions, so `0.3` means 30%. Process success does not imply
`accepted=true`. Acceptance requires fresh-run initial accuracy below 10%,
final accuracy above 30%, complete evaluations, and a strict-accuracy
improvement from SFT to RL. Compare `before_rl` with `after_rl`; improvement over
`before_sft` alone is not evidence of RL learning.

The run directory retains:

- `baseline_metrics.json` and `warmup_metrics.jsonl`: fresh baseline and SFT updates.
- `evaluation_passes.jsonl` and `evaluations.jsonl`: individual passes and stage means.
- `protocol.jsonl` and `probe.jsonl`: seeded tasks and unfiltered per-group rewards.
- `rollouts.jsonl`: calls, final answers, schema order, and rewards, excluding image payloads.
- `updates.jsonl`, `rl_checkpoints.jsonl`, and `rl_report.jsonl`: updates, checkpoints, and stopping reason.

Retain failed runs and incomplete evaluations; do not count them as successful
learning demonstrations or replace them silently with another seed.
Repeated passes are not independent tasks or training runs. Evaluation holds
out wording, not the mapping or all color compositions. Inspect the artifacts
from your own run using the metrics described above; this repository does not
bundle a separate measured-results report for this recipe.

## Small service check

After the local MCP smoke, this exercises the same SFT-to-RL flow with tiny
datasets. It can still incur service charges and is not a learning benchmark:

```bash
python -m interactive_training.recipes.mcp_image_tool.train_azure \
  project_endpoint="<your-project-endpoint>" \
  log_path=./runs/mcp-image-tool-service-check \
  behavior_if_log_dir_exists=raise \
  max_steps=1 mixed_train_examples=2 \
  max_train_examples=4 max_test_examples=2 \
  group_size=4 groups_per_batch=2 evaluation_repetitions=1
```

The signal gate may skip the optimizer update if all rewards in every group
are identical. Use the full training command above for the learning comparison.

## Use your MCP server

### Stdio

Set an executable and its arguments without invoking a shell. Add
`mcp_smoke_only=true` first to verify the connection without starting training:

```bash
python -m interactive_training.recipes.mcp_image_tool.train_azure \
  project_endpoint="<your-project-endpoint>" \
  mcp_server_command=python \
  mcp_server_args='-m your_package.image_tool_server'
```

### Streamable HTTP

For a remote server, pass its HTTPS MCP endpoint. Put bearer credentials in an
environment variable rather than a command-line argument:

```bash
export MCP_IMAGE_TOOL_TOKEN='<your-token>'
python -m interactive_training.recipes.mcp_image_tool.train_azure \
  mcp_transport=streamable_http \
  mcp_server_url='https://mcp.example.com/mcp' \
  mcp_bearer_token_env=MCP_IMAGE_TOOL_TOKEN \
  mcp_smoke_only=true
```

After the smoke passes, provide the real `project_endpoint` and remove
`mcp_smoke_only=true` to train. The client keeps the HTTP connection and MCP
session open through every rollout and evaluation. Remote endpoints must use
HTTPS; plain HTTP is accepted only for loopback development servers.

For this task unchanged, the server must expose
`reveal_alien_color(alien_word: string)` and return an MCP image containing the
word's color. To adapt the recipe to a different MCP tool, update the task
records and reward in `data.py`, then adjust `REQUIRED_TOOL` and the prompt in
`mcp_env.py`. No renderer-specific conversion is needed:
`tools_from_mcp_session()` maps discovered MCP tools to Interactive Training's normal tool
protocol.

## Image tool results

For a vision-capable model, discover MCP tools once and keep the MCP session
open while the standard agent environment runs:

```python
from interactive_training.tool_use import build_agent_tool_env, tools_from_mcp_session

tools = await tools_from_mcp_session(mcp_session)
env = build_agent_tool_env(
  renderer=renderer,
  tools=tools,
  initial_messages=initial_messages,
  reward_fn=reward_fn,
)
```

Interactive Training normalizes MCP `TextContent`, `ImageContent`, textual and image
`EmbeddedResource` values, and image `ResourceLink` values, preserving their
order. Ordinary tools can return the same canonical content through `ToolResult`:

```python
from interactive_training.tool_use import ToolResult

result = ToolResult(messages=[{
  "role": "tool",
  "content": [
    {"type": "text", "text": "Current screen:"},
    {"type": "image", "image": "data:image/png;base64,<base64-data>"},
  ],
}])
```

Use `image_url` with a public HTTPS URL for a remote image reference. Interactive Training accepts
JPEG, PNG, and WebP, enforces byte and pixel limits, validates signatures and MIME
types, rejects malformed or animated images, and restricts remote images to
public HTTPS addresses. It does not upload images to an artifact store.
For a later tool accepting `uri`, forward the normalized data URI or HTTPS URL;
Interactive Training does not infer subsequent tool arguments. For dataset images, see the
[visual-spatial recipe](../visual_spatial/README.md#adapt-the-recipe-to-your-data).

## Files

- `server.py`: official low-level MCP color oracle and local protocol smoke.
- `data.py`: deterministic alien vocabulary and train/test prompt variants.
- `mcp_env.py`: recall-then-verify MCP environment and exact-call reward.
- `warmup.py`: training-only demonstrations executed through MCP and supervised updates.
- `mixed_rl.py`: bounded mixed-color RL, reward-variation gate, and repeated evaluations.
- `learning_report.py`: measured initial/final exact-accuracy acceptance.
- `training_client.py`: pixel-bounded training submissions with gradient accumulation.
- `train_azure.py`: Azure session creation, MCP lifetime, and RFT entrypoint.