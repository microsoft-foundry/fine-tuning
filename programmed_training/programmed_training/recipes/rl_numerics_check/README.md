# Long-context RL numerics check

Train a document-counting policy through Azure Fine-Tuning Sessions to exercise
multi-turn sampling, trainer evaluation of sampled tokens, GRPO updates and
recoverable checkpoints. This is a numerical-consistency workload, not a
general model-quality benchmark.

The task is adapted from the
[Tinker numerics recipe](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/rl_numerics_check).
This implementation uses the cookbook's existing tool environment, renderers,
Azure SDK and shared RL loop; it does not deploy or reconfigure engines.

Run this recipe through the cookbook CLI. No AML command-job preset has been
validated; it is explicitly exempted from command-job preset coverage until one
is qualified.
The CLI inherits the shared [Adam optimizer settings](../../../README.md#adam-optimizer-defaults)
and forwards all four `adam_*` overrides to the RL training configuration.

## Setup

Follow the cookbook [installation](../../../README.md#install) and
[authentication](../../../docs/auth.md) instructions. Use an authorized Foundry
project with fine-tuning admission and compatible sampling/training capacity.
Authenticate with `az login` or set `AZURE_AI_API_KEY`; successful authentication
alone does not enroll a subscription or qualify a model route.

The default corpus is the train split of
[WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext).
Up to 100 million characters are loaded on the client and cached in process.
No corpus is bundled; follow its CC BY-SA attribution and redistribution terms.
Alternatively, pass `corpus_path=<utf8-file>` with appropriately licensed text.
There is no retrieval server, sandbox, RLE deployment or extra recipe-specific
dependency to install.

## Model and rendering

The CLI defaults to the smaller thinking experiment on the experimental
`Qwen/Qwen3.8-27B-256K` identity, using the canonical `Qwen/Qwen3.8-27B`
tokenizer and **`qwen3_8`** renderer. This native renderer preserves historical
reasoning and uses `xhigh` effort. Unlike the other text recipes, this default
requires qualified 256K sampling and training capacity; it is not a small wiring
check. Read the capacity gate below before running.

Explicit `tokenizer_name` and `renderer_name` overrides take precedence.
For another `model_name`, tokenizer selection follows that model and automatic
rendering retains the model-aware non-thinking numerical baseline. No model
fallback or engine reconfiguration is performed.

`max_tokens=8192` caps each assistant generation, including reasoning and
visible output. `max_trajectory_tokens` bounds the complete episode, including
prompts, tool results and retained reasoning. The environment reserves the next
generation's budget before another turn. Neither setting expands a deployment's
context capacity or guarantees a complete rollout.

## Small wiring check

From the cookbook root, run in PowerShell:

```powershell
python -m programmed_training.recipes.rl_numerics_check.train_azure `
  project_endpoint="https://<resource>.services.ai.azure.com/api/projects/<project>" `
  model_name="Qwen/Qwen3.8-27B" renderer_name=qwen3_8 `
  task.num_pages_min=1 task.num_pages_max=1 task.page_chars=256 task.num_pages_hint=exact `
  group_size=2 groups_per_batch=1 max_train_examples=2 max_steps=2 `
  max_tokens=8192 max_trajectory_tokens=32768 `
  eval_every=0 save_every=1 max_wall_clock_seconds=1800 `
  behavior_if_log_dir_exists=raise log_path="<fresh-smoke-directory>"
```

Two updates exercise sampling from updated weights. Perfect reward on this
tiny task is only a wiring result, not evidence of learning or long-context
quality. Parse errors and context overflows must be reported, not counted as
useful reward variation. Wall-clock limits are checked at training-loop
boundaries; they are not hard deadlines for an in-flight sampling request.

### Training tier

The optional `training_type` follows the shared
[TrainingType contract](../../../docs/training.md#training-type):
`GlobalStandard`, case-insensitively. `DatazoneStandard` and `DeveloperTier`
remain unavailable in this preview.
Omitted `None` is not sent to the service. The tier applies to the training
session and any teacher/KL-reference sessions created by the shared loop.

## Task and reward

Each problem presents a word-counting task over consecutive, word-aligned
pages. The policy calls `read_page(page)` using integer page numbers starting
at 1. Only the first tool call per response executes; extra calls receive
explicit errors. Finish without a tool call and end with `Answer: <count>`.

Illustrative two-page trajectory, using synthetic text:

```text
User: Count "time" across pages 1 and 2.
Assistant tool call: read_page(page=1)
Tool: [Page 1] Time time's times lifetime
Assistant tool call: read_page(page=2)
Tool: [Page 2] time-based TIME
Assistant: Answer: 4
```

Matching ignores case and treats each maximal run of ASCII letters as a word.
The last `Answer:` integer is parsed. Missing answers score zero; otherwise:

```text
reward = max(0, 1 - abs(predicted - true) / (reward_tolerance * max(true, 1)))
```

`task.reward_tolerance` defaults to 0.2. Parse failures and context-budget
failures score -0.1. The default task has a fixed hidden count of 8 pages,
a public range of 7-9, and approximately 108,800 characters per page, with at
most `num_pages_max + 4` assistant turns. When `secret_num_pages` is omitted,
the existing deterministic draw selects 8 for this range; the documented
command pins it explicitly. Characters are not tokens; measure actual rendered
sequences.

### Difficulty controls

| Option | Behavior |
|---|---|
| `task.num_pages_hint=exact` | Reveals the independently sampled page count for each problem. |
| `task.num_pages_hint=range` | Reveals only the range; every problem shares a hidden page count. Set it with `task.secret_num_pages`, or use the deterministic draw. |
| `task.word_hint=exact` | Reveals the target word sampled from `task.words`. |
| `task.word_hint=first_last` | Reveals only first/last letters. Every configured word must have a unique letter pair. |
| `task.word_hint=hidden` | Reveals no word identity; requires one configured word. |

Use `task.words=until` or `task.words=until,after,during` to reduce the default
28-word vocabulary. With a hidden page count, pages past the true end still
return text, excluded from the gold count; there is no end-of-document signal.
Hidden constants stay fixed across problems, so learning them is not proof of
transfer. Hiding information can delay saturation but does not guarantee
difficulty. Keep task settings, seeds and reward tolerance fixed within a run.

## Default smaller thinking experiment

**Capacity gate:** `Qwen/Qwen3.8-27B-256K` is a separate experimental API identity
with 262144-token trainer and sampler profiles. It reuses the canonical
`Qwen/Qwen3.8-27B` weights/tokenizer. Source registration or session admission
does not prove sampling dispatch works for a particular project/subscription.
Verify both routes with a successful sample and trainer operation before
launching this workload. Do not silently fall back to the ordinary 32K identity.

The default workload uses **one group of four rollouts per update and ten
updates** (indices 0-9), with 8 hidden pages at 108,800 characters/page.
`max_train_examples=10` supplies ten groups, not ten individual rollouts.
Evaluation is disabled and a training checkpoint is saved after every update.
The 259200-second wall-clock budget is checked at loop boundaries, not a hard
deadline for an in-flight call.

```powershell
python -m programmed_training.recipes.rl_numerics_check.train_azure `
  project_endpoint="https://<qualified-resource>.services.ai.azure.com/api/projects/<project>" `
  model_name="Qwen/Qwen3.8-27B-256K" tokenizer_name="Qwen/Qwen3.8-27B" `
  renderer_name=qwen3_8 `
  task.num_pages_min=7 task.num_pages_max=9 task.page_chars=108800 `
  task.num_pages_hint=range task.secret_num_pages=8 task.word_hint=exact `
  seed=0 group_size=4 groups_per_batch=1 max_train_examples=10 max_steps=10 `
  max_tokens=8192 max_trajectory_tokens=262144 `
  eval_every=0 save_every=1 max_wall_clock_seconds=259200 `
  auto_resume=True auto_resume_max_retries=3 behavior_if_log_dir_exists=raise `
  log_path="<fresh-small-run-directory>"
```

This reduces planned rollouts per update from 64 to 4 and page reads per
complete traversal from 32 to 8. The nominal document size is unchanged:
`8 * 108800 = 32 * 27200 = 870400` characters. Rendered context depends on the
sampled text, actual pages read and retained thinking; fewer tool turns do not
establish a wall-clock speedup or a successful long-context task.

The third-update milestone means completion of indices 0, 1 and 2. Report
elapsed time, per-update durations, error rates, actual token depth and saved
checkpoints while continuing to the ten-update limit. Actual near-256K
qualification requires measured sequences, a successful update, checkpoint
save and checkpoint reload. A completed update with all parse errors and zero
gradient is execution evidence, not useful learning.

### Original bulk settings (historical, opt-in)

The previous bulk stress experiment used 8 groups of 8 rollouts per update,
32 hidden pages with a public range of 28-36, 27,200 characters/page and
26 updates. These settings are retained for comparison, not as the default or
a claim of a successful completed run. The earlier guide used 10,000
characters/page; the later bulk attempts used 27,200.

```powershell
python -m programmed_training.recipes.rl_numerics_check.train_azure `
  project_endpoint="https://<qualified-resource>.services.ai.azure.com/api/projects/<project>" `
  model_name="Qwen/Qwen3.8-27B-256K" tokenizer_name="Qwen/Qwen3.8-27B" `
  renderer_name=qwen3_8 `
  task.num_pages_min=28 task.num_pages_max=36 task.page_chars=27200 `
  task.num_pages_hint=range task.secret_num_pages=32 task.word_hint=exact `
  seed=0 group_size=8 groups_per_batch=8 max_train_examples=208 max_steps=26 `
  max_tokens=8192 max_trajectory_tokens=262144 `
  eval_every=0 save_every=3 max_wall_clock_seconds=259200 `
  behavior_if_log_dir_exists=raise log_path="<fresh-bulk-run-directory>"
```

### Token-calibrated 240K document context

The automated Qwen3.8-256K and Nemotron Ultra256K profiles add
`task.target_context_tokens=245760`. Page reading and counting rewards are
unchanged: the prompt starts short and grows through tool results. For each
problem, `task.page_chars` is an initial estimate; the actual tokenizer and
renderer resize the pages until the context after reading all valid document
pages is within 1,024 tokens of 240K. Calibration uses minimal sequential
`read_page` calls, not generated reasoning. It recomputes the count from the
resized document and fails explicitly if the target cannot be reached.

The 262,144-token trajectory limit and 8,192-token generation reservation
remain in force. Extra reasoning, repeated pages, early answers or parse
failures can still prevent a rollout from completing the document. Reports
retain `target_context_tokens` and `calibrated_context_tokens`, while
`max_sampled_prompt_tokens` measures prompts actually used for sampled
actions. Do not mistake calibration or a rendered-but-unsampled observation
for measured 240K sampler/trainer evidence. Other model profiles and the
standalone defaults are unchanged.

Both thinking protocols deliberately differ from the upstream figure: they enable
thinking, hide the page count and use Qwen3.8. The reference uses non-thinking
Qwen3.6 curves and a separate GLM historical-bug example. Compare metric
definitions, not identical numerical curves; do not fabricate a buggy control.

## Metrics and local reports

The primary metric is **sampler/trainer KL v1**
(`optim/kl_sample_train_v1`, in nats/token), shown first in the automated dashboard and local
evidence report. KL v2, counting reward/accuracy, gradient signal and context
depth are supporting diagnostics. None alone establishes useful learning.
Automated completion requires finite KL v1 and v2 for every completed update
in the final recovery attempt. Missing or invalid latest KL is shown explicitly,
never replaced by an older finite measurement.
Non-finite metric values are written as valid JSON objects such as
`{"__nonfinite__": "NaN"}` (also `Infinity` and `-Infinity`), preserving their
update index and other metrics. Dashboards show an invalid latest measurement
and chart gaps; evidence tables retain the invalid state. A final recovery
attempt without a metrics file records zero updates and incomplete evidence,
not the earlier attempt's completed total.

These are driver-side measurements, not metrics scraped from the Loom UI:

1. The sampler returns generated tokens and log probabilities.
2. The trainer evaluates the same target tokens. Only positive action-mask
   entries contribute to sampler/trainer KL; prompts and tool observations do
   not contribute, although they count toward context position.
3. For `d = logp_sampler - logp_trainer`, `optim/kl_sample_train_v1 = mean(d)`
   and `optim/kl_sample_train_v2 = mean(d*d)/2`, pooled over action tokens.
   The first estimator can be negative. Neither is the reference-policy penalty
   or a general model-quality score; no universal passing threshold is imposed.
4. The environment supplies reward, answer accuracy, page-reading, parse and
   overflow metrics; the backend returns optimizer metrics such as gradient norm.

The normal log directory contains `config.json`, `run_meta.json`,
`metrics.jsonl`, `checkpoints.jsonl` and final `timing.json`.

| Measurement | Interpretation |
|---|---|
| `env/all/reward/total`, `env/all/exact_match` | Counting reward and answer accuracy. |
| `env/all/parse_error`, `env/all/context_overflow` | Failures that can confound reward variation. |
| `env/all/ac_tokens_frac_at_max` | Fraction of assistant turns reaching the generation cap. |
| `env/all/by_group/frac_mixed`, `skyrl.ai/grad_norm` | Within-group reward differences and gradient signal; neither alone proves useful learning. |
| `env/all/max_context_tokens` | Mean of episode peak contexts, not the global maximum. |
| `optim/kl_sample_train_max_action_position` | Deepest evaluated action-token position in the actual training sequences. |
| `extension_rate` | Measured token-prefix preservation; retaining thinking does not guarantee exact extension for arbitrary responses. |

Depth diagnostics use
`optim/kl_sample_train_by_depth/{0_8k,8_16k,16_32k,32_64k,over_64k}/{tokens,v1,v2}`.
Positions are zero-based in the complete training sequence, with boundaries
8192, 16384, 32768 and 65536; the first next-token target has position 1.
Empty bins have count zero and no KL value. Existing aggregate KL calculations
are unchanged; depth counts and estimates are additional shared RL diagnostics.

Generate a dependency-free HTML/SVG report from one or more run directories:

```powershell
python -m programmed_training.recipes.rl_numerics_check.report `
  "<run-directory>" --output "<evidence-directory>\rl-numerics.html"
```

The report plots KL by update and step-0 context depth, with task metrics,
configuration, checkpoint records and timing. Missing measurements stay
explicit; nonpositive estimates use a linear scale. After three updates, ETA
uses completed `time/total` durations and the configured step limit. It excludes
unmeasured setup/finalization and capacity wait and assumes unchanged contention.
Use `--include-recovery-attempts` to follow the cookbook's explicit recovery
journal. Each attempt remains a separate panel and series; attempts without
completed metrics are identified rather than reconstructed. This optional mode
uses the installed cookbook's recovery helper; ordinary local reporting remains
dependency-free.

Evaluation is disabled in the stress command to isolate training cost. To
enable it, set `max_test_examples` and `eval_every` to positive values. Evaluation
uses different deterministic problem seeds from the same corpus, not held-out
text or evidence of generalization.

## Checkpoints and recovery

The shared RL loop saves training checkpoints to `checkpoints.jsonl` according
to `save_every`, plus a final checkpoint on successful completion. Saved
checkpoints are retained. The recipe launcher retains cleanup ownership of its
initial session and client across the entire shared-loop call, including logging,
checkpoint, tokenizer and training-client setup failures and cancellation.
A recipe-local SDK client shares the results of close calls so cleanup requested
by the shared loop or recovery is not repeated against a closed HTTP client;
cleanup errors are preserved rather than converted into successful results.
Launcher session/client cleanup has a bounded time budget, with time reserved
for closing the client. A cancelled cached close task is logged as a cleanup
failure without replacing an active training or creation exception.
After a successful recovery-supervised run has closed the original client,
launcher finalization does not replay a terminal close error already handled
by the supervisor. Unhandled conflicts, wrong-session errors and setup/cleanup
failures still propagate; this does not assert physical GPU teardown.
The shared RL loop itself is unchanged.

The recipe-local client records the allocation ID from a successful session
POST response through the SDK's public `send_request` method, before the SDK
polls for model readiness. If polling times out, is cancelled, or fails without
an ID on the exception, the launcher still attempts to close that allocation
before closing the SDK client, with recovery enabled or disabled. It also
retains IDs exposed by `TrainingEngineError`. Cleanup failures are logged
without replacing the original creation failure. No SDK internals are accessed
and no SDK source changes are required.

If the service allocates a session but its POST response is never received,
the recipe still has no allocation ID. A killed launcher or a failed remote
close also cannot guarantee release; check for outstanding allocations before
retrying these cases.

Like other public training recipes, `auto_resume=True` enables best-effort
recovery after eligible `engine_dead` failures, with
`auto_resume_max_retries=3`. Set `auto_resume=False` to opt out. Replacement
sessions use the latest usable checkpoint and progress, or restart from the
original input if none was committed; unsaved rollouts may be repeated.
This does not recover a killed local launcher or fix sampling-route failures.
See [automatic engine-loss recovery](../../../docs/training.md#automatic-engine-loss-recovery).

Reuse `log_path` with `behavior_if_log_dir_exists=resume` to resume from its
checkpoint ledger and cursor. To start a fresh run from weights, use a fresh
directory and `load_checkpoint_path=<session>/<checkpoint>`. An existing ledger
takes precedence over an explicit checkpoint override, with a warning.
See [checkpoint recovery](../../../docs/continual-fine-tuning.md).

## Validation status

As of October 27, 2026:

- Local tests cover task grading, tool parsing, token budgets, CLI forwarding,
  tier propagation, checkpoint cleanup, Qwen3.8 thinking retention/action masks,
  KL depth boundaries and report generation. Real-tokenizer tests use cached
  public tokenizer files; pytest never launches paid training.
- A historical Qwen3.6 bounded run completed two updates and saved `1`, `2`
  and `final` in 20.6 minutes with a 32768-token trajectory budget. Ten of
  sixteen rollouts had parse failures and three overflowed. This verifies a
  training path, not clean learning or Qwen3.8/256K execution.
- The Qwen3.8-256K preflight admitted a session and exported initial sampler
  weights, but completed zero optimizer updates. API logs showed both LoomBus
  flags disabled for the subscription/model and no enabled HTTP inference
  endpoint for the alias. A later client credential timeout ended the driver.
  Fix and qualify service-side sampling dispatch before retrying.
- No full-task or actual 256K training/checkpoint-reload qualification is claimed.

From the cookbook root:

```powershell
python -m pytest tests\test_rl_numerics_check.py tests\test_rl_kl_depth.py tests\test_rl_numerics_report.py -q
python -m programmed_training.recipes.rl_numerics_check.train_azure project_endpoint="https://<resource>.services.ai.azure.com/api/projects/<project>" --help
```
