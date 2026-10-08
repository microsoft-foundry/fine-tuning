# Paid smoke testing: operator guide

The [smoke workflow](.github/workflows/smoke-testing.yml) runs real interactive
training, not offline validation or held-out application-quality evaluation.
It is opt-in. **Do not enable or dispatch it without named operation and budget
approval.** Use [AGENTS.md](AGENTS.md) for the shared approval handoff.

## Configure and authorize

An authorized repository/environment administrator must:

1. Confirm the tenant/subscription/project, exact models and region, authorization,
   capacity, and current prices. Successful sign-in or a listed region does not
   establish training eligibility. Follow the [access checklist](interactive_training/docs/auth.md).
2. Review the [runner](interactive_training/scripts/run_recipe_smokes.py) and the
   [Tulu3 SFT](interactive_training/interactive_training/recipes/tulu3_sft/README.md)
   and [tool-NER RL](interactive_training/interactive_training/recipes/tool_ner_rl/README.md)
   data sources, licenses, downloads, tool usage, evaluation, and checkpoint scope.
   Record input counts/hashes privately; step limits do not bound every input,
   evaluation, or tool call.
3. Configure the `interactive-training-smoke` GitHub environment with required
   reviewers, prevent self-review where supported, and restrict permitted
   branches to approved code. Verify these protections actually exist: the YAML
   environment name alone does not configure them. If repository/plan settings
   cannot enforce review, keep the gate off until an approved alternative exists.
4. Configure `AZURE_AI_PROJECT_ENDPOINT` and `AZURE_AI_API_KEY` as environment
   secrets. Never print their values or place them in source, public logs, or
   issue/PR comments. This workflow requires both; do not silently switch its
   authentication method.
5. Approve either a single manual execution or recurring authority naming the
   cadence, models/data, evaluation/tool scope, budget per run and total period,
   expiry, responsible operator, alerts, evidence storage, and cleanup plan.
   The workflow does not enforce a monetary cap or approval expiry; an operator
   must arrange spending alerts and disable it when authority expires.

Keep repository variable `ENABLE_DAILY_TRAINING` unset or `false` until all
prerequisites, including evidence retention, are satisfied. Setting it to exactly
`true` enables **both** manual and scheduled jobs. The daily cron is `0 0 * * *`
(00:00 UTC on the default branch; scheduling can be delayed). An approved manual
run uses Actions > Smoke testing > Run workflow on an approved ref and still
requires the gate and environment approval.

For single-run authority, enable the gate only for the approved dispatch window,
reject any unrelated scheduled/queued run at the environment review, and turn the
gate off again immediately afterward. The current workflow has no manual-only
enable switch; do not leave it enabled on the basis of one manual approval.

## What the bounds mean

| Control | Current scope | Not a guarantee of |
|---|---|---|
| Tulu3 SFT | `Qwen/Qwen3.8-27B`, at most 30 steps, one epoch, evaluation/save every 10 steps | Bounded dollar cost or downstream quality |
| Tool-NER RL | `Qwen/Qwen3.6-35B-A3B`, two steps, `max_test_examples=16`, `tab_eval_documents=1`, evaluation/save every step | All data, rollouts, or tool activity being capped |
| Recipe wall clock | 7,200 seconds per recipe | Immediate remote termination at that time |
| Child-process timeout | 9,000 seconds per recipe | Remote session unload or checkpoint deletion |
| Status verification | Default 10-minute polling window after each successful child | Zero remaining resources/cost when status is `succeeded` |
| Workflow | Six-hour job timeout; sequential recipes with a ten-minute gap | Monetary cap, automatic recovery, or cancellation of remote work |
| Concurrency | Serialized workflow runs; `cancel-in-progress: false` | Automatic cancellation of an older run; unlimited queued-run prevention |

The runner stops on child failure, timeout, or failed status verification.
It uses unique log directories and rejects reuse. Check the current workflow
and runner before approving changed commands; these limits describe today's code.

## Preserve private evidence before enabling

The runner writes `interactive_training/logs/recipe-smoke-<unique-id>/`, including
driver logs and recipe `run_meta.json` with primary/reference/staged session IDs.
Driver files use mode 0600 on the Linux runner, and the displayed endpoint is
redacted. This is not proof that all file contents are sanitized.

**The current workflow has no evidence-export step.** Hosted-runner files are
ephemeral; a failure message pointing at a runner path is not durable access.
Before enabling, configure and verify an approved private export/retention
mechanism that runs on success and failure, plus a recovery procedure for hard
cancellation/runner loss. Keep the gate off if IDs and evidence cannot be recovered.
Do not upload raw logs as public or broadly accessible Actions artifacts.
Repository access is not automatically appropriate access to training data.

Record workflow run/ref, recipe/model/configuration, input evidence, owned session
and checkpoint IDs, terminal status, evaluation limitations, and resource/cost
reconciliation privately. Approve the destination, access controls, retention
period, and deletion owner. Publish only a sanitized outcome, never credentials,
populated endpoints, raw data, or runtime manifests. This guide does not create
storage, grant permissions, or provide a finished exporter.

## Disable, stop, and reconcile

1. To prevent new paid jobs, set `ENABLE_DAILY_TRAINING=false` (or remove it);
   optionally disable the workflow in Actions. Check queued/waiting runs.
   Disabling the gate/workflow is not cancellation of a running job.
2. Cancelling an Actions run stops local execution, not necessarily remote
   training. Preserve evidence first when possible; inspect every session owned
   by that run before retrying. A timeout or lost response is an unknown remote
   outcome, not proof of failure.
3. Follow [session management](interactive_training/docs/session-management.md)
   to inspect primary, reference, and staged sessions from private run metadata.
   If metadata is missing, reconcile approved project state and ownership with
   the operator; never guess IDs or clean up shared resources.
4. Obtain named approval for remote cancellation/unload and separate confirmation
   for checkpoint/resource deletion. Verify terminal state and retained artifacts.
   Local client closure, training completion, unload, and deletion are distinct.
5. Document remaining sessions/checkpoints/deployments and uncertain cost exposure.
   Retry only after reconciliation, with fresh approval if scope/budget changed.

Neither green Actions status nor a `succeeded` session demonstrates held-out
quality, serving readiness, or complete resource release.
