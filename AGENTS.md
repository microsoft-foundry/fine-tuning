# Coding-agent execution contract

These instructions apply to the repository. Use the public READMEs and maintained recipes; no separate fine-tuning skill installation is required.

## Route and resolve sources

1. Start with the root README, then the chosen track's README and setup guide. Use managed jobs for supported predefined methods; use interactive sessions for custom training-loop behavior. A custom managed grader alone does not require interactive training.
2. For managed work, read `managed_fine_tuning/README.md`, `catalog.yml`, the selected demo's `demo.yaml`, README, `.env.template`, and canonical `notebooks/demo.ipynb`. Use the learning path/task index to select the demo.
3. For interactive work, read `interactive_training/README.md`, `docs/quickstart.md`, `docs/auth.md`, and the selected recipe's README and configuration.
4. Interactive model/region eligibility comes from `interactive_training/docs/supported_models.md`. Managed eligibility comes from Foundry's model-specific documentation and the catalog prerequisites; notebook model roles are not a universal availability guarantee.
5. Track-specific setup guides define repository paths. Subtree metadata/templates define data and configuration; installed SDK signatures and implementation define operations. Historical changelog entries, benchmarks, and past runtime artifacts do not override these sources. If they conflict, report the conflict and stop before remote operations rather than guessing.

Before updating any README, check whether a more focused guide, recipe, or reference
document is a better home for the change. Keep READMEs useful as human-facing entry
points: link to detailed material rather than duplicating it or adding unrelated
implementation details.

## No-spend preflight

Read files and run offline tests, syntax/schema checks, dataset validation, and split/hash checks first. Do not run a whole notebook as a smoke test: later cells may upload data, allocate training, invoke tools, or submit/resume a paid job without another switch.

- Use the appropriate track's environment; managed uses Python 3.11-3.14 and its central lock, interactive uses its own package and SDK requirements.
- Check paths, kernel, working directory, exact model ID, schema, and prerequisites. Managed demos should run from their demo directory unless a tested alternative is documented.
- Preserve canonical/hash-pinned sample files. Put personal-data adaptations in separate, versioned inputs; never bypass validators to make them run.

Run root navigation and model-eligibility checks without cloud credentials:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Use the repository workflow's selected managed tests for the managed contracts. Run interactive tests from `interactive_training` using its test environment. Tests validate local contracts, not cloud eligibility.

Dependency installation, network downloads, authentication, and resource reads are not offline checks. When needed for the requested workflow, explain them briefly and proceed within that scope using the selected environment and supported credential providers. Ask for missing information that changes the scope, not repeated approval for routine setup or reads; never request secrets in chat.

## Paid execution: warn, then proceed

When the user requests a training, evaluation, or sampling run, perform its
required uploads, job/session creation or resume, checkpoint saves, and
owned-resource cleanup without a separate spend-approval prompt. This applies
to all users and environments, including Autopilot; it is not an internal-only
exception.

Before the first paid operation in the requested workflow, give one brief
warning: **"This workflow uses paid cloud resources and may incur charges."**
Include relevant cost information if readily available. Do not require a
budget amount, cost acknowledgement, or pricing lookup, and do not block or
repeatedly ask for approval because prices or a dollar budget are unknown.
Respect any limits the user supplies; time, step, and token limits are not
dollar caps.

Keep the run scoped to the user's request and the selected recipe. Record the
target, exact model/region, input provenance/counts/hashes, configuration, stop
conditions, artifacts, and cleanup ownership in private runtime evidence.
Explain meaningful scope choices briefly rather than turning them into a
spending-approval handoff. Continue to validate data, eligibility, permissions,
quota prerequisites, and required preview access; a cost warning does not
replace those checks.

Do not add unrequested deployments, resource provisioning, permission changes,
external tools, or recurring work. Cancellation/unload within the requested
run's stop and cleanup plan must target only its identified owned resources;
save recoverable state first and verify the remote result. Obtain separate
deletion confirmation. Stopping monitoring is not cancellation, and closing a
client is not remote unload.

Recurring runs require an explicit request naming the frequency, models/data,
evaluation scope, expiry, owner, and disable/reconciliation procedure. A manual
run request is not a request for recurring work. Follow
[the paid smoke operator guide](PAID_SMOKE_TESTING.md); do not enable its
workflow or change environment permissions automatically.

## Secrets and runtime state

Never print or commit populated `.env` files, keys, tokens, endpoint/resource identifiers, or runtime manifests. Use supported credential providers; do not add silent authentication fallbacks. Logs must redact secrets. Keep runtime identifiers in the selected demo's ignored `outputs/` or the interactive run location documented by its storage guide.

## Monitor, recover, evaluate, and clean up

Record job/session/checkpoint IDs locally and monitor terminal state with the documented helpers. A timeout is not proof that the remote operation failed or was cancelled. Inspect existing state before retrying an upload or submission; use documented reuse/resume contracts and do not create duplicates blindly.

Save durable checkpoints and driver state before interruption. Distinguish training-state checkpoints from sampler checkpoints and deployed endpoints. Most managed demos stop at training metrics; only Retail adds comparisons with externally provisioned deployments. Report that boundary, and never turn lower loss or a past benchmark into a claim of held-out task improvement.

At handoff, report operations performed, artifacts and remaining resources, evaluation limitations, and ongoing cost exposure without exposing secrets. Cancellation, session unload, local-client closure, and deployment deletion are separate actions. Do not clean up shared or unidentified resources.
