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

Dependency installation, network dataset downloads, credential prompts, or resource reads are not implied by an offline preflight. Explain them and obtain approval where needed.

## Explicit approval before spend or mutation

Approval is required before uploads, fine-tuning job creation/resume, deployment
creation, session allocation, interactive recipes, sampling/inference, external
tools, or cancellation/unload/deletion. Before the named operation, present:

- Task, target tenant/subscription/project, exact model and region, and verified
  access/role/quota prerequisites or unresolved checks.
- Input provenance, license/privacy constraints, counts and hashes, and intended
  upload destination. Do not expose populated endpoints or runtime identifiers.
- Method, steps/epochs/tokens, evaluation/tool-call scope, timeouts, stop
  conditions, and budget. State known pricing or unresolved cost uncertainty;
  time and step limits are not dollar caps.
- Expected artifacts, held-out evaluation and serving limitations, private
  evidence retention, and cleanup responsibilities for every owned resource.

Wait for explicit approval covering the operation and budget. "Help me fine-tune"
is not blanket spend authorization. Do not provision or change permissions
automatically. Obtain separate deletion confirmation; stopping monitoring is not
cancellation, and closing a client is not remote unload.

Recurring runs require explicit approval of frequency, models/data, evaluation
scope, budget per run and over the approved period, expiry, owner, and disable/
reconciliation procedure. Manual-run approval is not recurring-run authority.
Follow [the paid smoke operator guide](PAID_SMOKE_TESTING.md); do not enable its
workflow or change environment permissions automatically.

## Secrets and runtime state

Never print or commit populated `.env` files, keys, tokens, endpoint/resource identifiers, or runtime manifests. Use supported credential providers; do not add silent authentication fallbacks. Logs must redact secrets. Keep runtime identifiers in the selected demo's ignored `outputs/` or the interactive run location documented by its storage guide.

## Monitor, recover, evaluate, and clean up

Record job/session/checkpoint IDs locally and monitor terminal state with the documented helpers. A timeout is not proof that the remote operation failed or was cancelled. Inspect existing state before retrying an upload or submission; use documented reuse/resume contracts and do not create duplicates blindly.

Save durable checkpoints and driver state before interruption. Distinguish training-state checkpoints from sampler checkpoints and deployed endpoints. Most managed demos stop at training metrics; only Retail adds comparisons with externally provisioned deployments. Report that boundary, and never turn lower loss or a past benchmark into a claim of held-out task improvement.

At handoff, report operations performed, artifacts and remaining resources, evaluation limitations, and ongoing cost exposure without exposing secrets. Cancellation, session unload, local-client closure, and deployment deletion are separate actions. Do not clean up shared or unidentified resources.
