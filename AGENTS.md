# Coding-agent execution contract

These instructions apply to the repository. Use the READMEs, code and recipes; no separate fine-tuning skill installation is required.

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

Use the repository workflow's selected managed tests for the managed contracts. Run interactive tests from `interactive_training` using its test environment. Tests validate local contracts, not cloud eligibility.

Dependency installation, network dataset downloads, credential prompts, or resource reads are not implied by an offline preflight. Explain them and obtain approval where needed.

## Explicit approval before spend or mutation

The following operations incur cost or resource usage, either directly or with followup operations, and thus require explicit approval before execution:
- Fine-tuning job creation; as a part of Managed fine-tuning
- Fine-tuning session creation, sampling and training; as a part of Interactive training recipe
- Deployment creation and inference
- Evaluations that internally use sampling or deployment inference

Some operations don't incur cost and can be performed without explicit approval:
- File upload and download
- Local operations e.g. use of dashboard server
- Local evaluations or local grading that do not use sampling or deployment inference

Apart from these, deletion or cancellation of resources also require explicit approval. For instance deletion of fine-tuning jobs, sessions and deployments.

### Presentation of Approval Requests

Before executing operation(s) that incur cost or resource usage, present a few key details:
- Operation, target model, foundry project and region
- Training Method. Managed fine-tuning: SFT/RFT/DPO. Interactive training: Sampling/Training
- Training type: Standard/GlobalStandard/DatazoneStandard
- Deployment type: Standard/GlobalStandard/DatazoneStandard/others
- Deployment format (Fireworks/others) and SKU capacity

Wait for explicit approval covering the operation. "Help me fine-tune"
is not blanket spend authorization. Do not provision or change permissions
automatically.

Approval request should also cover recurring operations.


## Secrets and runtime state

Never print or commit populated `.env` files, keys, tokens, endpoint/resource identifiers, or runtime manifests. Use supported credential providers; do not add silent authentication fallbacks. Logs must redact secrets. Keep runtime identifiers in the selected demo's ignored `outputs/` or the interactive run location documented by its storage guide.

## Monitor, recover, evaluate, and clean up

Record job/session/checkpoint IDs locally and monitor terminal state with the documented helpers. A timeout is not proof that the remote operation failed or was cancelled. Inspect existing state before retrying an upload or submission; use documented reuse/resume contracts and do not create duplicates blindly.

Save durable checkpoints and driver state before interruption. Distinguish training-state checkpoints from sampler checkpoints and deployed endpoints. Most managed demos stop at training metrics; only Retail adds comparisons with externally provisioned deployments. Report that boundary, and never turn lower loss or a past benchmark into a claim of held-out task improvement.

At handoff, report operations performed, artifacts and remaining resources, evaluation limitations, and ongoing cost exposure without exposing secrets. Cancellation, session unload, local-client closure, and deployment deletion are separate actions. Do not clean up shared or unidentified resources.
