# Coding-agent execution contract

These instructions apply to the repository. Use the READMEs, code and recipes; no separate fine-tuning skill installation is required.

## Resolve sources

1. Start with the root README, then the chosen track's README and setup guide. Use managed jobs for supported predefined methods; use interactive sessions for custom training-loop behavior. A custom managed grader alone does not require interactive training.
2. For managed work, read `managed_fine_tuning/README.md`, `catalog.yml`, the selected demo's `demo.yaml`, README, `.env.template`, and canonical `notebooks/demo.ipynb`. Use the learning path/task index to select the demo.
3. For interactive work, read `interactive_training/README.md`, `docs/auth.md`, and the selected recipe's README and configuration.
4. Interactive model/region eligibility comes from `interactive_training/docs/supported_models.md`. Managed eligibility comes from Foundry's model-specific documentation and the catalog prerequisites; notebook model roles are not a universal availability guarantee.
5. Track-specific setup guides define repository paths. Subtree metadata/templates define data and configuration; installed SDK signatures and implementation define operations. Historical changelog entries, benchmarks, and past runtime artifacts do not override these sources. If they conflict, report the conflict and stop before remote operations rather than guessing.

## Explicit approval before spend or mutation

The following operations incur cost or resource usage, either directly or with followup operations, and thus require explicit approval before execution:
- Fine-tuning job creation; as a part of Managed fine-tuning
- Fine-tuning session creation, sampling and training; as a part of Interactive training recipe
- Deployment creation and inference
- Evaluations that internally use sampling or deployment inference

Apart from these, deletion or cancellation of resources also require explicit approval. For instance deletion of fine-tuning jobs, sessions and deployments.

Related instructions:
- Approval requests should have clear options e.g. yes or no. Don't raise requests where user needs to enter text. 
- Raise consolidated approval request over recurring operations beforehand instead of completing task partially and stopping due to lack of approval later.
- **Don't raise approval requests for just about anything; follow the specific guidance here.** For instance, don't raise requests for budget or read-only calls as they're not part of the guide here. Not all operations require explicit approval. Read the "Approval-exempt operations" section for details.

### Approval-exempt operations

These operations can be performed without explicit approval:
- Read-only API calls like querying model metadata or listing deployments
- File upload and download
- Dependency installation
- Local operations like use of dashboard server
- Local evaluations or local grading that do not use sampling or deployment inference

### Presentation of Approval Requests

Before executing operation(s) that incur cost or resource usage, try to share a few details based on local context:
- Foundry project and region
- Operation and target model
- Training Method. Managed fine-tuning: SFT/RFT/DPO. Interactive training: Sampling/Training
- Training type: Standard/GlobalStandard/DatazoneStandard
- Deployment type: Standard/GlobalStandard/DatazoneStandard/others
- Deployment format (Fireworks/others) and SKU capacity

## Secrets and runtime state

Never print or commit populated `.env` files, keys, or tokens.

## Monitor, recover, evaluate, and clean up

Record job/session/checkpoint IDs locally and monitor terminal state with the documented helpers. A timeout is not proof that the remote operation failed or was cancelled. Inspect existing state before retrying an upload or submission; use documented reuse/resume contracts and do not create duplicates blindly.

Save durable checkpoints and driver state before interruption. Distinguish training-state checkpoints from sampler checkpoints and deployed endpoints. Most managed demos stop at training metrics; only Retail adds comparisons with externally provisioned deployments. Report that boundary, and never turn lower loss or a past benchmark into a claim of held-out task improvement.

At handoff, report operations performed, artifacts and remaining resources, evaluation limitations, and ongoing cost exposure without exposing secrets. Cancellation, session unload, local-client closure, and deployment deletion are separate actions. Do not clean up shared or unidentified resources.

## Content update instructions

Before updating any README, check whether a more focused guide, recipe, or reference
document is a better home for the change. Keep READMEs useful as human-facing entry
points: link to detailed material rather than duplicating it or adding unrelated
implementation details.
