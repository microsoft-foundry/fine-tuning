# Post-training evaluation, serving, cancellation, and cleanup

Managed notebooks establish terminal job status and service-produced training
metrics. Except for the Retail Capstone's optional comparison, they do not
provision deployments or prove held-out application quality.

## Identify the trained artifact

After a job succeeds, retrieve it and require a non-empty
`fine_tuned_model` value:

```python
job = openai_client.fine_tuning.jobs.retrieve(job_id)
if job.status != "succeeded" or not job.fine_tuned_model:
    raise RuntimeError(
        f"Job is not ready for serving: status={job.status!r}, "
        f"error={job.error!r}"
    )
result_model = job.fine_tuned_model
```

Store the job ID and result-model identifier only under the demo's ignored
`outputs/`. A result model is a training artifact, not a serving endpoint.
Availability, deployment type, region, capacity, and model-version support must
be checked again in the target project.

## Evaluate and serve

1. Keep a held-out set that was not used for training, validation, teacher
   generation, grader tuning, or example selection.
2. Define task, quality, safety, and regression criteria before sampling.
3. Provision a deployment through your approved Foundry portal or management
   workflow, using `result_model` as the model artifact. Record the deployment
   name locally; do not add it to committed `.env` templates.
4. Invoke the deployment with the supported chat or Responses API and compare
   it with an appropriate base-model deployment on the same held-out cases.
5. Treat training loss/reward as optimization evidence only. Do not claim
   application improvement without held-out results.

The Retail Capstone demonstrates read-only verification and comparison of
already provisioned deployments. It never creates, updates, or deletes them.
Follow current
[Microsoft Foundry fine-tuning documentation](https://learn.microsoft.com/azure/foundry/openai/how-to/fine-tuning)
for model-specific deployment and inference support.

## Stop monitoring versus cancel training

Closing a notebook, timing out a polling loop, or closing local SDK clients
does not cancel a remote job. Cancellation is a separate mutating operation.
For the two-job Countdown state, follow its
[cancellation and cleanup guide](03-core-rft/01-rft-countdown/CANCELLATION_AND_CLEANUP.md)
instead of the first-SFT example below.
Run the following from the first-SFT cookbook directory after reviewing the
ignored `outputs/submission-state.json`. It reconstructs a live client, verifies
that the saved job belongs to the configured project, and requires you to paste
the exact job ID before it sends the cancellation request:

```python
import json
from pathlib import Path

from shared.auth import create_project_context
from shared.config import load_foundry_config

state = json.loads(
    Path("outputs/submission-state.json").read_text(encoding="utf-8")
)
job_id = state.get("job_id")
saved_endpoint = state.get("equivalence", {}).get("project_endpoint")
if not job_id or not saved_endpoint:
    raise RuntimeError("Saved state does not contain a verified project and job ID.")

config = load_foundry_config(".env")
if config.project_endpoint != saved_endpoint:
    raise RuntimeError("The configured project does not match the saved job state.")

confirmed_job_id = input("Paste the exact job ID to confirm cancellation: ").strip()
if confirmed_job_id != job_id:
    raise RuntimeError("Cancellation confirmation did not match the saved job ID.")

with create_project_context(config) as context:
    current = context.openai_client.fine_tuning.jobs.retrieve(job_id)
    if str(current.status).casefold() in {"succeeded", "failed", "cancelled"}:
        raise RuntimeError(f"Job is already terminal: {current.status}")
    requested = context.openai_client.fine_tuning.jobs.cancel(job_id)
    print(f"Cancellation request status: {requested.status}")
```

Before cancellation, confirm the exact project and job ID, expected impact,
authorization, and remaining cost uncertainty. Do not automatically retry an
ambiguous cancellation response. Retrieve the job in Foundry until it reaches a
terminal state; a cancellation request is not proof that cancellation completed.
This snippet is cancellation-only; deployment, inference, and comparison remain
separate approved workflows.

Uploaded-file deletion is also separate. Only after confirming that a saved
file ID is not shared by another job, reconstruct the context again and require
an exact per-file confirmation:

```python
import json
from pathlib import Path

from shared.auth import create_project_context
from shared.config import load_foundry_config

state = json.loads(
    Path("outputs/submission-state.json").read_text(encoding="utf-8")
)
saved_endpoint = state.get("equivalence", {}).get("project_endpoint")
file_ids = [
    state.get("training_file_id"),
    state.get("validation_file_id"),
]
if not saved_endpoint or not all(file_ids):
    raise RuntimeError("Saved state does not contain a verified project and file IDs.")

config = load_foundry_config(".env")
if config.project_endpoint != saved_endpoint:
    raise RuntimeError("The configured project does not match the saved file state.")

with create_project_context(config) as context:
    for file_id in file_ids:
        context.openai_client.files.retrieve(file_id)
        confirmation = input(f"Type DELETE {file_id} to delete this file: ").strip()
        if confirmation != f"DELETE {file_id}":
            raise RuntimeError("File deletion was not confirmed.")
        result = context.openai_client.files.delete(file_id)
        print(f"Deleted={result.deleted}")
```

Do not infer that a file is unshared from its filename. Check the owning jobs
and retention requirements first. Do not retry an ambiguous deletion response
without retrieving the file and reconciling its current state.

## Cleanup checklist

- **Local clients:** close OpenAI, project, and credential clients. This releases
  local resources only.
- **Training job:** retain the job and result references needed for audit or
  explicitly cancel a nonterminal job as described above.
- **Uploaded files:** delete only confirmed, unshared file IDs through an
  approved operation. File deletion is separate from job cancellation.
- **Deployment:** scale down or delete a deployment through the approved
  management workflow only after separate confirmation. This is the primary
  ongoing serving-cost boundary.
- **Local outputs:** retain sanitized manifests and required evidence; remove
  identifier-bearing ignored state only when the remote resources have been
  reconciled and the retention requirement is understood.

At handoff, report the terminal job state, result-model artifact, held-out
evaluation limitations, deployments that still exist, uploaded files retained,
and any ongoing cost exposure without exposing endpoints or identifiers.
