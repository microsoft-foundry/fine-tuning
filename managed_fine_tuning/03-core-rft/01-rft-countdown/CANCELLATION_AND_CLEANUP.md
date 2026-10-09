# Countdown job cancellation and cleanup

Countdown saves two jobs. The [first-SFT post-training examples](../../POST_TRAINING.md)
use a different state format and cannot be used unchanged here.

## Before requesting cancellation

Closing the notebook or timing out monitoring does **not** cancel training.
Cancellation loses progress and does not undo accrued charges. Cancel only an
identified owned job within the requested stop/cleanup plan or an explicit
cancellation request; no separate spending approval is required. Check Foundry
for jobs missing from local state after an ambiguous submission.

From the Countdown demo directory, review the ignored
`outputs/submission-state.json` privately. Its `jobs` map holds `model-grader`
and `python-grader` IDs, but no per-job project endpoint in current runs. Check
the project independently in Foundry; an optional `job_endpoints` entry in
older state is checked but is **not** sufficient proof on its own. Do not
retroactively fill it from `.env`, or share populated state or credentials.

With the managed environment installed, run this snippet **once** for the
requested cancellation. It retrieves and confirms **one** selected job before
requesting cancellation; it does not cancel both jobs:

```python
import json
from dataclasses import replace
from pathlib import Path

from shared.auth import create_project_context
from shared.config import load_foundry_config

state = json.loads(Path("outputs/submission-state.json").read_text(encoding="utf-8"))
label = input("Select one job (model-grader or python-grader): ").strip()
if label not in {"model-grader", "python-grader"}:
    raise ValueError("Select exactly one known Countdown job label.")
job_id = state.get("jobs", {}).get(label)
if not isinstance(job_id, str) or not job_id.strip():
    raise RuntimeError("No recorded job ID for the selected label; reconcile in Foundry.")

config = replace(load_foundry_config(".env"), allow_preview=True)
saved_endpoint = state.get("job_endpoints", {}).get(label)
if saved_endpoint is not None and saved_endpoint != config.project_endpoint:
    raise RuntimeError("Saved job project does not match the configured project.")

# Obtain this independently from Foundry, not from state or .env.
verified_endpoint = input("Paste the project endpoint from Foundry: ").strip().rstrip("/")
if verified_endpoint != config.project_endpoint:
    raise RuntimeError("Foundry project does not match the configured project.")

with create_project_context(config) as context:
    current = context.openai_client.fine_tuning.jobs.retrieve(job_id)
    if current.id != job_id or current.model != "qwen3.6-35b-a3b":
        raise RuntimeError("Retrieved job does not match the Countdown state.")
    status = str(current.status).casefold()
    if status in {"succeeded", "failed", "cancelled", "canceled"}:
        raise RuntimeError("The selected job is already terminal; do not cancel it.")
    if status in {"cancelling", "canceling"}:
        raise RuntimeError("Cancellation is already in progress; inspect status in Foundry.")

    if input("Type the selected job label to confirm: ").strip() != label:
        raise RuntimeError("Job label confirmation did not match.")
    if input("Paste the exact selected job ID to confirm: ").strip() != job_id:
        raise RuntimeError("Job ID confirmation did not match.")
    requested = context.openai_client.fine_tuning.jobs.cancel(job_id)
    print("Cancellation requested; returned status:", requested.status)
```

**Do not blindly retry** a failed or ambiguous request. Check the job in
Foundry until terminal; a cancellation response is not proof it stopped.
Reconcile the other job separately and retain private evidence.

## Other resources are separate operations

- **Uploaded files:** the jobs may share them; identify all owners before
    separately approving deletion and confirming each file ID. Countdown does
    not save these IDs, so the first-SFT deletion example does not apply.
- **Grader deployment:** consult the ignored
    `outputs/grader-deployment-state.json`. After relevant jobs are terminal,
    separately approve restoring prior capacity or removing a deployment you
    own; cancellation does neither.
- Keep final status, retained-file, deployment, and cost evidence private.