# Countdown job cancellation and cleanup

Countdown saves two jobs. The [first-SFT post-training examples](../../POST_TRAINING.md)
use a different state format and cannot be used unchanged here.

## Before requesting cancellation

Closing the notebook or timing out monitoring does **not** cancel training.
Cancellation loses progress and does not undo accrued charges. Cancel only an
identified owned job within the requested stop/cleanup plan or an explicit
cancellation request; no separate spending approval is required. Check Foundry
for jobs missing from local state after an ambiguous submission.

From the Countdown demo directory, review the ignored submission state privately.
Its `jobs` map holds `model-grader` and `python-grader` IDs. Current records also
retain `pending[label]`: the endpoint, exact creation payload, payload digest,
and training/validation file IDs and canonical hashes. The
[cancellation guard](cancellation.py) compares the retrieved job's model, both
file IDs, suffix, and complete method/grader (including hyperparameters and
response format) to **that label's saved payload**, not merely its job ID/model.
It also rejects conflicting observable training-type metadata. Missing/redacted
method, grader, or suffix identity stops this workflow; it is never inferred
from a deployment name or filled from the job being checked.

Independently open the intended job and project in Foundry and check their
ownership and operation/audit evidence. Copy the endpoint and intended job ID
from that independent review when prompted, **not** from the local state or
`.env`. Typing back an ID shown only by the same stale/swapped state is not
independent confirmation. An optional legacy `job_endpoints` entry is checked
but is not sufficient proof. Do not repair evidence to make cancellation pass,
or share populated state or credentials.

With the managed environment installed, run this snippet **once** for the
requested cancellation. It retrieves and confirms **one** selected job before
requesting cancellation; it does not cancel both jobs:

```python
import json
import sys
from dataclasses import replace
from pathlib import Path

# Resolve only this demo's cancellation guard; shared auth is unchanged.
sys.path.insert(0, str(Path.cwd()))
from cancellation import cancel_countdown_job_once
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
# Obtain both independently from the intended Foundry job and its audit evidence.
verified_endpoint = input("Paste the project endpoint from Foundry: ").strip().rstrip("/")
verified_job_id = input("Paste the independently verified intended job ID: ").strip()
confirmed_label = input("Type the selected job label to confirm: ").strip()

with create_project_context(config) as context:
    requested = cancel_countdown_job_once(
        context.openai_client,
        state=state,
        label=label,
        project_endpoint=config.project_endpoint,
        confirmed_project_endpoint=verified_endpoint,
        confirmed_job_id=verified_job_id,
        confirmed_label=confirmed_label,
    )
    print("Cancellation requested; returned status:", requested.status)
```

### Legacy or incomplete identity: independent workflow only

The snippet deliberately refuses job-only legacy state, missing intents,
unavailable suffixes, and redacted/missing grader metadata. Do not synthesize
`pending`, substitute a current recipe, or self-confirm from the retrieved job.
Instead stop this snippet, preserve the original state, and independently match
the intended owned job using the original submission/request/audit evidence,
both input files and their hashes, suffix, and grader/method in the correct
Foundry project. Only after that review and an explicit cancellation confirmation
may the operator request cancellation for that **independently identified** job
in Foundry. If identity cannot be proved, contact the service operator; no
automated fallback is authorized. This does not change shared authentication or
the cancellation workflows for other tracks.

**Do not blindly retry** a failed or ambiguous request. Check the job in
Foundry until terminal; a cancellation response is not proof it stopped.
Reconcile the other job separately and retain private evidence.

## Other resources are separate operations

- **Uploaded files:** the jobs may share them; identify all owners before
    separately approving deletion and confirming each file ID. Current intents
    retain these IDs under `pending[label].files`; legacy records may not.
    Neither is deletion authorization, and the first-SFT example does not apply.
- **Grader deployment:** consult the ignored
    `outputs/grader-deployment-state.json`. The first operation-intent record is
    saved **before** ARM create/update and is never replaced on reruns, including
    when an ARM request fails or its response is ambiguous. Match its project,
    subscription, resource group, account, deployment, model/version/SKU, and
    recipe to independent Foundry/ARM evidence. `previously_existed` distinguishes
    an existing zero-capacity deployment from a name that was absent;
    `previous_deployment_id` and `previous_capacity` preserve the existing
    resource's baseline. `requested_capacity` is an intent, not proof of success
    or exclusive ownership: inspect live capacity, ownership, and all dependent
    jobs before cleanup. A mismatch or legacy record requires reconciliation,
    not overwriting or deleting state to force another update. After relevant
    jobs are terminal, separately approve restoring prior capacity or removing
    a deployment you own; cancellation does neither.
- Keep final status, retained-file, deployment, and cost evidence private.