# Session Management

Discover and inspect existing sessions/checkpoints without launching training, then explicitly clean up **only an owned session** when needed. Read-only listing still calls Azure and requires project authorization; unload/heartbeat examples are separate management operations, not local checks.

These operations all live on `client.sessions` and `client.checkpoints` in the `azure-ai-finetuningsessions` SDK — they're the read/management half of the API surface (the train loop covered in [training.md](./training.md) drives the write half).

For an interactive walkthrough of the common path (list → inspect session → list checkpoints), see [`notebooks/manage_sessions.ipynb`](../notebooks/manage_sessions.ipynb).

> [!NOTE]
> The notebook's aggregate slices retain an older `ready` status filter. Use its raw list/get/checkpoint cells for inspection, not that filter as evidence of GPU residency or successful cleanup. The pinned SDK's declared statuses are listed below; notebook code is unchanged by this documentation pass.

> [!NOTE]
> All examples assume you've authenticated as described in [auth.md](./auth.md) and have your `project_endpoint`. The snippets use the **sync** client `azure.ai.finetuningsessions.FineTuningSessionClient`; an `aio.FineTuningSessionClient` with the same surface exists if you need async.

## Build a client

```python
import os
from azure.ai.finetuningsessions import FineTuningSessionClient
from azure.ai.finetuningsessions.models import FoundryFeaturesOptInKeys

PROJECT_ENDPOINT = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "").strip()
if not PROJECT_ENDPOINT:
    raise ValueError("Set AZURE_AI_PROJECT_ENDPOINT to your Foundry project URL")
PREVIEW = FoundryFeaturesOptInKeys.FINETUNING_SESSIONS_V1_PREVIEW
API_VERSION = "v1"

api_key = os.environ.get("AZURE_AI_API_KEY")
if api_key:
    from azure.core.credentials import AzureKeyCredential
    credential = AzureKeyCredential(api_key)
    extra = {}
else:
    from azure.identity import DefaultAzureCredential
    credential = DefaultAzureCredential(exclude_managed_identity_credential=True)
    extra = {"credential_scopes": ["https://ai.azure.com/.default"]}

client = FineTuningSessionClient(endpoint=PROJECT_ENDPOINT, credential=credential, **extra)
```

> [!TIP]
> Operation-group methods below have **required keyword-only** `foundry_features=PREVIEW, api_version=API_VERSION` arguments in SDK 1.0.0b1. Omitting them raises a Python argument error; the high-level async convenience methods have different signatures. Close the HTTP client and identity credential when finished ([client cleanup](#close-local-clients)).

## List your sessions

```python
result = client.sessions.list(
    foundry_features=PREVIEW,
    api_version=API_VERSION,
    limit=50,        # optional; default is service-decided
    offset=0,        # optional; for pagination
)

for s in result.data:
    print(s.session_id, s.status, s.base_model, "rank=", s.lora_rank, s.last_request_time)

print("page:", result.cursor.offset, "..", result.cursor.offset + len(result.data),
      "of", result.cursor.total_count)
```

Each `SessionSummary` has:

| Field | Notes |
|---|---|
| `session_id` | `session_<...>` — what you pass as `source_session_id` in [continual-fine-tuning.md](./continual-fine-tuning.md). |
| `base_model` | e.g. `"Qwen/Qwen3.8-27B"`; existing sessions retain the model they were created with. |
| `status` | Declared `SessionStatus` values: `queued`, `running`, `succeeded`, `failed`. A returned value can also be a string. Status alone is not proof of GPU residency; do not gate work on the obsolete `ready` label. |
| `is_lora`, `lora_rank` | Adapter config. |
| `corrupted` | Indicates lost live state. Check saved training checkpoints separately; do not infer that every persisted checkpoint disappeared. |
| `last_request_time` | Most recent request — useful for finding idle/forgotten sessions. |

For additional pages, advance `offset` by the number of rows actually returned. Stop when the new offset reaches `total_count` **or the page is empty**; using the requested `limit` can skip rows if the service returns fewer than requested.

## Inspect a single session

```python
s = client.sessions.get(
    session_id="session_<your-recorded-run-id>",
    foundry_features=PREVIEW,
    api_version=API_VERSION,
)
print(s.session_id, s.status, s.type)
print(s.model_data.base_model, s.model_data.model_name, s.model_data.lora_config)
```

Returns a full `Session` (status + nested `model_data`). Use the recorded ID to inspect a failed run or confirm model/LoRA settings. A saved training checkpoint can initialize a **new** session even after its source is no longer live; continual training does not require the old session to be `running` ([checkpoint restore](./continual-fine-tuning.md)).

## List checkpoints for a session

```python
ckpts = client.checkpoints.list(
    session_id="session_<your-recorded-run-id>",
    foundry_features=PREVIEW,
    api_version=API_VERSION,
)
for c in ckpts.checkpoints:
    print(c.checkpoint_id, c.checkpoint_type, c.time)
```

`checkpoint_type` distinguishes `"training"` (full state, for `load_checkpoint_path`) from `"sampler"` (persisted sampler-format weights without optimizer state). The separate per-step ephemeral sync is not a persisted sampler save. See [SDK save semantics](./sdk-reference.md#persistent-versus-ephemeral-checkpoints).

> [!TIP]
> If the local ledger is lost, you cannot recover its exact dataset cursor. A retained compatible **training** checkpoint can still initialize a fresh run: pass `load_checkpoint_path=<session_id>/<checkpoint_id>` and use a new `log_path` ([continual training](./continual-fine-tuning.md)). Discovery does not guarantee indefinite checkpoint retention or access from another project.

## Get checkpoint metadata

```python
info = client.checkpoints.get(
    session_id="session_<your-recorded-run-id>",
    checkpoint_id="final",
    foundry_features=PREVIEW,
    api_version=API_VERSION,
)
print(info.base_model, "is_lora=", info.is_lora, "rank=", info.lora_rank)
```

## Free GPU memory: `begin_unload`

Normal Azure RL/SFT/NER workflows **attempt proactive session cleanup** in `finally`. The shared RL path calls the async convenience `await client.close_session(session_id)`, which stops its SDK heartbeat and submits the session-completion request. Closing an HTTP client alone does not perform this remote cleanup. Failures can leave resources active, especially when setup fails before entering the protected loop.

Use the sync operation-group unload below for an explicitly selected, owned leftover session. First inspect its checkpoints and complete any required saves using the **live training client**. Do not execute this against an example ID or bulk-clean other users' sessions.

```python
poller = client.sessions.begin_unload(
    session_id="session_<your-recorded-run-id>",
    foundry_features=PREVIEW,
    api_version=API_VERSION,
)
result = poller.result()   # blocks until the LRO completes
print(result)
```

> [!WARNING]
> Unload can discard **unsaved in-memory weights/optimizer state**. On a live raw async client, await both `await client.save_weights_async(session_id, name)` and its returned task; on the cookbook adapter, await `save_state_async(name)` and `result_async()`. A sampler-only save is not training recovery. Wait for completed save evidence **before** unloading. Resume through a new compatible session/`FromCheckpoint`, not by assuming the old session can be reattached. Do not substitute `delete_session`: deletion can cascade to checkpoints.

## Heartbeat & idle expiry (FYI — you usually don't call this)

High-level session helpers manage background heartbeats while the client process is alive. Idle expiry is a service policy, not a documented budget/cleanup guarantee; do not depend on an assumed fixed timeout. The manual call below is **not read-only** and can extend activity; use it only for an owned session when needed for diagnostics:

```python
hb = client.sessions.heartbeat(
    session_id="session_<your-recorded-run-id>",
    foundry_features=PREVIEW,
    api_version=API_VERSION,
)
```

A `404` can mean a missing resource, wrong endpoint/project/session ID, or expired/deleted state. Check the request context and list/get response before diagnosing it; it does not by itself prove successful cleanup or checkpoint deletion.

## Putting it together: inspect cleanup candidates

List all pages and filter locally to **your recorded run IDs**. This example only proposes inspection candidates; it deliberately performs no unload and does not treat age/status as permission to remove resources:

```python
owned_session_ids = {"session_<your-recorded-run-id>"}  # Replace deliberately.
candidates = []
offset = 0
while True:
    page = client.sessions.list(
        foundry_features=PREVIEW, api_version=API_VERSION,
        limit=100, offset=offset,
    )
    candidates.extend(
        s for s in page.data if s.session_id in owned_session_ids
    )
    offset += len(page.data)
    if offset >= page.cursor.total_count or not page.data:
        break

for s in candidates:
    print("Inspect before deciding:", s.session_id, s.status, s.last_request_time)
```

For each candidate, confirm its owner/job is finished and required training saves completed, then deliberately target that ID with the unload example. Re-inspect it and retain any management error/request ID; successful local process exit or a cleanup log message alone is not independent confirmation of release. Local log deletion does not unload remote compute.

## End-of-run checklist

1. Record the intended owned session ID and original model/rank in private run
    metadata. Include any separately created reference/teacher sessions or
    external tools; do not infer ownership from an old timestamp.
2. Confirm required training saves completed and retain the full local ledger,
    config, and dataset manifest. Sampler-only state cannot restore optimizer
    state or the data cursor.
3. Check the launcher's cleanup result. A timeout, Ctrl+C, closed HTTP client,
    stopped dashboard, or missing metrics does not prove the service stopped.
4. Inspect the owned session using list/get and the management-operation result.
    If its release is uncertain, resolve that state with the administrator; do
    not treat a status label or `404` alone as a billing confirmation.
5. Within the requested run's cleanup plan, unload only the identified leftover
    session after saving needed state. Separate deletion requires separate
    confirmation because it can remove checkpoints. Never sweep a shared project.
6. Close local clients/credentials and approved sidecars. Separately provisioned
    serving deployments, VMs, tool sandboxes, and storage are not removed by
    session cleanup; report the remaining owners/resources and cost uncertainty.
7. Archive private evidence under the [retention policy](./storage.md#archive-and-deletion-decisions).
    Do not claim zero ongoing cost without the applicable resource and billing
    evidence; this SDK workflow is not a billing audit.

## Close local clients

After the management examples, release local HTTP/token resources even if a call failed (use `try/finally` in a script):

```python
try:
    client.close()
finally:
    if not api_key:
        credential.close()
```

This closes local clients, **not** Azure sessions. For a complete async lifecycle with checkpoint restore and remote cleanup, see [evaluation and inference](./evaluation-and-inference.md#use-a-training-checkpoint-after-the-original-session-closes).
