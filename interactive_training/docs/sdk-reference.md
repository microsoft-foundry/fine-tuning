# SDK Reference

A focused reference for the slice of the `azure-ai-finetuningsessions` SDK the cookbook exercises, plus the cookbook wrappers that bridge from raw HTTP-style calls to the train loop in [training.md](./training.md).

This is **not** the full SDK reference. Examples target the pinned public preview, `azure-ai-finetuningsessions==1.0.0b1`; see the [published package and its documentation links](https://pypi.org/project/azure-ai-finetuningsessions/). For rendered custom data, start with [custom-data.md](./custom-data.md); for checkpoint use after training, see [evaluation and inference](./evaluation-and-inference.md).

For the **management** half of the SDK (`client.sessions.list / get / begin_unload / heartbeat`, `client.checkpoints.list / get`) — which the recipe does not exercise but you'll want for discovery, inspection, and cleanup — see [session-management.md](./session-management.md) and the companion [`notebooks/manage_sessions.ipynb`](../notebooks/manage_sessions.ipynb).

> [!NOTE]
> Sync examples import `azure.ai.finetuningsessions.FineTuningSessionClient`; async examples import `azure.ai.finetuningsessions.aio.FineTuningSessionClient`. DTOs come from `azure.ai.finetuningsessions.models`. These clients and the cookbook wrappers are different interfaces, not interchangeable handles.

## Where the boundary is

The SDK gives you a **session** and a **client** for posting requests to that session. The cookbook layers a small adapter (`AzureSDKTrainingClient`) on top to turn raw HTTP calls into awaitable `forward_backward_async` / `optim_step_async` / `save_*` methods with futures.

```
┌── your recipe (interactive_training/recipes/math_rl/train_azure.py) ──┐
│   parses CLI args, picks model + dataset,                       │
│   awaits client.create_session(...), then drives:              │
│                                                                 │
│   training_client.forward_backward_async(...)  ◄── cookbook ────┼─── AzureSDKTrainingClient
│   training_client.optim_step_async(...)                         │    (interactive_training/rl/train_azure.py)
│   training_client.save_state_async(...)                         │
│                                                                 │
└──────────────────────┬──────────────────────────────────────────┘
                       ▼
┌── azure-ai-finetuningsessions SDK ─────────────────────────────┐
│   FineTuningSessionClient  ── HTTP client                       │
│   FineTuningSession        ── handle to a live session          │
│   FromCheckpoint, LoRAConfig, AdamParams, LossFn  ── DTOs       │
└─────────────────────────────────────────────────────────────────┘
```

## SDK surface

### `FineTuningSessionClient`

The HTTP client. One per process is usually enough.

```python
from azure.ai.finetuningsessions import FineTuningSessionClient

client = FineTuningSessionClient(
    endpoint="https://<project>.services.ai.azure.com/api/projects/<name>",
    credential=credential,                          # AzureKeyCredential or DefaultAzureCredential
    credential_scopes=["https://ai.azure.com/.default"],  # only with DefaultAzureCredential
)
```

| Arg | Type | Notes |
|---|---|---|
| `endpoint` | `str` | Project endpoint URL. Comes from your Azure AI project. |
| `credential` | `AzureKeyCredential \| TokenCredential` | See [auth.md](./auth.md). |
| `credential_scopes` | `list[str]` | Required when using `DefaultAzureCredential`; omit when using `AzureKeyCredential`. |

### `FineTuningSession.create`

Synchronous classmethod that provisions a session on the service. **Blocks** while the model loads (up to `timeout_sec`). The async cookbook path instead awaits `client.create_session(...)`, as shown in [Putting it together](#putting-it-together).

```python
from azure.ai.finetuningsessions import FineTuningSession
from azure.ai.finetuningsessions.models import LoRAConfig, FromCheckpoint

session: FineTuningSession = FineTuningSession.create(
    client,
    base_model="Qwen/Qwen3.8-27B",
    lora_config=LoRAConfig(rank=32),
    type="training",
    from_checkpoint=None,                # or FromCheckpoint(source_session_id=..., checkpoint_id=...)
    timeout_sec=600.0,
)

session.session_id   # → "session_<...>" — record this in run_meta.json
```

| Arg | Type | Notes |
|---|---|---|
| `base_model` | `str` | HF-style identifier of the base model (e.g. `"Qwen/Qwen3.8-27B"`). |
| `lora_config` | `LoRAConfig` | LoRA adapter configuration. Pass `LoRAConfig(rank=N)`. |
| `type` | `Literal["training"]` | Currently only `"training"` is supported; there is no separate inference session type. A session's sampler is used for both rollouts and evaluation. |
| `from_checkpoint` | `FromCheckpoint \| None` | When non-`None`, the new session's LoRA weights are bootstrapped from the referenced checkpoint. See [continual-fine-tuning.md](./continual-fine-tuning.md). |
| `timeout_sec` | `float` | How long to wait for the model to load before raising. |

### `LoRAConfig`

```python
from azure.ai.finetuningsessions.models import LoRAConfig

LoRAConfig(rank=32)
```

| Field | Type | Notes |
|---|---|---|
| `rank` | `int` | LoRA adapter rank, typically 2–32. See `lora_rank` in [training.md](./training.md#tunable-parameters). |

### `FromCheckpoint`

```python
from azure.ai.finetuningsessions.models import FromCheckpoint

FromCheckpoint(
    source_session_id="session_3999c86d",
    checkpoint_id="final",
)
```

| Field | Type | Notes |
|---|---|---|
| `source_session_id` | `str` | The `session_<...>` identifier of the previous session. |
| `checkpoint_id` | `str` | The `name` field from a row in that session's `checkpoints.jsonl` (e.g. `"final"`, `"4"`). |

### `AdamParams`

```python
from azure.ai.finetuningsessions.models import AdamParams

AdamParams(learning_rate=2e-5, beta1=0.9, beta2=0.95, eps=1e-8)
```

Passed to `optim_step`. The shared loops construct these parameters from the effective learning rate and optimizer settings; they do not universally reuse one object for the whole run. RL uses the recipe's learning rate for its substeps; SFT computes the scheduled rate for each step.

### `LossFn`

Enum of valid `loss_fn` strings. See [training.md](./training.md#loss-functions) for what each one means.

```python
from azure.ai.finetuningsessions.models import LossFn

LossFn.IMPORTANCE_SAMPLING   # "importance_sampling"
LossFn.PPO                   # "ppo"
# ... cispo, sapo, cross_entropy
```

You can pass either the enum or the string to `forward_backward`.

### `Datum`, `ModelInput`, `TensorData`

```python
from azure.ai.finetuningsessions.models import Datum, ModelInput, ModelInputChunk, TensorData

datum = Datum(
    model_input=ModelInput(chunks=[ModelInputChunk(tokens=[101, 2023])]),
    loss_fn_inputs={
        "target_tokens": TensorData(data=[2023, 102]),
        "weights": TensorData(data=[1.0, 1.0]),  # cross-entropy example
    },
)
```

These token IDs only illustrate the shape. Build real inputs with the selected
model's tokenizer/renderer; arbitrary token IDs are not meaningful training data.

The wire-format atom passed to `forward_backward`. `target_tokens` is required; the rest of `loss_fn_inputs` depends on which `loss_fn` you intend to use. `TensorData.data` is `list[int] | list[float]`; integer for `target_tokens`, float for everything else.

| `loss_fn` | Required `loss_fn_inputs` keys | Set by |
|-----------|--------------------------------|--------|
| `cross_entropy` (SFT) | `target_tokens`, `weights` | The renderer; see [training.md § SFT](./training.md#sft-per-token-weights). |
| `importance_sampling`, `ppo`, `cispo`, `sapo` (RL) | `target_tokens`, `advantages`, `logprobs` | The RL pipeline in [`interactive_training/rl/`](../interactive_training/rl/); see [training.md § RL](./training.md#rl-implicit-masking-via-advantages). |

In the RL case there is no separate `weights` field on the wire — per-token masking is folded into `advantages` (zero at observation positions, `traj_advantage` at sampled-action positions). See the training-page sections above for how this affects multi-turn and tool-calling recipes.

## Cookbook wrappers (`AzureSDKTrainingClient`)

Defined in [`interactive_training/rl/train_azure.py`](../interactive_training/rl/train_azure.py).
Wraps an **async SDK client**, a session ID, and a tokenizer. It does not accept
the synchronous `FineTuningSession` handle as its first argument.

```python
from interactive_training.rl.train_azure import AzureSDKTrainingClient

# client is azure.ai.finetuningsessions.aio.FineTuningSessionClient
# session_id is returned by await client.create_session(...)
training_client = AzureSDKTrainingClient(client, session_id, tokenizer)
```

### `forward_backward_async(batch, loss_fn=None, loss_fn_config=None) → Future`

Enqueue a forward+backward pass over `batch` (a `list[Datum]`). Returns a future you `await future.result_async()` for the per-datum loss outputs.

- Maps to: `POST /fine_tuning/sessions/<id>/forward_backward`
- `loss_fn`: one of the `LossFn` strings; defaults to `"importance_sampling"`.
- `loss_fn_config`: optional `dict[str, Any]` of loss-specific knobs (e.g. PPO clip ratio).

### `forward_async(batch, loss_fn=None, loss_fn_config=None) → Future`

Run a read-only forward pass for evaluation. The result contains per-datum
`loss_fn_outputs`; `total_loss` is `None` because forward-only operations do not
report the aggregate training objective. SFT recipes use the built-in
`NLLEvaluator` to compute weighted mean NLL from these outputs.

### `forward_backward_custom_async(data, loss_fn) → Future`

Research escape hatch: forward-backward with a Python-side custom loss. Uses a surrogate-gradient trick — one server-side forward to obtain logprobs, local autograd on the user's loss to get `g = ∂C/∂logprobs`, then a server-side `forward_backward` with `loss_fn="cross_entropy"` and `weights = -g` so that `∂L_server/∂θ = ∂C/∂θ` exactly.

- `data: list[Datum]` — each datum's `loss_fn_inputs` must contain `target_tokens` and may optionally contain `weights`. Additional client-side auxiliary keys are available to the custom loss but stripped from the wire payload. If `weights` is omitted, the adapter injects zeros for the forward call.
- `loss_fn: Callable[[list[Datum], list[torch.Tensor]], tuple[torch.Tensor, dict[str, float]]]` — receives the original data and a list of leaf logprob tensors (`requires_grad=True`); returns `(scalar_loss, custom_metrics)`. **Must produce a gradient for every datum's logprobs** — a `None` gradient raises `ValueError`. Custom metrics merge into `result.metrics` alongside backend metrics.
- Cost: roughly 1.5× under idealized forward/backward FLOP arithmetic (one extra forward pass), not a measured runtime or price guarantee. Prefer the built-in losses when they suffice.
- Helpers: [`interactive_training.tensor_utils`](../interactive_training/tensor_utils.py) provides `tensor_data_to_torch` / `tensor_data_from_torch` for the common 1-D round-trip.

See [loss_functions.md § Writing a custom loss](./loss_functions.md#writing-a-custom-loss) for the math derivation and the full constraint list (including the 1-D `TensorData` limitation).

### `optim_step_async(adam_params=None) → Future`

Apply accumulated gradients with the given Adam params.

- Maps to: `POST /fine_tuning/sessions/<id>/optim_step`
- Defaults to `AdamParams(learning_rate=3e-4, beta1=0.9, beta2=0.95, eps=1e-8)` if `None`.

### `save_state_async(name, ttl_seconds=None, *, step_number=None, metrics=None) → Future`

Save full training state (LoRA weights + optimizer state) under the given checkpoint name.

- Maps to: `POST /fine_tuning/sessions/<id>/checkpoint`
- Calls raw `await client.save_weights_async(session_id, name, step_number=..., metrics=...)`; await the wrapper future's `result_async()` before using its `.path`.
- `ttl_seconds` is currently accepted for compatibility but not forwarded to the SDK.

### `save_weights_for_sampler_async(name, ttl_seconds=None) → Future`

**Persist** a sampler-format checkpoint to remote storage (without optimizer state). Used by `save_checkpoint_async(kind="sampler" | "both")` at durable checkpoint boundaries. This is **not** the per-step ephemeral refresh.

- Maps to: `POST /fine_tuning/sessions/<id>/checkpoint_sample`
- Calls raw `await client.save_weights_for_sampler_async(session_id, name)`; await `result_async()` to obtain `.checkpoint_id` / `.path`.
- `ttl_seconds` is not forwarded. Do not infer retention or a cost/memory advantage from the method name.

### `save_weights_and_get_sampling_client_async(name=None) → AzureSDKSamplingClient`

**Ephemeral** weight sync: calls the raw SDK's method of the same name with the session ID and `name` (default `step<N>`), awaits its result, then returns an `AzureSDKSamplingClient`. It does not call the persistent method above. The loop uses it between durable saves to refresh rollout weights without blob persistence.

### `create_sampling_client(model_path) → AzureSDKSamplingClient`

Construct a local wrapper bound to the adapter's **current session** and the supplied sampler checkpoint identifier. This does not create a remote session, load training state, or make a different session's checkpoint available. Keep a sampler ID together with its owning session ID. To restart after closing the source session, use a saved **training** checkpoint to create a new compatible session, then sync its sampler ([worked helper](./evaluation-and-inference.md#use-a-training-checkpoint-after-the-original-session-closes)).

### `get_tokenizer() → Tokenizer`

Return the tokenizer instance the training client was constructed with. Convenience accessor used by reward / advantage computation.

### `close_poller()`

Compatibility no-op. It does not close the session or HTTP client. Call
`await client.close_session(session_id)` and close the async client when finished.

## `AzureSDKSamplingClient`

Returned by `save_weights_and_get_sampling_client_async` / `create_sampling_client`. The relevant method:

### `sample_async(prompt, num_samples=1, sampling_params=None) → _SampleResult`

Sample `num_samples` completions for one `ModelInput` prompt. Await the method;
the returned wrapper exposes `.sequences`. There is no synchronous `sample`
method on the cookbook sampling wrapper.

- `sampling_params` is a struct with `temperature`, `max_tokens`, `top_p`, etc.

For batched group sampling, use `asyncio.gather` over one `sample_async` call per prompt.

## Persistent versus ephemeral checkpoints

| Purpose | Raw async SDK | Cookbook adapter |
|---|---|---|
| Save resumable weights + optimizer state | `save_weights_async(session_id, name)` | `save_state_async(name)` |
| Persist sampler-format weights | `save_weights_for_sampler_async(session_id, name)` | `save_weights_for_sampler_async(name)` |
| Refresh sampler in memory | `save_weights_and_get_sampling_client_async(session_id, name)` | `save_weights_and_get_sampling_client_async(name=None)` |

The raw `*_async` save methods return an `asyncio.Task` **after submitting the request**. Completion is a second await. They do not return a cookbook sampling client, even when the name includes `get_sampling_client`:

```python
# Inside an async function; client is the raw aio SDK and session_id is live.
task = await client.save_weights_for_sampler_async(session_id, "step100")
saved = await task
checkpoint_id = saved.checkpoint_id
```

The adapter's persistent saves return futures with `result_async()`:

```python
# Inside an async function; trainer is AzureSDKTrainingClient.
future = await trainer.save_weights_for_sampler_async("step100")
saved = await future.result_async()
checkpoint_id = saved.checkpoint_id
```

At the operation-group boundary, `SaveSamplerWeightsRequest(path=name, seq_id=0)` **without** `sampling_session_seq_id` requests persistence. Providing `sampling_session_seq_id` requests an ephemeral update; the convenience SDK method manages that counter. Operation-group calls also require `foundry_features` and `api_version` ([session management](./session-management.md)).

Await all required saves before cleanup. A durable save does not establish indefinite retention, cross-session sampler-ID portability, or a deployed inference endpoint. The cookbook's compatibility `ttl_seconds` is not enforced by the pinned adapter. See [storage](./storage.md#local-artifacts-versus-remote-state) and [sampling examples](./evaluation-and-inference.md).

## Putting it together

A helper that performs one SFT update on an already-rendered batch and saves a
checkpoint. Set `AZURE_AI_PROJECT_ENDPOINT`, authenticate with `az login`, and
pass real `Datum` values built for Qwen3.8. **Calling this helper allocates a remote
training session and may incur charges.** It is not a complete dataset recipe.

```python
import os
from azure.ai.finetuningsessions.aio import FineTuningSessionClient
from azure.ai.finetuningsessions.models import AdamParams, Datum, LoRAConfig
from azure.identity.aio import DefaultAzureCredential
from interactive_training.rl.train_azure import AzureSDKTrainingClient
from interactive_training.tokenizer_utils import get_tokenizer

async def train_one_batch(batch: list[Datum]) -> str:
    if not batch:
        raise ValueError("Provide at least one rendered training datum")
    endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "").strip()
    if not endpoint:
        raise ValueError("Set AZURE_AI_PROJECT_ENDPOINT to your Foundry project URL")
    tokenizer = get_tokenizer("Qwen/Qwen3.8-27B")
    async with DefaultAzureCredential() as credential:
        async with FineTuningSessionClient(
            endpoint=endpoint,
            credential=credential,
            credential_scopes=["https://ai.azure.com/.default"],
        ) as client:
            session_id = await client.create_session(
                base_model="Qwen/Qwen3.8-27B",
                lora_config=LoRAConfig(rank=32),
                type="training",
            )
            try:
                trainer = AzureSDKTrainingClient(client, session_id, tokenizer)
                forward = await trainer.forward_backward_async(batch, loss_fn="cross_entropy")
                await forward.result_async()
                optimizer = await trainer.optim_step_async(AdamParams(learning_rate=2e-5))
                await optimizer.result_async()
                checkpoint = await trainer.save_state_async("example-final")
                return (await checkpoint.result_async()).path
            finally:
                await client.close_session(session_id)
```

See [`interactive_training/recipes/math_rl/train_azure.py`](../interactive_training/recipes/math_rl/train_azure.py) for the real recipe and [`interactive_training/rl/train_azure.py`](../interactive_training/rl/train_azure.py) for the generic loop that pipelines these calls.
