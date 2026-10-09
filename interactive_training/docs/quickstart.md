# Quickstart

Install the cookbook, perform local checks, run a bounded GSM8K workflow on Azure, and verify its artifacts and cleanup. This is an infrastructure check, **not a quality benchmark**. For SFT or other tasks, use the [recipe chooser](./recipes.md#choose-a-path).

## Before you start

| Requirement | Check |
|---|---|
| Python **3.11 or newer**, Git, and a writable local directory | `python --version`; Windows/WSL use separate virtual environments |
| Network access for packages, public tokenizers, and datasets | PyPI, the CPU Torch index on Linux/Windows, and Hugging Face must be reachable |
| Azure AI Foundry **project endpoint** in a supported region | [Resource and project setup](./auth.md#prerequisite--an-azure-ai-foundry-resource) |
| Credentials authorized for this project | An API key, or an identity with the required project access; successful `az login` alone is not proof of authorization |
| Eligibility and capacity for the selected model | [Supported models](./supported_models.md); availability depends on project region, capacity, and quota |

> [!WARNING]
> Training, evaluation, sampling, and model loading use paid cloud resources.
> When a user requests this workflow, warn once and proceed without a separate
> budget or spend-approval prompt. Include cost information if readily available;
> unknown pricing is not a blocker.

No local GPU or local model-weight download is required for this text recipe. Keep the Python driver and its network connection alive until it finishes. Do not place sensitive data in a public run directory.

## 1. Open the cookbook directory

For a new checkout:

```bash
git clone https://github.com/microsoft-foundry/fine-tuning.git
cd fine-tuning/interactive_training
```

Already have the repo? Open **fine-tuning/interactive_training**, not the parent fine-tuning directory. It contains `pyproject.toml`, `dashboard_server.py`, and the inner Python package `interactive_training/`. The outer cookbook and inner Python package now both use `interactive_training`; use the renamed `python -m interactive_training...` commands below.

## 2. Create and activate a virtual environment

Choose the commands for the shell and operating system you are actually using.

**Linux / macOS / WSL (Bash):**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Native Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Git Bash on native Windows:** after `python -m venv .venv`, use `source .venv/Scripts/activate`. **cmd.exe:** use `.venv\Scripts\activate.bat`.

If PowerShell policy blocks activation, do not weaken system-wide policy. Activation is optional: use `.\.venv\Scripts\python.exe` in place of `python` in the remaining commands. Re-activate the same environment in each new terminal. Do not reuse a Windows environment from WSL or vice versa.

## 3. Install and check locally

On Linux/Windows, with the environment active:

```bash
python -m pip install -e . --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
```

On macOS, omit the extra index and use PyPI's platform Torch build. The extra index offers CPU Torch wheels for driver-side utilities; pip searches both indexes, so check the selected build if local GPU dependencies are unexpectedly large. See [Torch troubleshooting](./troubleshooting.md#torch-install-picks-the-wrong-wheel).

The install brings in the pinned SDK, **`azure-ai-finetuningsessions==1.0.0b1`**. No SDK source checkout or bundled fine-tuning wheel is needed. Existing preview users should follow the one-time [migration instructions](../README.md#install).

These checks do **not** create a remote session or train a model:

```bash
python -m pip check
python -c "import sys, importlib.metadata as m; import interactive_training as c; import azure.ai.finetuningsessions as s; print(sys.executable); print(c.__file__); print(m.version('azure-ai-finetuningsessions')); print(s.__file__)"
python -m interactive_training.recipes.math_rl.train_azure --help
```

Expected: no broken dependencies, the cookbook import points at this checkout, SDK version `1.0.0b1`, and help lists `max_steps`, `max_train_examples`, and `project_endpoint`. Import location/version checks do not by themselves prove package-download provenance.

**Optional tokenizer/renderer check:** this may download public tokenizer files from Hugging Face, but does not call Azure:

```bash
python -c "from interactive_training.tokenizer_utils import get_tokenizer; from interactive_training.renderers import get_renderer; t=get_tokenizer('Qwen/Qwen3.8-27B'); r=get_renderer('qwen3_8_low_reasoning', tokenizer=t); p=r.build_generation_prompt([{'role':'user','content':'What is 2 + 2?'}]); print('Prompt rendered:', len(p.chunks), 'chunks')"
```

> [!IMPORTANT]
> The math launcher has **no `preflight_only` or `preflight_dataset` CLI option**. Running it with a dummy endpoint or `max_steps=0` is not an offline preflight. The [Tulu3 SFT launcher](../interactive_training/recipes/tulu3_sft/README.md) does have `preflight_only=true`; that validates data without creating a session, but can download data/tokenizers and write local metadata.

## 4. Select the project and model

Reuse or create a Foundry resource in a [supported region](./supported_models.md#available-regions), then select its **project**. Copy the project endpoint from the project details; it has this shape:

```text
https://<account>.services.ai.azure.com/api/projects/<project-name>
```

Use the project URL, not an ARM resource ID, a model deployment URL, or only the account hostname. Portal labels vary; see [authentication](./auth.md). Project authorization is still required; a supported model and region do not guarantee capacity. Do not grant yourself broad roles or launch a paid session to test access. See the [access checklist](./auth.md#confirm-access-before-spending) if access is unclear.

Set the non-secret endpoint for this walkthrough:

**Bash:**

```bash
export AZURE_AI_PROJECT_ENDPOINT="<your-project-endpoint>"
```

**PowerShell:**

```powershell
$env:AZURE_AI_PROJECT_ENDPOINT = "<your-project-endpoint>"
```

The command below passes this value explicitly as `project_endpoint`. Setting the environment variable alone does not configure the math CLI.

## 5. Authenticate

For local identity-based access:

```bash
az login
az account set --subscription "<your-subscription>"
```

If `AZURE_AI_API_KEY` is set, the math recipe uses that key instead. Otherwise it uses `DefaultAzureCredential`, including managed identity and other credentials before the CLI fallback. Check the credential actually selected. Supply keys through an approved secret mechanism; do not put them in recipe arguments, source, notebook outputs, or shared logs. See [auth.md](./auth.md) for both modes and [troubleshooting](./troubleshooting.md#auth-fails-a-few-seconds-in).

## 6. Run the bounded remote workflow

> [!WARNING]
> **This step creates a remote training session and may incur charges.** `max_steps=2` bounds training iterations, not model-loading time, evaluation calls, total generated tokens, or price. Respect any supplied spending limits; this guide does not require a dollar budget or a separate cost acknowledgement.

The general text default is Qwen3.8 with low-effort thinking. Specify `max_tokens=1200`: the math launcher's five-token default is for its toy arithmetic configuration, not GSM8K reasoning. Reasoning and the final answer share the completion budget.

**Bash (Linux / macOS / WSL / Git Bash):**

```bash
python -m interactive_training.recipes.math_rl.train_azure \
    project_endpoint="$AZURE_AI_PROJECT_ENDPOINT" \
    model_name="Qwen/Qwen3.8-27B" renderer_name=qwen3_8_low_reasoning \
    env=gsm8k learning_rate=2e-5 temperature=1.0 max_tokens=1200 lora_rank=32 \
    group_size=4 groups_per_batch=8 loss_fn=importance_sampling seed=42 \
    max_steps=2 max_train_examples=16 max_test_examples=8 \
    eval_every=1 save_every=1 \
    log_path=./runs/first-math-check behavior_if_log_dir_exists=raise
```

**PowerShell (single line avoids Bash continuation syntax):**

```powershell
python -m interactive_training.recipes.math_rl.train_azure project_endpoint="$env:AZURE_AI_PROJECT_ENDPOINT" model_name="Qwen/Qwen3.8-27B" renderer_name=qwen3_8_low_reasoning env=gsm8k learning_rate=2e-5 temperature=1.0 max_tokens=1200 lora_rank=32 group_size=4 groups_per_batch=8 loss_fn=importance_sampling seed=42 max_steps=2 max_train_examples=16 max_test_examples=8 eval_every=1 save_every=1 log_path=./runs/first-math-check behavior_if_log_dir_exists=raise
```

This uses eight prompts with four completions per training iteration; validation uses a separate eight-example subset. The recipe loads public GSM8K data. Completion budgets can still truncate responses; low reasoning does not guarantee an answer fits.

The fixed run directory makes artifacts easy to find. `raise` deliberately refuses an existing directory: choose a new name for another independent run, or follow [recovery](./training.md#resuming-after-an-interruption) with the original config and `behavior_if_log_dir_exists=resume`. Do not use `delete` to resolve a recovery error.

No first-run duration or total price is promised. Before launching, establish
the requested run's observation/stop plan and distinguish a session-creation
timeout from cancellation. The CLI has a creation timeout and optional
`max_wall_clock_seconds`, but a loop-boundary time limit is not a hard request,
save, or billing cap. Inspect the last phase and owned session before retrying
a slow operation ([stall diagnosis](./troubleshooting.md#session-created-but-no-progress)).

CLI arguments here are `key=value`, not `--key value`. Dashboard and some visual-spatial sampling commands use a different parser; see [CLI conventions](./README.md#cli-conventions). Do not detach training with PowerShell `Start-Process`; keep errors and stdout visible.

## 7. Check what succeeded

Look in **`./runs/first-math-check/`**, relative to the cookbook directory:

| Stage | Evidence | What it proves |
|---|---|---|
| Launcher setup | `run_meta.json` is written; session ID may initially be `null` | Local setup started, **not** that Azure admitted the session |
| Model loaded | Console `Session ready: session_id=...`; `azure.session_id` updated | Session creation completed |
| Loop running | `config.json`, `metrics.jsonl`, training metrics, and trajectory HTML | Sampling, grading, and update operations are progressing |
| Saved state | `checkpoints.jsonl` has saved `state_path`/`sampler_path` entries | Save operations completed; do not rely on a checkpoint name printed before the save finishes |
| Normal completion | `final` ledger row after trained batches, final held-out metrics, and successful process exit | Bounded workflow completed; inspect warnings and verify cleanup separately |

For this fresh synchronous GSM8K run, the initial evaluation occurs before the first update; the final evaluation follows the final save. Check `test/env/all/correct` and `test/env/all/total_episodes`; inspect any dropped-evaluation warnings. A zero score can be a real failure of formatting or token budget even when infrastructure works. Missing metrics are not a zero score.

There is **no required accuracy gain or promised runtime** for this two-step check. To claim improvement, use a fixed held-out set and the [evaluation guide](./evaluation-and-inference.md).

View the local artifacts:

```bash
python dashboard_server.py --root ./runs
# Open http://127.0.0.1:8000/
```

The dashboard has no authentication and may display prompts/completions. Keep it on loopback and use private forwarding; see [dashboard security](./dashboard.md#security). Stop the dashboard with Ctrl+C; that does not unload Azure sessions.

## 8. Verify cleanup and keep recovery artifacts

The shared Azure RL loop **attempts** `close_session` in its `finally` block, and the launcher closes the HTTP client and identity credential. Cleanup failures are logged, so a printed "Session closed" message alone is not confirmation that remote compute was released. If setup fails before entering that loop, automatic cleanup may not reach the session.

Use the recorded session ID with the [read-only management guide](./session-management.md#inspect-a-single-session) or [management notebook](../notebooks/manage_sessions.ipynb). If it remains active, first confirm the required checkpoint saves completed, then explicitly unload **that owned session**. Do not bulk-unload other project users' sessions.

Keep the config, metrics, checkpoint ledger, and source session ID for reproducibility/recovery. Deleting local logs does not delete remote checkpoints or release compute. Do not assume a checkpoint is retained indefinitely; verify service retention requirements before depending on it.

Before handing off, complete the [end-of-run checklist](./session-management.md#end-of-run-checklist)
and record any remaining session, external tool/VM, or separately provisioned
serving resources privately. Training cleanup does not delete serving resources
or establish that ongoing cost is zero.

## Next steps

- [Concepts](./concepts.md): sessions, runs, renderers, data, rewards, and updates.
- [Custom data](./custom-data.md): use your conversations or implement your own task grader.
- [Evaluation and inference](./evaluation-and-inference.md): compare before/after and sample saved weights.
- [Training](./training.md): budgets, cadence, and recovery.
- [Troubleshooting](./troubleshooting.md): find the stage that failed before retrying a paid launch.
