# Troubleshooting

Find the stage that failed before retrying a paid launch. Preserve the config, last successful checkpoint row, session ID, and relevant error/request IDs; remove secrets and private prompt text before sharing diagnostics.

## Start with the symptom

| Symptom | First check | Next action |
|---|---|---|
| Import/dependency error | Active interpreter, cookbook import path, SDK version, `python -m pip check` | [Editable install](#editable-install-changes-dont-take-effect) / [pinned SDK](#stale-sdk-the-installed-wheel-lags-the-cookbook); do not launch Azure to test an import |
| Unknown CLI argument or no visible logs | The selected launcher's `--help` and shell syntax | [CLI conventions](./README.md#cli-conventions); do not copy programmatic fields into CLI flags |
| Activation blocked on Windows | Shell and selected interpreter | Use the environment's Python directly ([quickstart](./quickstart.md#2-create-and-activate-a-virtual-environment)); do not weaken system-wide policy |
| Tokenizer/dataset cannot download | Network/proxy, model identifier, cache permissions, dataset terms | [Downloads](#tokenizer-or-dataset-download-fails); this is not a GPU-capacity error |
| 401 / 403 during creation | Actual selected credential, project endpoint, preview authorization | [Auth](#auth-fails-a-few-seconds-in); successful CLI login alone is insufficient |
| Queue/capacity or model-load failure | Correct project/model/region and admission; server's error detail | Confirm capacity with the administrator; more client-side retries do not create capacity |
| Session exists but no progress | Last completed phase, service request state, client/network health | [Slow/stalled operations](#session-created-but-no-progress); do not start duplicate jobs blindly |
| Truncated/invalid answers or no quality gain | Completion cap, renderer, held-out coverage, grader, actual trajectories | [Results](#answers-truncate-or-quality-does-not-improve); two steps need not improve quality |
| No SFT updates | Number of **full** batches after splitting | [Custom-data batching](./custom-data.md#2-render-and-inspect-locally); `max_steps=0` does not cap SFT |
| Run directory already exists | Whether this is a fresh comparison or recovery | New name for fresh run; original config plus `behavior_if_log_dir_exists=resume` for [recovery](./training.md#resuming-after-an-interruption); never delete recovery evidence |
| Cleanup error / leftover session | Owned session ID and completed saves | [Targeted management](./session-management.md#free-gpu-memory-begin_unload); no bulk unload |

## `torch` install picks the wrong wheel

The [main install](../README.md#install) offers CPU Torch through `--extra-index-url https://download.pytorch.org/whl/cpu`. Pip searches **both** indexes; an extra index does not guarantee which compatible build it selects. Driver-side tensor/data utilities do not require local CUDA for the text workflow; training runs on Azure. Inspect `python -m pip show torch` and `python -c "import torch; print(torch.__version__, torch.version.cuda)"` before reinstalling. CUDA-wheel size depends on platform/release; do not diagnose it from a fixed download-size claim.

To switch an existing environment to the CPU build, reinstall torch from the CPU index (Linux/Windows — on macOS the PyPI wheel is already CPU-only, so no action is needed):

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall
python -m pip check
```

> [!NOTE]
> The `[tool.uv.*]` config in `pyproject.toml` pins torch to the CPU index for **uv** workflows (`uv sync`, `uv pip`) and any locally generated `uv.lock`. Plain `pip` does not read that config, so the documented pip commands pass `--extra-index-url` instead.

## Windows: HuggingFace cache symlink warning

On first tokenizer download you may see:

> `huggingface_hub` cache-system uses symlinks by default ... will be ignored on Windows

The cache can fall back to copies, using more disk depending on revisions and shared files. It is usually a warning, not a training error. Avoid running as administrator solely to suppress it. If Developer Mode is permitted by local policy, it can enable symlinks; otherwise suppress the warning without changing cache behavior:

```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

In cmd.exe use `set HF_HUB_DISABLE_SYMLINKS_WARNING=1`; in Bash use `export HF_HUB_DISABLE_SYMLINKS_WARNING=1`. These are shell-specific forms of the same setting, not interchangeable commands.

Running inside **WSL** avoids this entirely.

## PowerShell: `Start-Process` swallows recipe output

> [!WARNING]
> Do **not** launch the recipe with `Start-Process python -ArgumentList @(...)`. It detaches from the console so stdout/stderr (and any chz parse error) is silently discarded.

Call `python -m interactive_training.recipes.math_rl.train_azure ...` directly so you see the logs.

## PowerShell / cmd: line continuations

The `\` line-continuations in the docs are bash-only. In **PowerShell** use a backtick `` ` `` at end of line; in **cmd.exe** use `^`. Easiest: put the whole invocation on a single line, or run in WSL.

## `key=value` vs `--key=value`

Training launchers using [`chz`](https://github.com/openai/chz) take `key=value` (no leading `--` and no spaces around `=`): `learning_rate=2e-5`, **not** `--learning-rate 2e-5` or `--learning_rate=2e-5`. Boolean fields take `key=true` / `key=false`. Run the chosen launcher's `--help` to see its fields/defaults. Dashboard and visual-spatial sampling use flag-style parsers ([CLI conventions](./README.md#cli-conventions)). Math does not accept `preflight_only`, `preflight_dataset`, `oversample_cushion`, or the shared programmatic console fields.

## Auth fails a few seconds in

If the recipe fails while acquiring a token or creating a session, check the
credential actually selected and its access to the project. For Azure CLI auth,
verify the tenant/subscription in your `az login` session. Math RL also supports
managed identity (including AML UAMI); code/tool/search/NER launchers exclude
managed identity but still use the rest of the `DefaultAzureCredential` chain.
See [auth.md](./auth.md) for the recipe-specific behavior and local HTTP limits.

## Long-running jobs: don't run them from a laptop or dev box

The client process drives the training loop and the SDK's heartbeat. Sleep, hibernation, network suspension, or process termination can interrupt operations and eventually make live state unavailable. The exact idle-expiry/reaping policy is service-specific; do not depend on a fixed timeout or assume unsaved state survives a disconnected driver.

If this happens, you can pick up from the last checkpoint — see [Resuming after an interruption](./training.md#resuming-after-an-interruption).

For anything beyond a quick smoke test (i.e. anything that runs longer than the time you're willing to babysit a terminal), launch the recipe from a **persistent, always-on machine** instead:

- **An approved always-on VM/container**, or Azure ML compute, with enough CPU/RAM/disk for the selected data and grader. The model GPUs are service-side, but code grading, retrieval, and large datasets can demand substantial driver resources. Provisioning must be explicitly requested; warn about costs without a separate spending-approval prompt.
- Use Cloud Shell only for short administrative work when its runtime/storage/network constraints fit; it is not a guarantee of an unattended always-on training driver.
- If you must use your laptop, at minimum keep it plugged in, disable sleep/hibernate (`caffeinate` on macOS, `systemd-inhibit` on Linux, or "never sleep when plugged in" in Windows power settings), and keep the lid open and the network connected.

A typical Azure VM workflow:

```bash
# one-time on the VM
az login
git clone https://github.com/johnwu0604/fine-tuning.git
cd fine-tuning/interactive_training
python -m venv .venv && source .venv/bin/activate
# Offer CPU Torch for driver utilities; inspect the selected build.
python -m pip install -e . --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu

# launch under tmux/screen so the run survives SSH disconnects
tmux new -s run
# Run the requested bounded command from docs/quickstart.md here.
# Ctrl-b d to detach; `tmux attach -t run` to come back
```

Logs land under `<logs-root>/<recipe>/<run_name>/` on the VM (see [storage.md](./storage.md)). You can run [dashboard_server.py](../dashboard_server.py) on the same VM and tunnel port 8000 over SSH (`ssh -L 8000:localhost:8000 ...`) to view it from your laptop.

## Stale session is still holding GPU memory

Normal recipes attempt cleanup, but a crashed notebook, setup failure, or failed close can leave resources active. Inspect the exact owned ID and completed training saves, then deliberately unload it if appropriate ([session management](./session-management.md#free-gpu-memory-begin_unload)). Closing the HTTP client, stopping the dashboard, or deleting local files does not release the session. The management notebook's legacy `ready` aggregate filter is not a current residency check; use its raw list/get cells.

## Editable install: changes don't take effect

The documented install uses `-e` (editable) for exactly this reason. If you accidentally ran without `-e` (`pip install . --index-url https://pypi.org/simple`) and then started editing source, your edits won't be picked up — Python is loading the copy in `site-packages`. Reinstall with `-e`:

```bash
python -m pip install -e . --index-url https://pypi.org/simple --no-deps
```

Verify with:

```bash
python -c "import interactive_training.recipes.math_rl.train_azure as m; print(m.__file__)"
```

The path should point at your source tree, not `site-packages`.

## Stale SDK: the installed wheel lags the cookbook

The public cookbook installs the SDK from PyPI, not from a bundled wheel. SDK
import errors or unexpected/missing arguments can mean the installed release
does not match the cookbook dependency (`azure-ai-finetuningsessions==1.0.0b1`).

```bash
python -m pip show azure-ai-finetuningsessions
python -m pip install --upgrade -e . --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip check
```

If migrating from `azure-ai-finetuning-sessions` or a same-version local preview,
perform the conditional [SDK uninstall/reinstall](#replacing-an-older-sdk-preview).
The [cookbook package rename](#updating-an-older-cookbook-checkout)
alone does not require uninstalling the public SDK.
Custom recipes must import `azure.ai.finetuningsessions`, not the old
`azure.ai.finetuning_sessions` namespace. Both distributions must not coexist.

Beta releases have distinct version numbers. A new cookbook revision must pin a
compatible published beta before it is released. Do not build from a sibling
SDK checkout or use a local fine-tuning SDK wheel to work around a missing
public API.

## Migrating an older preview environment

Use the selected virtual environment, not a global Python installation. Apply
only the migration that matches the installed packages. A fresh environment
needs only the normal [installation](../README.md#install), not uninstall
commands or local-directory deletion.

### Updating an older cookbook checkout

If that environment still has the previous cookbook distribution installed,
remove its stale editable-install metadata, then install the renamed cookbook
from the cookbook root:

```bash
python -m pip uninstall -y interactive-post-training
python -m pip install -e . --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
```

This changes the cookbook distribution to `interactive-training` and imports
to `interactive_training`. It does **not** require uninstalling the published
`azure-ai-finetuningsessions` SDK. Update private scripts and module commands;
no old cookbook import alias is provided. The rename does not move existing
logs or caches ([storage compatibility](./storage.md#logs-root)).

### Replacing an older SDK preview

Skip this section if the environment already uses the pinned public SDK.
Only if it contains the older `azure-ai-finetuning-sessions` distribution or a
locally built SDK preview, remove those copies before reinstalling the cookbook:

```bash
python -m pip uninstall -y azure-ai-finetuning-sessions azure-ai-finetuningsessions
python -m pip install -e . --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
```

This conditional cleanup also handles local previews with the same version as
a published release. Update old SDK imports from `azure.ai.finetuning_sessions`
to `azure.ai.finetuningsessions`; do not keep both SDK distributions installed.
On macOS, omit the extra CPU Torch index as described in the installation guide.

## Tokenizer or dataset download fails

Separate package installation, tokenizer loading, and dataset loading from Azure session creation. Verify the model/dataset identifier, network/proxy policy, cache path permissions/free space, and any dataset access/license requirements. Do not delete the entire shared Hugging Face cache as a first response. If a cached revision is corrupt, identify that specific revision and use the library's documented cache tools with approval.

The quickstart's tokenizer check has **no Azure call** but can download files. Tulu3's `preflight_only=true` can build/render data before session creation; math has no equivalent CLI and constructs data after creating its session. An offline cache miss can therefore fail an otherwise correctly installed environment. Do not disable TLS checks or add credentials to logs to work around downloads.

## SFT truncation or empty loss masks

Check which builder actually forwards the context/truncation fields. The current [Tulu3 builder](../interactive_training/recipes/tulu3_sft/README.md#sft-on-tulu3) does not, even though the CLI accepts them. A successful preflight is not proof that labels survived; inspect decoded batches, rendered lengths, and positive `weights` locally before a remote launch.

For builders that enforce the checks, shorten or split overlength examples, or raise `max_length` only within the verified model/service context budget. Lowering `max_length` cannot preserve content that already exceeds it. Image inputs also consume context. Separately, a split smaller than `batch_size` can produce zero full batches; changing length alone does not fix that. Keep fail-fast checks enabled when preservation is required and follow the [custom-data inspection](./custom-data.md#2-render-and-inspect-locally).

## Session created but no progress

Read the last completed phase in the console/`logs.log`, then inspect the service request/error details and owned session. Cold model/sampler loading, checkpoint serialization, queueing, and grading can take time; no universal first-step duration is promised. A submitted request or warning is not completion. Keep the driver alive and do not submit duplicate paid runs just because no metric row appeared yet.

If a request is terminally failed, preserve its error code/request ID and follow the service's retry guidance. For GPU/memory errors, verify the actual model/rank/context/group/batch configuration and checkpoint/sampler operation involved; do not reinterpret a sampler allocation as a cheap disk-only save. Reduce only the relevant workload within the requested scope for a subsequent run, or escalate with sanitized evidence. Resume from a completed **training** checkpoint when available; unsaved state cannot be reconstructed from logs.

## Answers truncate or quality does not improve

Check `*/env/all/ac_tokens_frac_at_max`, completion-length tails, `format`, scored-episode coverage, and actual trajectory HTML. `max_tokens` includes reasoning; the math default of five tokens is for toy arithmetic, not GSM8K. Renderer selection must match the model. Changing reasoning mode or budget can change results independently of training, so keep the comparison contract fixed.

Check labels, loss masks, grader correctness, split leakage, constant-reward filtering, learning rate, and number of real updates before spending a larger budget. Training reward, held-out NLL, and task accuracy are different measurements. Use [evaluation and inference](./evaluation-and-inference.md); a workflow smoke test can succeed with zero quality gain, and missing evaluation is not a zero score.

## Minimal support evidence

Share only sanitized: cookbook commit, Python/platform, SDK version/import root, launcher name and non-secret config, session/request IDs through an approved support channel, failing phase/error, last completed checkpoint, and attempted cleanup result. Exclude keys/tokens, raw private data, full HTTP logs, and sensitive `code.diff`/notebook outputs. Never attach an entire run directory without review.
