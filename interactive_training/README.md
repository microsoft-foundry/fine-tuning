# Interactive Training Cookbook

Examples and reusable training loops for the **Azure AI Fine-Tuning Sessions public preview**. For ML practitioners who can run Python and want to control their data, rewards, and evaluation without managing service-side GPU topology or sharding. Training is driven by a local Python process; the model runs on Azure, not on your laptop.

> [!IMPORTANT]
> Interactive training requires explicit access approval. Request access through the [preview sign-up form](https://aka.ms/foundry-interactive-training-signup) and wait for approval before creating a training session.

## How to use this repo

**New here? Follow the [quickstart](./docs/quickstart.md)**: check the environment, confirm project access, run a bounded example, inspect its artifacts, and verify cleanup. Do not start with a historical benchmark command.

| Your goal | Start here | What you need |
|---|---|---|
| First end-to-end workflow check | [Quickstart](./docs/quickstart.md) | Python 3.11+, project access, model capacity, and configured run limits |
| Teach a model from good example responses (SFT) | [Tulu3 recipe](./interactive_training/recipes/tulu3_sft/README.md), then [your own data](./docs/custom-data.md#supervised-chat-data) | Labeled conversations and a separate evaluation set |
| Optimize a measurable task reward (RL) | [Recipe chooser](./docs/recipes.md#choose-a-path), then [custom rewards](./docs/custom-data.md#reinforcement-learning-data-and-rewards) | Prompts, a trustworthy grader, and held-out tasks |
| Learn from preferred/rejected responses | [DPO recipe](./interactive_training/recipes/preference/dpo/README.md) | Preference pairs |
| Distill a teacher into a student | [Distillation recipe](./interactive_training/recipes/distillation/README.md) | Compatible tokenization and capacity for two sessions |
| Measure improvement and use the saved model | [Evaluation and inference](./docs/evaluation-and-inference.md) | A completed run or a saved training checkpoint |
| Use SDK primitives or extend a recipe | [SDK reference](./docs/sdk-reference.md) and [contributing](./docs/contributing.md) | Familiarity with async Python and the client/service boundary |

**Directory convention:** run commands from `fine-tuning/interactive_training/`, the directory containing this README, `pyproject.toml`, and `dashboard_server.py`—not from the parent fine-tuning directory. The outer cookbook and inner Python package now both use `interactive_training`; use `python -m interactive_training...` for module commands. The cookbook distribution is `interactive-training`. The separately published SDK is installed as a dependency; no sibling SDK checkout is needed. See [storage rename compatibility](./docs/storage.md#logs-root) for the unchanged legacy logs configuration.

> [!WARNING]
> Training, evaluation, and sampling allocate or use remote compute and may incur charges. Package/tokenizer/dataset downloads also need network access. Code-execution, retrieval, and MCP recipes have additional prerequisites; read their guides before starting sidecars or tools. A successful smoke run is not proof of model-quality improvement or a production deployment.

For coding agents, the repository's [execution contract](https://github.com/microsoft-foundry/fine-tuning/blob/main/AGENTS.md) defines the offline verification path, nonblocking paid-run warning, and scope/cleanup safeguards.

Shared text recipes default to **`Qwen/Qwen3.8-27B`**. Unless overridden,
tokenization follows `model_name` and the renderer is selected for that model.
The **renderer** is the model-specific chat-format encoder and response parser,
not a different model or a deployment configuration.
The automatic Qwen3.8 renderer is **`qwen3_8_low_reasoning`**, with thinking
enabled at low effort. Explicit `qwen3_8` still selects `xhigh`; specialized
recipes that explicitly prefer non-thinking keep that behavior.
Image recipes default to **`meta-models/Muse-Glimmer-30B`**.
Recorded benchmarks retain their original model names.

## Supported models & regions

See [Supported models](./docs/supported_models.md) for the authoritative
interactive model identifiers, current eligibility, and fine-tuning regions.
Check [authentication and access](./docs/auth.md#confirm-access-before-spending)
for project authorization, model capacity, and quota before running a recipe.

## Install

The cookbook uses **`azure-ai-finetuningsessions==1.0.0b1` from [PyPI](https://pypi.org/project/azure-ai-finetuningsessions/)**.
Python imports use `azure.ai.finetuningsessions`, including the `aio` namespace.
No local fine-tuning SDK build, sibling source checkout, or bundled SDK wheel is required.

```bash
git clone https://github.com/johnwu0604/fine-tuning.git
cd fine-tuning/interactive_training
python -m venv .venv
source .venv/bin/activate  # Linux / macOS / WSL only
python -m pip install --upgrade pip
python -m pip install -e . --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
```

On native Windows, activate with `.\.venv\Scripts\Activate.ps1` in PowerShell,
or `source .venv/Scripts/activate` in Git Bash. Windows and WSL environments are
not interchangeable. The [quickstart](./docs/quickstart.md#2-create-and-activate-a-virtual-environment)
also shows how to run without activation when local policy prevents it.

Requires Python 3.11 or newer. The editable install (`-e`) lets recipe edits take
effect immediately. The extra index offers CPU-only Torch for driver-side
utilities; training runs on the Azure-hosted engine. On macOS, use the PyPI Torch
build. See [troubleshooting](./docs/troubleshooting.md) for platform-specific help.

**Updating an older environment?** Follow the
[cookbook rename migration](./docs/troubleshooting.md#updating-an-older-cookbook-checkout).
The cookbook rename alone does not require uninstalling the public SDK. Use the
separate [SDK cleanup](./docs/troubleshooting.md#replacing-an-older-sdk-preview)
only when replacing an older or locally built SDK preview. Fresh environments
need neither cleanup nor local-directory deletion.

### Image recipe dependencies

For image recipes, install the image extra (Pillow and Torchvision, required by
the fast image processor):

```bash
python -m pip install -e '.[image]' --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
```

## Updating

The cookbook pins a tested published SDK release in `pyproject.toml` (currently
`azure-ai-finetuningsessions==1.0.0b1`). New betas use new version numbers; each cookbook sync
selects and validates its SDK release explicitly instead of rebuilding a wheel.

```bash
git pull
python -m pip install --upgrade -e . --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
```

Editable installs pick up source edits, but do not automatically install changed
dependencies. Reinstall after pulling to pick up the cookbook's declared SDK
version and other dependency updates. Normal beta upgrades do not need
`--force-reinstall`. Follow the [cookbook rename migration](./docs/troubleshooting.md#updating-an-older-cookbook-checkout)
when updating an older cookbook installation. Use the separate
[SDK migration cleanup](./docs/troubleshooting.md#replacing-an-older-sdk-preview) only when replacing an
older or locally built SDK preview.

If an SDK import, method, or argument is missing, check the installed package
against the cookbook pin; see [SDK troubleshooting](./docs/troubleshooting.md#stale-sdk-the-installed-wheel-lags-the-cookbook).
Do not substitute a locally built SDK wheel for a missing public release. A cookbook
change requiring an unreleased SDK must wait for a compatible PyPI beta.

## Your first run

Use the [quickstart](./docs/quickstart.md) for the complete prerequisite and platform-specific walkthrough. After installing and authenticating to a project in a [supported region](./docs/supported_models.md), this command performs **at most two training iterations** on a small GSM8K subset. Replace the endpoint placeholder; run from the cookbook directory.

```bash
python -m interactive_training.recipes.math_rl.train_azure \
    project_endpoint="<your-project-endpoint>" \
    model_name="Qwen/Qwen3.8-27B" renderer_name=qwen3_8_low_reasoning \
    env=gsm8k \
    learning_rate=2e-5 temperature=1.0 max_tokens=1200 lora_rank=32 \
    group_size=4 groups_per_batch=8 loss_fn=importance_sampling seed=42 \
    eval_every=1 save_every=1 max_steps=2 \
    max_train_examples=16 max_test_examples=8 \
    log_path=./runs/first-math-check behavior_if_log_dir_exists=raise
```

This is a small workflow check, not a quality benchmark. Runtime depends on
model loading, queueing, and evaluation cadence; no Qwen3.8 timing is claimed.
The 1,200-token budget includes reasoning and the final answer; low effort does
not guarantee that every completion fits. Increase it if responses are truncated.
This example writes to `./runs/first-math-check` and refuses to overwrite it.
See the [success checklist](./docs/quickstart.md#7-check-what-succeeded) before
increasing the budget. Without an explicit `log_path`, the platform's default
logs root applies ([docs/storage.md](./docs/storage.md)).

To view metrics in a browser:

```bash
python dashboard_server.py --root ./runs
# → http://127.0.0.1:8000/
```

See [docs/quickstart.md](./docs/quickstart.md) for a fuller walkthrough.

## The RL train loop

The programmatic shared RL `Config` and `RLTestSetEvaluator` accept
`require_full_validation=True` to require every validation group to complete.
This rejects empty validation datasets and any group still failing after the
configured rollout retries, before publishing records or returning metrics.
Observer callback errors also propagate in this mode. Observers remain optional:
strict metrics-only evaluation uses the same completeness checks. The default
is `False`, preserving best-effort evaluation that drops failed groups. The math
Azure CLI does not expose `require_full_validation`; do not append it to the
quickstart command.

The full implementation lives in [`interactive_training/rl/train_azure.py`](./interactive_training/rl/train_azure.py). The following is **pseudocode inside an async function**; dataset, grading, and batch construction are recipe-specific:

```python
# 1. Create an Azure AI Fine-Tuning Session (loads model on the service side)
# client is azure.ai.finetuningsessions.aio.FineTuningSessionClient
session_id = await client.create_session(**experiment_params)
training_client = AzureSDKTrainingClient(client, session_id, tokenizer)

# 2. Build your dataset — YOU control batching, ordering, curriculum, and filtering
dataset = build_dataset("gsm8k", batch_size=groups_per_batch)

for step in range(num_steps):
    # 3. Get the next batch of prompts
    batch = dataset.get_next_batch() # get_next_batch() can be dynamic since it's user defined.

    # 4. Sync weights to the sampler and get a sampling client
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    # 5. SAMPLE — generate group_size completions for each prompt in the batch
    completions = await asyncio.gather(*[
        sampling_client.sample_async(prompt, num_samples=group_size, sampling_params=...)
        for prompt in batch.prompts
    ])

    # 6. GRADE — compute rewards (client-side, you own this logic)
    rewards = grade(completions, batch.ground_truth_answers)

    # 7. COMPUTE ADVANTAGES — default group-mean baseline, no std division
    advantages = rewards - group_mean

    # 8. FORWARD_BACKWARD — send data to the service
    forward = await training_client.forward_backward_async(training_data, loss_fn="importance_sampling")
    await forward.result_async()

    # 9. OPTIM_STEP — update weights on the service
    optimizer = await training_client.optim_step_async(adam_params)
    await optimizer.result_async()
```

Steps 5, 8, and 9 run on service GPUs. Everything else — dataset construction, batching order, grading logic, advantage computation — runs locally under your control. This is the key difference from black-box fine-tuning: you own the data pipeline and reward function, the service owns the GPU compute.

See [docs/training.md](./docs/training.md) for tunable knobs and the loss-function catalog, and [docs/sdk-reference.md](./docs/sdk-reference.md) for the API surface used above.

## Recipes

- **[math_rl](./interactive_training/recipes/math_rl/)** — RL on math problems. Environments: `arithmetic` (smoke test), `gsm8k`, `math`.
- **[tool_rl](./interactive_training/recipes/tool_rl/)** — RL for tool / function calling on [xLAM-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k). Single-turn, execution-free; grades via BFCL-style AST matching.
- **[code_rl](./interactive_training/recipes/code_rl/)** — RL on competitive programming with the [DeepCoder-Preview](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset) dataset. Multi-turn tool-use with a sandbox `check_solution` tool; grades via LiveCodeBench-style test execution.
- **[search_tool](./interactive_training/recipes/search_tool/)** — RL for multi-hop QA with a Wikipedia search tool (Search-R1 style). Flow: a prebuilt FAISS index of the wiki-18 corpus (embedded with E5, `intfloat/e5-base-v2`) is downloaded from Hugging Face and served by a local CPU retrieval server → the model's `search` queries are encoded with the *same* E5 model and matched by brute-force inner product → top-k passages are returned across up to `max_turns` tool calls → the final `Answer:` text is graded by normalized exact match. No embedding build and no GPU are required.
- **[tulu3_sft](./interactive_training/recipes/tulu3_sft/)** — Supervised fine-tuning on [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture). Cross-entropy on assistant turns only; tracks `test/nll` on a held-out split.
- **[visual_spatial](./interactive_training/recipes/visual_spatial/)** — Vision-language RL on the `visual_spatial` subset of [microsoft/Do-You-See-Me](https://huggingface.co/datasets/microsoft/Do-You-See-Me). Requires a supported vision-language model.

- **[preference/dpo](./interactive_training/recipes/preference/dpo/)** — Offline preference optimization with a frozen reference and resumable checkpoints.
- **[distillation](./interactive_training/recipes/distillation/)** — On-policy teacher/student distillation.
- **[tool_ner_rl](./interactive_training/recipes/tool_ner_rl/)** — Tool-grounded entity extraction with exact-span rewards.
- **[mcp_image_tool](./interactive_training/recipes/mcp_image_tool/)** — SFT followed by RL with image-returning MCP tools; requires a compatible vision model.

See [docs/recipes.md](./docs/recipes.md) for prerequisites and specialized workflows.

## Docs

| If you want to… | See… |
|---|---|
| Understand the client-vs-service split & train loop | [docs/concepts.md](./docs/concepts.md) |
| Set up auth | [docs/auth.md](./docs/auth.md) |
| See which models & regions are supported | [docs/supported_models.md](./docs/supported_models.md) |
| Tune knobs (LoRA rank, group size, loss functions) | [docs/training.md](./docs/training.md) |
| Pick or customize a loss function (DPO, distillation, custom regularizers) | [docs/loss_functions.md](./docs/loss_functions.md) |
| Reference the SDK + cookbook training-client API | [docs/sdk-reference.md](./docs/sdk-reference.md) |
| Find your logs, metrics, and `run_meta.json` | [docs/storage.md](./docs/storage.md) |
| View metrics in a browser | [docs/dashboard.md](./docs/dashboard.md) |
| List / inspect / unload sessions and checkpoints | [docs/session-management.md](./docs/session-management.md) |
| Continue training from a checkpoint | [docs/continual-fine-tuning.md](./docs/continual-fine-tuning.md) |
| Adapt a recipe to my own data and rewards | [docs/custom-data.md](./docs/custom-data.md) |
| Compare before/after and sample a saved checkpoint | [docs/evaluation-and-inference.md](./docs/evaluation-and-inference.md) |
| Resume a crashed / interrupted run | [docs/training.md](./docs/training.md#resuming-after-an-interruption) |
| Browse recipes | [docs/recipes.md](./docs/recipes.md) |
| Compare against reference benchmark numbers | [docs/benchmarks.md](./docs/benchmarks.md) |
| Add a new recipe or docs page | [docs/contributing.md](./docs/contributing.md) |
| Something went wrong | [docs/troubleshooting.md](./docs/troubleshooting.md) |
