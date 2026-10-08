# RL for Competitive Programming on DeepCoder

This recipe trains an LLM with reinforcement learning to solve competitive
programming problems by writing code, executing it in a sandbox, and (if needed)
revising it. The training mixture is the
[DeepCoder-Preview blend](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset)
(PrimeIntellect + TACO + LiveCodeBench v5); rewards come from running the
generated code against the problem's unit tests.

This is a **tool-calling, multi-turn** RL recipe — episodes
run up to 2 assistant turns and the model interacts with a `check_solution`
tool between turns.

## Setup

### 1. Sandbox

Generated code is run in a sandbox during both training and eval. Two backends
are supported.

**SandboxFusion (default).** [SandboxFusion](https://bytedance.github.io/SandboxFusion/)
gives you a local Docker-backed sandbox:

```bash
export INTERACTIVE_TRAINING_ROOT="$PWD"  # run from the cookbook root
docker run -it -p 127.0.0.1:8080:8080 \
   -v "${INTERACTIVE_TRAINING_ROOT}/interactive_training/recipes/code_rl/sandbox_config/local.yaml:/root/sandbox/sandbox/configs/local.yaml" \
    volcengine/sandbox-fusion:server-20250609
```

The training script reads the endpoint from `SANDBOX_URL` (default
`http://localhost:8080/run_code`).

**Modal (alternative).** [Modal](https://modal.com/docs/guide/sandbox) provides
cloud sandboxes without local Docker. After `python -m pip install modal && modal token new`,
pass `sandbox_backend=modal` on the training command. Tunable via
`MODAL_POOL_SIZE` (default 32) and `MODAL_CREATION_RATE_LIMIT` (default 4).
Authentication needs network access, and cloud sandbox use can incur charges;
launch only an approved workload.

### 2. Azure authentication

Authenticate to the Azure AI Fine-Tuning Sessions endpoint with either an API
key (`export AZURE_AI_API_KEY=...`) or `az login` (`DefaultAzureCredential`) —
see [docs/auth.md](../../../docs/auth.md).

## Training

The current default is `Qwen/Qwen3.8-27B`. After starting the sandbox, try a
bounded workflow check (not a benchmark):

```bash
python -m interactive_training.recipes.code_rl.train_azure \
   project_endpoint="<your-project-endpoint>" \
   renderer_name=qwen3_8_low_reasoning \
   max_steps=2 max_train_examples=16 max_test_examples=8 \
   groups_per_batch=8 group_size=4 max_tokens=1200 \
   eval_every=1 save_every=1
```

### Historical Qwen3-32B benchmark configuration

```bash
python -m interactive_training.recipes.code_rl.train_azure \
    project_endpoint="<your-azure-ai-project-endpoint>" \
    model_name="Qwen/Qwen3-32B" \
    renderer_name=qwen3_disable_thinking \
    learning_rate=2e-5 \
    temperature=1.0 \
    max_tokens=1200 \
    lora_rank=32 \
    group_size=8 \
    groups_per_batch=256 \
    loss_fn=importance_sampling \
    sandbox_backend=sandboxfusion \
    seed=42 \
    eval_every=10 \
    save_every=999999 \
    remove_constant_reward_groups=true
```

On `Qwen/Qwen3-32B`, this configuration moves LiveCodeBench v5 mean accuracy
from 12.65% to 20.24% (at 21% of full training); see
[docs/benchmarks.md](../../../docs/benchmarks.md). These results do not measure
the Qwen3.8 default. To select another supported model, pass its `model_name` and
omit the Qwen-specific `renderer_name` override so the recipe selects its renderer.

Historical experiments also used `openai/gpt-oss-20b` with LoRA rank capped at
`16`. It is now a [legacy model](../../../docs/supported_models.md#legacy-models)
and is not supported for new interactive training. Use the public entrypoint
with a currently supported model; no separate setup script is required.

## Tool: `check_solution`

The model is given a single tool:

```text
check_solution(code: str) -> {"passed": bool, "details": ...}
```

It executes `code` in the sandbox against the task's unit tests and returns the
pass/fail summary. Each `DeepcoderTool` instance is bound to one task, so the
tests themselves are never shown to the model — only the verdict on its attempt.

## Data flow (one rollout)

1. Load a `(problem, tests)` row from the DeepCoder-Preview dataset.
2. The renderer (`create_conversation_prefix_with_tools`) emits a system message
   with the `check_solution` tool schema in the model-specific wire format
   (`<tools>{...}</tools>` for Qwen3, Harmony channels for gpt-oss, etc.).
3. Append the LiveCodeBench-style user prompt and sample the first assistant
   response.
4. If the assistant called the tool, run `check_solution` in the sandbox and
   feed the result back as the next observation, then sample the second
   assistant response.
5. Episode ends after at most `max_turns=2` assistant turns, or earlier if the
   model produces a final answer without calling the tool.
6. Grade by extracting the code block from the **final** assistant message and
   running it against the unit tests.

The tool call is best understood as a "free check": the model can spend its
first turn to try a candidate, see whether it passes, and revise on its second
turn. The reward is graded only on the final answer.

## Reward function

Graded once at episode end:

```
reward = format_coef * (has_code_block - 1) + correct
```

- **`has_code_block`** ∈ {0, 1}: 1 iff a ```python ... ``` block is present in
  the final assistant message.
- **`correct`** ∈ {0, 1}: 1 iff the extracted code passes **all** unit tests in
  the sandbox; 0 otherwise.
- **`format_coef`** = 0.1 by default — a small shaping term (`-0.1` reward) for
  responses with no code block, so the policy gets a gradient on format even
  when correctness is 0.

Logged per trajectory: `format`, `correct` (the latter is the main RL signal).

## Files

- [`code_env.py`](code_env.py) — `DeepcoderEnvGroupBuilder`, dataset loading,
  rollout wiring.
- [`code_grading.py`](code_grading.py) — code extraction and sandbox test runner.
- [`deepcoder_tool.py`](deepcoder_tool.py) — `DeepcoderTask`, `DeepcoderTool`
  (the `check_solution` tool), and `DeepcoderReward`.
- [`lcb_utils.py`](lcb_utils.py) — LiveCodeBench-style system-prompt construction.
- [`train_azure.py`](train_azure.py) — CLI entry point.

## References

- DeepCoder: https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51
- DeepCoder-Preview dataset: https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset
- LiveCodeBench: https://livecodebench.github.io/
- SandboxFusion: https://bytedance.github.io/SandboxFusion/
