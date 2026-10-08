# Concepts

Read this once to understand what an interactive training recipe does. For executable steps, use the [quickstart](./quickstart.md); for tuning, use [training.md](./training.md).

## Choose the learning signal

| You have… | Approach | Learning signal | Evaluation question |
|---|---|---|---|
| Good example responses | **Supervised fine-tuning (SFT)** | Cross-entropy on selected assistant tokens | Does held-out NLL fall, and do task outputs improve? |
| Prompts and a reliable task grader | **Reinforcement learning (RL)** | Rewards from generated trajectories, converted into group-relative advantages | Does held-out task correctness/reward improve at the same sampling budget? |
| Preferred/rejected responses to the same prompt | **Preference optimization (DPO)** | A custom differentiable preference objective with a frozen reference | Do held-out preferences and task behavior improve? |
| A teacher to imitate | **Distillation** | Teacher/student log-probability comparison | Does the student improve on an independent task evaluation? |

SFT does not require sampling for each training batch; RL does. The current
shared SFT implementation still warms a sampler and saves sampler-format
weights, so "no per-batch sampling" does not mean no inference allocation.
A reward curve, lower training loss, or a successful API call is not sufficient
evidence of general task improvement. See [evaluation and inference](./evaluation-and-inference.md).

## Vocabulary

| Term | Meaning in this cookbook |
|---|---|
| **Recipe** | A launcher plus task-specific data, prompts, grading, and configuration; not a hosted training-job specification |
| **Azure session** | Service-side model/adapter state and compute, identified by `session_id`; its lifetime differs from local logs |
| **Run** | One local driver execution and its directory of config, metrics, and checkpoint metadata; recovery can create a new Azure session |
| **Renderer** | Model-specific chat-format encoder and response parser; switching it does not change the model or enable vision |
| **LoRA adapter / rank** | Trainable low-rank updates around the base model; rank controls the size of the low-rank factors and their capacity/memory trade-off, not the number of GPUs or a guaranteed quality gain |
| **Environment (`Env`)** | Client-side prompt/interaction/grading contract; `env=gsm8k` selects a dataset/grader combination, not an Azure deployment environment |
| **Prompt group** | Multiple independently sampled trajectories for the same task; `group_size` is completions per prompt, `groups_per_batch` is prompts per iteration |
| **Advantage** | Group-relative learning signal derived from rewards; constant-reward groups contain no relative reward signal |
| **Training checkpoint** | Saved weights plus optimizer state, used for recovery or a fresh session initialized from that state |
| **Sampler checkpoint** | Saved sampling-format weights, without optimizer state; different from a transient per-step sampler update |

Example: eight prompt groups with four completions each produce up to 32 training trajectories for an iteration, **plus** any evaluation/retry/tool calls. This is not a fixed cost estimate.

## What you control vs. what the service controls

An interactive training job is a **client-side recipe** (the Python code in this repo) driving a **service-side engine** (the Azure AI Fine-Tuning Session) via the `azure-ai-finetuningsessions` SDK. The split is sharp:

| **You decide (client-side)**         | **Service decides (engine-side)**         |
|--------------------------------------|-------------------------------------------|
| Dataset & prompt formatting          | GPU topology & tensor parallelism         |
| Number of completions per prompt (`group_size`) | FSDP / sharding strategy       |
| Sampling temperature & max tokens    | Internal microbatch sizes                 |
| Reward / grading function            | Serving batch construction                |
| Advantage computation                | Memory management & KV-cache strategy     |
| Learning rate & optimizer params     | Kernel selection & runtime scheduling     |
| LoRA rank                            | Weight synchronization mechanics          |
| Loss function (`importance_sampling`, `ppo`, `cispo`, `sapo`, `cross_entropy`) | Distributed communication |
| Checkpointing & eval cadence         | Hardware placement                        |

You express the RL loop in a small Python recipe; the Azure-hosted engine executes the `sample`, `forward_backward`, and `optim_step` requests. Keep the client process alive: it drives progress and the SDK heartbeat.

## Contrast with TRL

In TRL/GRPO you configure distribution explicitly:

```python
GRPOConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=32,   # you must compute this from GPU count
    num_generations=8,
    # Other task and optimizer settings omitted in this conceptual comparison.
)
# + accelerate_config.yaml choosing FSDP vs DeepSpeed, shard strategy, etc.
```

With Interactive Training you never specify per-GPU parameters. The service auto-selects topology and batching for the model size — you only describe *what* to train, not *how* to distribute it:

```python
# Interactive Training: no GPU topology, no accelerate config
# (gradient accumulation is opt-in: just call forward_backward multiple times
#  before optim_step)
forward = await training_client.forward_backward_async(batch, loss_fn="importance_sampling")
await forward.result_async()
optimizer = await training_client.optim_step_async(adam_params)
await optimizer.result_async()
```

## Project layout

The outer cookbook directory and **inner Python package** now both use
`interactive_training/`; the tree below shows the inner package.
Run installation and recipe commands from the outer cookbook directory, using
`python -m interactive_training...` for module commands.

```
interactive_training/
  rl/
    train.py            # Generic client-side RL train loop, not the service engine
    train_azure.py      # Azure SDK adapters for train.py
  supervised/
    train.py            # Generic SFT train loop (cross-entropy on assistant turns)
  recipes/
    math_rl/
      train_azure.py    # CLI entry point — wires env + config into rl/train_azure.py
      math_env.py       # GSM8K / MATH prompt builders & grading
      math_grading.py   # Exact-match boxed-answer extraction
    tool_rl/
      train_azure.py    # RL on xLAM-60k function-calling
      env.py            # Tool-call prompt builder
      grading.py        # BFCL-style AST matching with bipartite scoring
      data.py           # xLAM-60k loader
    code_rl/
      train_azure.py    # RL on DeepCoder-Preview competitive programming
      code_env.py       # Multi-turn tool-use env with a `check_solution` tool
      code_grading.py   # LiveCodeBench-style test execution & reward
      deepcoder_tool.py # Sandbox-backed code execution tool
    search_tool/
      train_azure.py    # RL on Search-R1-style multi-hop QA with tool use
      search_env.py     # Search dataset loader + multi-turn tool-use env
      tools.py          # FAISS + E5 retrieval `search` tool + answer reward
      retrieval_server.py # CPU FAISS + E5 retrieval server (run as a sidecar)
    tulu3_sft/
      train_azure.py    # CLI entry point — drives supervised/train.py on
      chat_datasets.py  #   the allenai/tulu-3-sft-mixture chat dataset
  ...                   # Additional utilities, environments, and recipes
```

The pattern: a generic loop in `interactive_training/rl/` (RL) or `interactive_training/supervised/` (SFT), an environment + grading module per task (RL only), and a small recipe entry point that wires them together and exposes a CLI.

## Checkpoint lifecycle

1. A live session holds mutable training weights. An optimizer step updates them.
2. A per-step **ephemeral** sync makes those weights available for rollout sampling; it is not a durable save.
3. Explicit saves persist training state and/or sampler weights. The local ledger records the completed saves; the weight files are remote, not embedded in the ledger.
4. Closing/unloading the session requests release of in-memory resources; verify the remote result. Unsaved state is not persisted, and local logs alone cannot recreate it.
5. Recovery creates a new session from saved training state and restores the local cursor. Continual training uses a fresh log directory and starts its dataset cursor over.

See [SDK save primitives](./sdk-reference.md#persistent-versus-ephemeral-checkpoints), [storage](./storage.md), and [sampling after training](./evaluation-and-inference.md#use-a-training-checkpoint-after-the-original-session-closes) for the concrete APIs and boundaries.
