# Recipes

A recipe is a small launcher that wires task-specific data and configuration into a shared RL or SFT loop. Run its commands from the **cookbook directory**, not the recipe subdirectory. Each recipe guide lists its prerequisites and source files.

## Choose a path

| Goal | Recipe | Additional prerequisites / safety | Evidence to inspect |
|---|---|---|---|
| First text workflow | [math_rl](../interactive_training/recipes/math_rl/README.md) / [quickstart](./quickstart.md) | Public tokenizer/data downloads; `arithmetic` has no held-out dataset | GSM8K held-out `test/env/all/correct`, episode count, saved checkpoint |
| Learn from labeled conversations | [tulu3_sft](../interactive_training/recipes/tulu3_sft/README.md) | Full-batch preflight is available but has a [truncation/context limitation](../interactive_training/recipes/tulu3_sft/README.md#sft-on-tulu3) | Held-out `test/nll`, positive loss masks, then task evaluation |
| Function-call structure | [tool_rl](../interactive_training/recipes/tool_rl/README.md) | Dataset access; execution-free AST grading | Held-out correctness and format, not successful execution of real tools |
| Executable program solutions | [code_rl](../interactive_training/recipes/code_rl/README.md) | SandboxFusion or Modal; **executes generated code**, including in evaluation | Held-out test pass rate and sandbox failures |
| Retrieval-augmented QA | [search_tool](../interactive_training/recipes/search_tool/README.md) | Vector-search extra, large index download, running retrieval server | Held-out answer correctness and tool-call behavior |
| Grounded entity extraction | [tool_ner_rl](../interactive_training/recipes/tool_ner_rl/README.md) | Public dataset terms; artifact content may be sensitive | Exact-span micro-F1, errors, comparable evaluation manifests |
| Image-grounded tasks | [visual_spatial](../interactive_training/recipes/visual_spatial/README.md) | Image extra, compatible vision model and enabled image fine-tuning | Exact-match task evaluation, not general vision quality |
| Image-returning external tools | [mcp_image_tool](../interactive_training/recipes/mcp_image_tool/README.md) | MCP/image extras, approved tool processes and data access | Fresh/post-SFT/final task evaluations and tool success |
| Pairwise preferences | [preference/dpo](../interactive_training/recipes/preference/dpo/README.md) | Preference pairs and reference-session capacity | Held-out losses plus an independent task/preference evaluation |
| Teacher imitation | [distillation](../interactive_training/recipes/distillation/README.md) | Compatible tokenization, teacher checkpoint, two sessions | Teacher/student difference plus independent quality evaluation |

All remote examples need project authorization, capacity, and budget. A CLI cap does not make a launch offline. Start with [installation/auth checks](./quickstart.md), adapt through [custom data](./custom-data.md), then follow [evaluation and inference](./evaluation-and-inference.md).

## Available recipes

Shared text training and evaluation entrypoints default to `Qwen/Qwen3.8-27B`.
Tokenizer and renderer selection follows the model unless explicitly overridden.
General Qwen3.8 renderer selection uses `qwen3_8_low_reasoning`; the NER recipe
retains its explicit non-thinking preference for its bounded two-turn workflow.
Measured examples retain their original models; vision and specialized model
recipes are deliberate exceptions to the text default.

- **[preference/dpo](../interactive_training/recipes/preference/dpo/)** - Direct Preference Optimization on HH-RLHF, HelpSteer3, or UltraFeedback, with a frozen reference, held-out NLL, and resumable checkpoints.

- **[math_rl](../interactive_training/recipes/math_rl/)** — RL on math problems. Includes three environments:
  - `arithmetic` — deterministic toy addition prompts, useful as an infrastructure check; no held-out test dataset and no guaranteed learning gain.
  - `gsm8k` — grade-school word problems with boxed-answer exact match.
  - `math` — the MATH dataset.

- **[tool_rl](../interactive_training/recipes/tool_rl/)** — RL for tool / function calling on [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k).
  - Single-turn, execution-free — the model emits tool calls graded against gold answers.
  - Uses BFCL-style AST matching with bipartite scoring for smooth RL gradients.
  - Requires Hugging Face authentication (`hf auth login`) and a tool-aware renderer (e.g. `qwen3`, `gpt_oss`, `deepseek_v3`).

- **[code_rl](../interactive_training/recipes/code_rl/)** — RL on competitive programming problems using the [DeepCoder-Preview](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset) dataset.
  - Multi-turn tool-use: a `check_solution` tool lets the model test code in a sandbox before submitting.
  - Grades via LiveCodeBench-style stdin/stdout and functional test execution.
  - Requires a sandbox backend — local Docker [SandboxFusion](https://bytedance.github.io/SandboxFusion/) (default) or [Modal](https://modal.com/docs/guide/sandbox).

- **[search_tool](../interactive_training/recipes/search_tool/)** — RL for multi-hop QA with a search tool over a Wikipedia vector index (Search-R1 style).
  - Multi-turn tool-use with a `search(query_list: list[str])` tool that calls a CPU FAISS + E5 retrieval server (`retrieval_server.py`).
  - Grades the final assistant message using an `Answer:`-prefixed normalized exact-match reward.
  - Requires extra dependencies (`.[vector-search]`) and a running retrieval server fronting the prebuilt Search-R1 wiki-18 E5 FAISS index (no embedding build, no GPU).

- **[mcp_image_tool](../interactive_training/recipes/mcp_image_tool/)** — RFT with a live image-returning MCP tool.
  - Uses the official MCP SDK to discover and call tools through Interactive Training's standard multi-turn tool environment.
  - Includes a local, network-free stdio server whose dashboard image must be read to earn exact-match reward.
  - Defaults to Muse Glimmer 30B and supports customer stdio servers through CLI configuration.

- **[tulu3_sft](../interactive_training/recipes/tulu3_sft/)** — Supervised fine-tuning on [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture), AllenAI's public instruction-tuning blend.
  - Cross-entropy loss on the last assistant message by default (configurable renderer mask).
  - Drives the shared SFT loop in `interactive_training/supervised/train.py`.
  - Tracks `test/nll` on a held-out split — observed to drop from ~0.82 → ~0.48 over 1740 steps on `Qwen/Qwen3-32B`.

- **[visual_spatial](../interactive_training/recipes/visual_spatial/)** — Vision-language RL on the `visual_spatial` subset of [microsoft/Do-You-See-Me](https://huggingface.co/datasets/microsoft/Do-You-See-Me).
  - Demonstrates ordered image/text inputs, dataset preflight, exact-match evaluation, and vision/projector LoRA controls.
  - Requires a vision-language model with image fine-tuning enabled for your project; see the [image-training requirements](./supported_models.md#vision-fine-tuning).

- **[tool_ner_rl](../interactive_training/recipes/tool_ner_rl/)** - Mixed OpenPII/TAB entity extraction with source-grounding and finalization tools, exact-span F1 reward, and no external environment service.

- **[distillation](../interactive_training/recipes/distillation/)** — On-policy distillation from a teacher checkpoint, using compatible teacher/student tokenization.


## Default models

These are CLI defaults for the shared recipes, not claims that every recipe/model
combination has been benchmarked. Explicit `model_name`, `tokenizer_name`, and
`renderer_name` overrides remain supported. When switching models, omit tokenizer
and renderer overrides unless needed so the recipe selects the matching template.

| Recipe | Default model |
|---|---|
| `math_rl`, `code_rl`, `tool_rl`, `search_tool` | `Qwen/Qwen3.8-27B` |
| `tulu3_sft`, `preference/dpo`, `distillation` | `Qwen/Qwen3.8-27B` |
| `tool_ner_rl` (training and evaluation) | `Qwen/Qwen3.8-27B` |
| `mcp_image_tool` | `meta-models/Muse-Glimmer-30B` |
| `visual_spatial` (SFT, RFT, staged training and sampling) | `meta-models/Muse-Glimmer-30B` |

Distillation defaults its teacher to the student's base model. General Qwen3.8
workflows use low reasoning; entity extraction retains its model-aware
non-thinking preference. Image recipes still require image fine-tuning to be
enabled on the endpoint; a default model does not grant that capability.


## Adding a new recipe

See [contributing.md](./contributing.md).
