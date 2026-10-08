# RL for Tool / Function Calling on xLAM-60k

This recipe trains an LLM with reinforcement learning to emit correct tool calls
(a.k.a. function calls — the terms are used interchangeably throughout the
literature, including by Salesforce, OpenAI, and the BFCL paper) for the
[Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
dataset.

## Why tool calling?

The xLAM dataset gives `(query, available_tools, gold_tool_calls)` tuples.
There is no live tool execution — we just need the model to emit the right tool
call(s) given the query and tool schemas. This recipe is **single turn** and
**execution-free**.

## Setup

### 1. Hugging Face dataset access

The xLAM dataset is gated. You must accept its terms before downloading:

1. Go to https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
2. Click **"Agree and access repository"** (requires a Hugging Face account).
3. Authenticate locally so the `datasets` library can download it:
   ```bash
   hf auth login
   ```
   When prompted, paste a [Hugging Face access token](https://huggingface.co/settings/tokens)
   (a **read** token is sufficient).

### 2. Azure authentication

Authenticate to the Azure AI Fine-Tuning Sessions endpoint with either an API
key (`export AZURE_AI_API_KEY=...`) or `az login` (`DefaultAzureCredential`) —
see [docs/auth.md](../../../docs/auth.md).

## Training

The current default is `Qwen/Qwen3.8-27B`. After granting dataset access, start
with a bounded workflow check:

```bash
python -m interactive_training.recipes.tool_rl.train_azure \
  project_endpoint="<your-project-endpoint>" \
  renderer_name=qwen3_8_low_reasoning \
  max_steps=2 max_train_examples=16 max_test_examples=8 \
  groups_per_batch=8 group_size=4 max_tokens=1024 \
  eval_every=1 save_every=1
```

### Historical Qwen3-32B benchmark configuration

The explicit model and renderer below reproduce the historical measurement;
the reported scores are not Qwen3.8 results.

```bash
python -m interactive_training.recipes.tool_rl.train_azure \
    project_endpoint="<your-azure-ai-project-endpoint>" \
    tokenizer_name="Qwen/Qwen3-32B" \
    model_name="Qwen/Qwen3-32B" \
    renderer_name=qwen3_disable_thinking \
    learning_rate=2e-5 \
    temperature=1.0 \
    max_tokens=1024 \
    lora_rank=32 \
    group_size=8 \
    groups_per_batch=256 \
    loss_fn=importance_sampling \
    seed=42 \
    eval_every=20 \
    save_every=20
```

After 100 steps of training on `Qwen/Qwen3-32B`, `test/env/all/bfcl_strict` improves from ~0.5951 to ~0.8268 (exact numbers may vary slightly between runs due to RL stochasticity).

## Data flow (one rollout)

1. Load a `(query, tools, gold_answers)` row from the dataset.
2. Convert xLAM tool schema to JSON Schema (`int` → `integer`, `list` → `array`,
   etc.) so the renderer can embed it.
3. The renderer (`create_conversation_prefix_with_tools`) renders a system message
   containing the tool schemas in the model-specific wire format
   (`<tools>{...}</tools>` for Qwen3, `<|start|>system...` for GPT-OSS, etc.).
4. Append the user query and tokenize.
5. Sample one assistant response from the model. Parse out tool-call blocks
   into structured `ToolCall` objects (delegated to the renderer's
   `parse_response`).
6. Grade with BFCL-style AST matching against the gold answers and end the episode.

## Reward function

```
reward = format_coef * (format_score - 1) + correct
```

- **`format_score`** ∈ {0, 1}: 1 iff the assistant message contains at least one
  parseable tool call AND no unparseable ones. Penalizes pure text replies and
  malformed JSON.
- **`correct`** ∈ [0, 1]: bipartite-matched calls / `max(|pred|, |gold|)`. Each
  call is matched per BFCL strict AST rules (see below). This gives a smooth
  gradient — partial credit when the model gets some calls right.
- **`format_coef`** = 0.1 by default.

### Per-call match rules (BFCL AST semantics)

Reference:
[BFCL evaluation methodology](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html#evaluation-metrics).

- Function name: exact match, with `.` ↔ `_` substitution allowed.
- Argument keys: exact set equality (no missing, no hallucinated extras).
- Per-value:
  - `bool`: strict (the string `"true"` does not match `True`).
  - `int` / `float`: an `int` is accepted where a `float` is expected; a `float`
    is **not** accepted where an `int` is expected.
  - `str`: case-insensitive after stripping whitespace and the punctuation set
    `,./-_*^`.
  - `list` / `tuple`: order-sensitive, recursive value match.
  - `dict`: key-set equality, recursive value match (key order ignored).

### Logged metrics

Per trajectory:

- `format` — the format score above.
- `correct` — the bipartite partial score (also the RL signal).
- `bfcl_strict` — 1.0 iff every gold matched and `|pred| == |gold|`. Logged for
  comparison against BFCL-style metrics — **not used in the reward** (it's too
  sparse for RL).
- `n_pred`, `n_gold` — call-count diagnostics.

## Held-out eval

The xLAM-60k dataset ships as a single split. We deterministically shuffle with
`seed` and take the last `max_test_examples` rows (default 1000) as the held-out eval
set, which is run every `eval_every` iterations.

⚠️ **This eval is NOT directly comparable to the BFCL leaderboard.** xLAM-60k
was the *training* data for Salesforce's xLAM models; the leaderboard numbers
come from the separate BFCL test set scored by the official harness:

For an independent BFCL evaluation, follow the
[official harness instructions](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
in a separate evaluation environment and verify that its model-access interface
supports your intended setup. A cookbook checkpoint identifier is not a local
weights export that the BFCL CLI can load directly; this recipe supplies no
export or serving adapter. Obtain approval and budget before any evaluation
that invokes paid inference.

The xLAM holdout `bfcl_strict` metric here is internally consistent for tracking
training progress but uses the same data distribution as training — it does not
substitute for an independent BFCL run.

## Files

- [`data.py`](data.py) — Dataset loading + xLAM-to-JSON-Schema type conversion.
- [`grading.py`](grading.py) — BFCL AST per-call match + bipartite scoring.
- [`env.py`](env.py) — `ToolCallEnv`, `XLAMDatasetBuilder`.
- [`train_azure.py`](train_azure.py) — CLI entry point (Azure AI Fine-Tuning Sessions SDK).

## References

- APIGen / xLAM paper: https://arxiv.org/abs/2406.18518
- BFCL methodology: https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html
- BFCL leaderboard: https://gorilla.cs.berkeley.edu/leaderboard.html
