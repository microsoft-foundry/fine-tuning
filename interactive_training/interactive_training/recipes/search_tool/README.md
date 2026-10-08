# RL for Multi-Hop QA with a Search Tool (Search-R1 Style)

This recipe trains an LLM with reinforcement learning to answer multi-hop QA
questions by iteratively calling a `search` tool over a Wikipedia vector index.

Like `code_rl`, this is a **tool-calling, multi-turn** RL setup. The model can
alternate between reasoning and tool calls for up to `max_turns` turns, then is
graded on its final answer.

**End-to-end flow:** a prebuilt FAISS index of the wiki-18 corpus (embedded
with E5, `intfloat/e5-base-v2`) is downloaded from Hugging Face and served by a
local CPU retrieval server (`retrieval_server.py`) → the model's `search`
queries are encoded with the *same* E5 model and matched against the index by
brute-force inner product → the top-k passages are returned. No embedding
build and no GPU are required: the index already exists and E5 query encoding
runs on CPU.

## Setup

### 1. Install extra dependencies

The search recipe needs the FAISS/serving and parquet dependencies:

```bash
# The extra index makes pip pull CPU-only torch — this recipe is entirely CPU
# (E5 query encoding + brute-force FAISS), so the CPU build is all you need and
# you skip ~4 GB of unused CUDA libraries. On macOS pip falls back to PyPI's
# torch, which is already CPU-only.
python -m pip install -e '.[vector-search]' --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. Download the prebuilt E5 index and corpus

Retrieval is served from the Search-R1 prebuilt **E5** index (no build step):

- Index: [`PeterJinGo/wiki-18-e5-index`](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index)
  — a FAISS `IndexFlatIP` split into `part_aa` + `part_ab`; concatenate them
  into a single `e5_Flat.index` (`cat part_a* > e5_Flat.index`).
- Corpus: [`PeterJinGo/wiki-18-corpus`](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus)
  — `wiki-18.jsonl` whose line order matches the index rows.

The flat index holds ~21M float32 vectors (~64 GB) and is loaded fully into
RAM, so the serving host needs enough memory. **Plan on at least 128 GB of
RAM** for the retrieval server host — the ~64 GB resident index plus headroom
for the OS, Python, E5 query encoding, and the FastAPI server. The bench pins
this to a large-memory compute.

### 3. Start the retrieval server

`retrieval_server.py` loads the FAISS index + corpus, encodes queries with E5
(mean pooling + L2 normalize, the `"query: "` prefix) on CPU, and serves
`/retrieve` on `localhost:8000` (the recipe defaults):

```bash
python -m interactive_training.recipes.search_tool.retrieval_server \
    --index-path /data/e5_Flat.index \
    --corpus-path /data/wiki-18.jsonl \
    --model intfloat/e5-base-v2 \
    --host 127.0.0.1 --port 8000
```

### 4. Azure authentication

Authenticate to the Azure AI Fine-Tuning Sessions endpoint with either an API
key (`export AZURE_AI_API_KEY=...`) or `az login` (`DefaultAzureCredential`) -
see [docs/auth.md](../../../docs/auth.md).

## Training

The current text-model default is `Qwen/Qwen3.8-27B`. For a new run, omit the
model/tokenizer overrides; automatic renderer selection uses
`qwen3_8_low_reasoning`. You can pass that name explicitly for clarity or select
`qwen3_8_disable_thinking` when you deliberately want a non-thinking policy.
For a small workflow check, add
`max_steps=2 max_train_examples=16 max_test_examples=8 groups_per_batch=8 group_size=4`.
The retrieval server still requires the large index described above.

### Historical Qwen3-32B configuration

The explicit model and renderer below are retained with their recorded results;
they do not measure Qwen3.8 quality or throughput.

```bash
python -m interactive_training.recipes.search_tool.train_azure \
    project_endpoint="<your-azure-ai-project-endpoint>" \
    model_name=Qwen/Qwen3-32B \
    tokenizer_name=Qwen/Qwen3-32B \
    renderer_name=qwen3_disable_thinking \
    learning_rate=4e-5 \
    temperature=1.0 \
    max_tokens=2048 \
    lora_rank=32 \
    group_size=8 \
    groups_per_batch=32 \
    loss_fn=importance_sampling \
    max_turns=5 \
    format_coef=0.1 \
    retrieval_host=localhost \
    retrieval_port=8000 \
    n_results=3 \
    seed=2 \
    eval_every=20 \
    save_every=20
```

### Results

Eval `correct` over training (Qwen3-32B, settings above):

| Step | Eval correct |
| ---: | -----------: |
|    0 |          33% |
|   20 |          46% |
|  140 |          52% |
|  180 |          54% |

The recipe uses train split `PeterJinGo/nq_hotpotqa_train/train.parquet` and,
by default, evaluates on 1000 examples from `test.parquet`
(`max_test_examples=1000`).

## Tool: `search`

The model is given one tool:

```text
search(query_list: list[str]) -> str
```

For each query, the tool:
1. Sends the raw query strings to the retrieval server.
2. The server encodes them with E5 and retrieves top-k docs from the FAISS index.
3. Returns concatenated snippets back to the model as tool output.

## Data flow (one rollout)

1. Load one `(question, answer_list, data_source)` datum.
2. Render system prompt + tool schema using
   `create_conversation_prefix_with_tools`.
3. Append the user question.
4. Sample assistant output; if it contains a tool call, execute `search` and
   append tool output to history.
5. Continue until no tool call or `max_turns` reached.
6. Grade only the final assistant message.

## Reward function

Graded once at episode end:

```
reward = format_coef * (format - 1) + correct
```

- `format` in `{0, 1}`: 1 iff the final assistant message contains exactly one
  parseable `Answer:` segment.
- `correct` in `{0, 1}`: 1 iff normalized final answer matches any gold answer.
- `format_coef` defaults to `0.1`.

Logged per trajectory: `format`, `correct`.

## Files

- [`search_env.py`](search_env.py) - dataset loading and multi-turn tool-use env wiring.
- [`tools.py`](tools.py) - `RetrievalTool`, retrieval config, answer normalization, reward.
- [`retrieval_server.py`](retrieval_server.py) - CPU FAISS + E5 retrieval server.
- [`train_azure.py`](train_azure.py) - CLI entry point for Azure AI Fine-Tuning Sessions.

## References

- Search-R1 paper: https://arxiv.org/abs/2503.09516
- Dataset: https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train
