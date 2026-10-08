# Supervised Fine-Tuning on the Tulu3 Mixture

Tulu3 is AllenAI's public instruction-tuning mixture (`allenai/tulu-3-sft-mixture`) — a curated blend of chat, reasoning, code, math, and safety data used to post-train the Tulu3 series of models. This recipe wires it into the shared supervised train loop (`interactive_training/supervised/train.py`) with cross-entropy loss on the assistant turns only.

Each row in the mixture is a multi-turn chat; the renderer (selected automatically from the base model via `model_info.get_recommended_renderer_name`) tokenizes the conversation. By default this builder trains only the **last assistant message** (`LAST_ASSISTANT_MESSAGE`), not every assistant turn; `train_on_what` can override the mask. Periodic eval reports `test/nll` — the held-out negative log-likelihood — which is the standard SFT health signal.

Run from the **cookbook directory** after [installing and authenticating](../../../docs/quickstart.md). Training is remote and may incur charges; the 30-step example below is smaller than the benchmark, not a price or runtime guarantee. For your own conversations, use the existing [JSONL builder](../../../docs/custom-data.md#supervised-chat-data); this launcher selects Tulu3 and does not accept an arbitrary file-path flag.

## SFT on Tulu3
The default text model is `Qwen/Qwen3.8-27B`. For a shorter run, use:
```bash
python -m interactive_training.recipes.tulu3_sft.train_azure \
    project_endpoint="<your-project-endpoint>" \
    model_name="Qwen/Qwen3.8-27B" \
    learning_rate=5e-4 \
    lora_rank=32 \
    batch_size=128 \
    max_length=1200 \
    num_epochs=1 \
    max_steps=30 \
    eval_every=10 \
    save_every=10 \
    behavior_if_log_dir_exists=raise
```

Use `preflight_dataset=true` to render training/evaluation batches before
allocating a session, or `preflight_only=true` to return after that scan without
training. This can download tokenizers/data and write local metadata; floor
batching can omit tail source rows ([batching caveat](../../../docs/custom-data.md#2-render-and-inspect-locally)).

> [!WARNING]
> **Existing Tulu3 limitation:** the CLI accepts `fail_on_truncation` and
> `model_context_length`, but [Tulu3Builder](./chat_datasets.py) does not pass
> them to `conversation_to_datum`. Setting those flags does not currently
> reject truncated/over-context rows here. Preflight can succeed after
> truncation to `max_length`, even with no remaining assistant loss tokens;
> it is not a guarantee of complete labels or service-context validity.
> Inspect rendered lengths and positive loss weights locally. The existing
> [conversation-file builder example](../../../docs/custom-data.md#2-render-and-inspect-locally)
> forwards the checks, but requires its own programmatic launcher; it is not
> a replacement Tulu3 CLI flag. This limitation is documented, not changed.

Keep `max_steps` **positive** when budgeting: zero/`None` disables the SFT cap,
not an evaluation-only mode. Compare held-out `test/nll` and task-specific
generation quality ([evaluation](../../../docs/evaluation-and-inference.md));
lower training NLL alone is not proof of improvement. Keep the completed ledger
and original config for [recovery](../../../docs/training.md#resuming-after-an-interruption),
or use a fresh directory for [continual training](../../../docs/continual-fine-tuning.md).
The launcher attempts session cleanup in `finally`; inspect owned leftovers
if it reports an error ([session management](../../../docs/session-management.md)).

### Historical Qwen3-32B benchmark configuration

The following configuration and measured NLL values are for Qwen3-32B, not the
current Qwen3.8 default:
```bash
python -m interactive_training.recipes.tulu3_sft.train_azure \
    project_endpoint="<your-project-endpoint>" \
    model_name="Qwen/Qwen3-32B" \
    learning_rate=5e-4 \
    lora_rank=32 \
    batch_size=128 \
    max_length=1200 \
    num_epochs=1 \
    max_steps=1740 \
    eval_every=100 \
    save_every=100 \
    behavior_if_log_dir_exists=raise
```

`max_steps=1740` matches the published `chat_sl` Tulu3 recipe, where `test/nll` plateaus. We observe `test/nll` go from 0.82 at step 0 to 0.48 after 1740 steps.

Here is an example assistant turn from the held-out set, showing the kind of response the model is being trained to imitate:

```
A plane takes off at 6:00 a.m. and flies for 4 hours from New York City to Chicago. The plane stays at the port in Chicago for 1 hour and then departs for Miami. If the aircraft took three times as many hours to fly to Miami than it took to fly from New York to Chicago, calculate the total time to travel from New York to Miami.<|im_end|>
<|im_start|>assistant
The flight from New York to Chicago takes 4 hours. The stop in Chicago adds 1 hour. The flight from Chicago to Miami takes 3 × 4 = 12 hours.

Total: 4 + 1 + 12 = 17 hours.<|im_end|>
```
