# Agent Traces to SFT

Use [`notebooks/demo.ipynb`](notebooks/demo.ipynb) to understand the compatibility contract and [`live_validate.py`](live_validate.py) to run it end to end: generate fresh hosted-agent conversations, acquire authorized Application Insights traces, repair and validate their message structure, remove unsafe content, fine-tune a smaller student, deploy it, and evaluate teacher, base, and fine-tuned models.

The notebook trains only on the exact trace export acquired, transformed, validated, and split during the current run. Acquisition failure stops the workflow; there is no committed trace fallback.

`data/push_prompts.py` is a customer-facing refactor of the original trace-generation helper: its behavior is retained, while resource-specific placeholders and compressed imports were removed. The prompt and tool assets listed in `data/SOURCE_DATA_SHA256SUMS.txt` remain byte-for-byte copies.

The original compatibility contract is the primary result: 100 conversation units, one canonical row per conversation, exact 80/10/10 train/validation/test splits, a `gpt-4.1-mini` teacher, `gpt-4.1-nano-2025-04-14` student, 3 epochs, learning-rate multiplier 1.0, batch size 1, and the original first-assistant-action tool-call evaluator with pass threshold 8. Gain is `(fine-tuned average - base average) / base average * 100`; pass-rate gain is reported separately.

The source is ambiguous because its notebook requests 100 transformed trace rows while its README describes approximately 100 conversations across five sessions. This rehaul resolves that ambiguity conservatively: it generates at least 100 distinct conversations and selects the longest repaired snapshot as the canonical row for each. Exact initial-prompt hashes prove train/validation/test conversation overlap is zero. Row-level splitting of overlapping snapshots is never restored.

The live export repair restores missing assistant call IDs and tool-response references deterministically before validation. The checked-in tool catalog remains in fine-tuning/chat format; inference converts it to the flat Responses API function-tool shape without changing the training artifact. Full-trajectory structural score, exact tool-name sequence, exact arguments, exact trajectory, and extra-call rate remain secondary diagnostics rather than replacing the original-compatible primary metric.

Representative historical metrics are in [`assets/metrics/representative-run.json`](assets/metrics/representative-run.json). Install `cookbooks/requirements.lock`, copy `.env.template` to `.env`, review the privacy controls, and run:

```powershell
python live_validate.py --fresh-conversations 120 --target-conversations 100
```

The runner refuses partial acquisition, non-100 compatibility datasets, split overlap, residual sensitive-pattern matches, failed uploads, non-terminal training, or non-exact job reuse. Runtime configuration and evidence are ignored.

This is the final module in the current sequential curriculum.

> Actual results may vary by model version, data, configuration, region availability, and service conditions.
