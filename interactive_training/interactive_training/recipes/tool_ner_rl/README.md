# Tool-Grounded Entity Extraction with RL

LLM extraction usually starts with an output format: entity strings, labeled spans, JSON, or source text annotated with XML tags. Application code then parses the response, aligns mentions with the source, and validates the result.

This recipe brings source grounding and final selection into a two-turn tool workflow. The model proposes text, a Python tool returns matching source spans, and the model submits labeled span IDs. RL optimizes the model's policy against completed extraction quality; the tools remain deterministic. No sandbox, retrieval service, or separate environment server is required.

## Run It

Training and evaluation now default to `Qwen/Qwen3.8-27B` with its model-specific
non-thinking renderer. The bundled UID selection and measurements below were
produced with Qwen3.6; re-evaluate that selection when changing models. The
explicit model/renderer in the measured commands is intentionally retained.

Follow the cookbook [installation guide](../../../README.md#install) and [Azure authentication guide](../../../docs/auth.md). Run commands from the cookbook root with `PROJECT_ENDPOINT` set. Model availability depends on the service and region; the measured run used `Qwen/Qwen3.6-35B-A3B`.

The recipe bundles the measured OpenPII training selection as UIDs only. Pinned public datasets download on first use; no separate filtering run or environment service is needed. The command below uses the full run settings:

```bash
python -m interactive_training.recipes.tool_ner_rl.train_azure \
    train_source_pool_size=10000 validation_pool_size=2000 \
    max_test_examples=500 tab_fraction=0.25 tab_eval_documents=8 \
    model_name=Qwen/Qwen3.6-35B-A3B renderer_name=qwen3_5_disable_thinking \
    lora_rank=32 lora_alpha=32 \
    group_size=8 groups_per_batch=16 pods_keep=4 \
    max_oversample_rounds=4 max_concurrent_groups=4 \
    learning_rate=1e-5 seed=20260828 \
    max_steps=240 max_wall_clock_seconds=57600 \
    eval_every=15 save_every=15 \
    log_path=./logs/tool-ner-mixed behavior_if_log_dir_exists=raise
```

For a smoke run, replace the limits with `max_steps=2`, `max_test_examples=16`, `tab_eval_documents=1`, `eval_every=1`, and `save_every=1`, using a fresh log directory. This checks the workflow, not model quality; it still downloads the datasets. The full measured training run took about 15 hours 38 minutes, excluding preparation and post-training evaluation.

Optional async training: add `max_steps_off_policy=1`. PODS still keeps `pods_keep` rollouts per nonconstant group (four of eight by default). Only the sync refill loop is disabled; the shared async queue replenishes groups and checks staleness. `groups_per_batch` also controls async rollout workers; `max_concurrent_groups` is a sync-only limit. Both modes default to 64 retained trajectories per update. Async is smoke-tested, but not validated for equivalent learning quality.

In-place resume is rejected because the full sampling state is not restored. To continue from saved weights, set `load_checkpoint_path` and use a fresh `log_path`; this restarts the data schedule and step counter, not an exact resume.

## Task and Tools

| Dataset | Task | Labels |
| --- | --- | --- |
| OpenPII | Extract annotated personal information | 19 PII categories |
| TAB | Select mentions to mask for a specified person | Eight native entity categories |

Training schedules 75% OpenPII and 25% TAB tasks before dynamic filtering; the retained trajectory and token proportions may differ. The measured run used 6,541 selected OpenPII documents and 1,004 eligible TAB training documents. TAB supplies full-document context with 128-word active windows and 16-word overlap. Only active-window mentions may be submitted.

Each rollout exposes two tools:

```text
ground_entities(proposals) -> source matches and stable span IDs
finalize(entities)         -> validated labeled spans; ends the episode
```

For a synthetic TAB task protecting Jane Doe, the interaction might be:

```text
Source: "Jane Doe filed the application."
Turn 1: ground_entities(proposals=[{"text": "Jane Doe", "label": "PERSON"}])
Tool:   unique match, span_id="s_...", start=0, end=8  (abbreviated)
Turn 2: finalize(entities=[{"label": "PERSON", "span_ids": ["s_..."]}])
```

Grounding confirms where text occurs, not whether it should be masked or how it should be labeled. Repeated text returns candidate occurrences with context. The model selects the intended IDs; gold annotations stay behind the grader. Each turn allows 768 generated tokens. The shared RL trainer samples groups of rollouts, computes relative advantages, updates the policy, and saves checkpoints.

## Reward

Reward is exact labeled-span F1, minus 0.1 for missing or invalid finalization. A correct entity must match both its source boundaries and label. Tools and postprocessing are fixed code, not learned parameters.

OpenPII and TAB retain separate schemas and metrics. TAB includes DIRECT and QUASI identifiers and excludes NO_MASK mentions. Evaluation samples each window once, scores it against each annotator, and deduplicates overlapping windows in document coordinates. These are custom TAB metrics, not the official scorer.

Evaluation also reports **binary character coverage**: sensitive versus non-sensitive, ignoring entity labels and segmentation. It compares the non-whitespace source characters covered by predicted and gold spans, counting overlapping spans only once per document/reference and summing counts for micro metrics. TAB annotator references are scored separately, not unioned. Recall measures annotated sensitive text covered for redaction; precision measures how much selected text is annotated sensitive. This is a diagnostic metric, not the training reward.

## Results

One 240-step mixed run produced these exact micro-F1 scores:

| Validation set | Base | Step 240 |
| --- | ---: | ---: |
| OpenPII validation, 500 documents | 80.44% | 90.11% |
| TAB dev, all 127 documents | 62.13% | 74.99% |
| TAB dev, 119-document subset excluding the setup eight | 61.55% | 74.63% |

OpenPII uses a fixed 500-document sample from its validation split; this recipe has no separate untouched OpenPII test set. Eight of the 127 TAB dev documents were used during setup and periodic evaluation; the other 119 form the subset above, not a test split. All 127 were later evaluated across checkpoints. The TAB test split remains untouched.

**Binary character coverage**, with each cell showing base to step 240:

| Validation set | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| OpenPII validation, 500 documents | 99.25% -> 98.65% | 90.82% -> 97.79% | 94.85% -> 98.22% |
| TAB dev, all 127 documents | 56.52% -> 81.87% | 78.06% -> 76.06% | 65.56% -> 78.86% |
| TAB dev, 119-document subset excluding the setup eight | 55.96% -> 81.74% | 77.72% -> 75.63% | 65.07% -> 78.56% |

OpenPII character scores come from a fresh paired evaluation on the same 500 validation documents; TAB uses the saved predictions from the exact-span evaluation above. TAB character F1 improved as over-redaction decreased, but sensitive-character recall fell. Higher F1 alone does not imply stronger privacy protection. These are single-run observations, not guaranteed scores; sampled generations vary between runs.

### Evaluate Your Run

The main command reproduces the training setup, with periodic evaluation on 500 OpenPII validation documents and eight TAB setup documents. To evaluate all 127 TAB dev documents, set `CHECKPOINT` to a saved URI from your run's `checkpoints.jsonl`, then run:

```bash
python -m interactive_training.recipes.tool_ner_rl.eval_azure \
    model_name=Qwen/Qwen3.6-35B-A3B renderer_name=qwen3_5_disable_thinking \
    dataset_name=tab dataset_split=dev tab_max_documents=127 \
    load_checkpoint_path="$CHECKPOINT" lora_rank=32 \
    num_groups_to_log=0 log_path=./logs/tool-ner-tab-eval
```

For OpenPII exact-span and character coverage metrics on the fixed validation selection:

```bash
python -m interactive_training.recipes.tool_ner_rl.eval_azure \
    dataset_name=openpii dataset_split=validation \
    model_name=Qwen/Qwen3.6-35B-A3B lora_rank=32 \
    load_checkpoint_path="$CHECKPOINT" \
    max_examples=500 sample_pool_size=2000 sample_seed=20260829 \
    selection_method=random max_documents_scanned=2500 \
    samples_per_task=1 num_groups_to_log=0 \
    log_path=./logs/tool-ner-openpii-eval
```

For a base-model comparison, omit `load_checkpoint_path` and use a fresh `log_path` for each evaluation. These commands write document-level metrics and detailed prediction artifacts. Read `binary_pii` in `error_summary.json` for character coverage; `binary_pii_exact_span` ignores labels but still requires exact entity boundaries. The trainer's OpenPII `exact_f1` is mean-document F1, not the micro-F1 reported above. Offline tests live alongside other recipe tests: `python -m pytest tests/test_tool_ner_rl*.py`.

For checkpoints from a new Qwen3.8 run, use `model_name=Qwen/Qwen3.8-27B` and
`renderer_name=qwen3_8_disable_thinking` instead. Always match the base model and
LoRA configuration to the checkpoint; the default cannot infer them from its URI.

## Datasets

- [OpenPII](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m): personal information extraction.
- [TAB](https://github.com/NorskRegnesentral/text-anonymization-benchmark): contextual anonymization; [paper](https://arxiv.org/abs/2202.00443).

`tab_source_path` accepts a local copy of the pinned TAB file, not a custom corpus; local and downloaded inputs must match the same checksum.

See [dataset attribution and notices](DATA_NOTICES.md) for the included excerpts. Documents are sent to the model service, and evaluation artifacts contain source text even with HTML logging off. Use approved data and protect logs. High F1 does not guarantee safe anonymization.

Evaluation artifacts and offline-filter manifests are written atomically with owner-only file permissions on POSIX filesystems (Linux/macOS). On Windows, files inherit the output directory's ACL; the recipe does not configure or verify Windows access control. Before running on Windows, create the output directory and restrict its ACL to authorized users. Use access-controlled output directories on all platforms.