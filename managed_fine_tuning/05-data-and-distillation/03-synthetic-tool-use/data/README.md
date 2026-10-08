# Preserved synthetic tool-use data

`SOURCE_DATA_SHA256SUMS.txt` records the exact source assets from
`Demos/SyntheticDatagen-ToolUse`: the unchanged 15-tool OpenAI catalog, its
OpenAPI fixture and the returned train/validation/test split. The generated
retail conversations contain synthetic sample personas and order details,
not local Azure project identities or telemetry evidence.

The source recipe converts OpenAI functions into OpenAPI 3.0.3 POST operations,
recursively omitting empty `required` arrays, and requests three
`ToolUseFineTuning` generation batches with `max_samples=60` per batch. Rows are
deduplicated by `json.dumps(row, sort_keys=True)` in first-occurrence order.
Python `random.Random(42)` shuffles row indices; training receives
`int(0.8 * count)` rows, validation `int(0.1 * count)`, and test the remainder.

The service returned 30 unique rows in this exact dataset: 24 training, 3
validation and 3 test rows. Requested maximum sample counts are not actual
returned counts. All bytes are retained without generation, semantic filtering
or renderer normalization. The original generation/split implementation remains
in `Demos/SyntheticDatagen-ToolUse/notebook.ipynb`.

Only training and validation are uploaded. The supervised training recipe is
3 epochs and learning-rate multiplier 1.0 with `GlobalStandard`.
