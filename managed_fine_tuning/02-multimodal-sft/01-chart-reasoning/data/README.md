# ChartQA data

The canonical workflow uses the current-branch source demo's final v4 files
without regenerating or transforming them:

- `chart-qa-run-20261002-v4-train.jsonl`: 50 rows
- `chart-qa-run-20261002-v4-val.jsonl`: 20 rows

The source demo created these files from the official ChartQA train and
validation partitions after retaining human-authored examples, sampling each
partition independently with seed 42, converting images to RGB JPEG at quality
95, and embedding them as data URLs. The system prompt, indexed question text,
answers, and image bytes are preserved exactly.

[`hash-manifest.csv`](hash-manifest.csv) records source and destination SHA-256
hashes, row counts, and sizes. The notebook refuses to upload a file if any
value differs.
Git text conversion is disabled for these byte-pinned JSONL files so their
hashes remain identical across platforms.

Review the [ChartQA paper](https://aclanthology.org/2022.findings-acl.177/) and dataset card for attribution and terms.
