# ChartQA data

The notebook retrieves [`HuggingFaceM4/ChartQA`](https://huggingface.co/datasets/HuggingFaceM4/ChartQA), retains only human-authored examples, preserves the official train/validation/test partitions, and samples each independently with seed 42.

Images are converted from source bytes to RGB JPEG at quality 95. Row fingerprints assert no overlap across train, validation, and holdout. Training consumes the exact generated files returned by preparation.

`preserved/` contains all 12 committed source JSONL files byte-for-byte, including each historical train, validation, and test variant. [`hash-manifest.csv`](hash-manifest.csv) records source and destination SHA-256 hashes, row counts, and sizes.

Review the [ChartQA paper](https://aclanthology.org/2022.findings-acl.177/) and dataset card for attribution and terms.
