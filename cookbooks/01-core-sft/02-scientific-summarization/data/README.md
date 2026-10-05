# Dataset provenance

These files preserve the exact demonstration records, including prompts and answers. Only checkout line endings differ from some source Git blobs. The notebook validates canonical LF byte counts and SHA-256 values without rewriting the checked-in files.

| Cookbook file | Source | Rows | Bytes | SHA-256 |
|---|---|---:|---:|---|
| `scientific-summarization-train.jsonl` | `Demos/SFT_PubMed_Summarization/training.jsonl` | 1000 | 19202662 | `db97747a8b73373c6eae74a23cd9ce30cbb1901eebdcf443fc9375aa5fc14cb7` |
| `scientific-summarization-validation.jsonl` | `Demos/SFT_PubMed_Summarization/validation.jsonl` | 100 | 2248667 | `8da5c9c840e130c7204b5ba207ce8b02f7b989c715af44c15bf59ffb3e96dbae` |

Each non-empty JSONL line is parsed before upload and must contain a non-empty `messages` array. The validated canonical LF bytes are uploaded without changing records, prompts, or answers.
