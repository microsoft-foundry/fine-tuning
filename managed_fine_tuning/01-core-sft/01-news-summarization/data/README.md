# Dataset provenance

These files preserve the exact demonstration records, including prompts and answers. Only checkout line endings differ from some source Git blobs. The notebook validates canonical LF byte counts and SHA-256 values without rewriting the checked-in files.

| Cookbook file | Source | Rows | Bytes | SHA-256 |
|---|---|---:|---:|---|
| `news-summarization-train.jsonl` | `Demos/SFT_CNN_DailyMail/training.jsonl` | 1992 | 9057765 | `c6dbdefa54a81868d69f568ba7c54a31684a43f2274e0a45e49ebefb662c532a` |
| `news-summarization-validation.jsonl` | `Demos/SFT_CNN_DailyMail/validation.jsonl` | 229 | 986359 | `a50b24bf6c2edfe5151eb431c018e8ed7d6a45034c8acd19e8150d94e7230910` |

Each non-empty JSONL line is parsed before upload and must contain a non-empty `messages` array. The validated canonical LF bytes are uploaded without changing records, prompts, or answers.
