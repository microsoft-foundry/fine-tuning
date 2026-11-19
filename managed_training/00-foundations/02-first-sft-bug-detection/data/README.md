# Dataset provenance

These files preserve the exact demonstration records, including prompts and answers. Only checkout line endings differ from some source Git blobs. The notebook validates canonical LF byte counts and SHA-256 values without rewriting the checked-in files.

| Cookbook file | Source | Rows | Bytes | SHA-256 |
|---|---|---:|---:|---|
| `bug-detection-train.jsonl` | `Sample_Datasets/Supervised_Fine_Tuning/Text-Bug-Detection/training_data.jsonl (the file used by Demos/SFT_Bug_Detection)` | 224 | 213931 | `5f0514a7ae2f529e0f78f7ba2a00beb54c525b16832030e616894025dbe3f1df` |
| `bug-detection-validation.jsonl` | `Sample_Datasets/Supervised_Fine_Tuning/Text-Bug-Detection/validation_data.jsonl (the file used by Demos/SFT_Bug_Detection)` | 20 | 20895 | `dd89ca979020003573bbf0b199b93fa712168ba56d953881648e90cff41e924c` |

The source demo also tracked a 10-row held-out scale. Those rows are not copied, uploaded, or used here.

Each non-empty JSONL line is parsed before upload and must contain a non-empty `messages` array. The validated canonical LF bytes are uploaded without changing records, prompts, or answers.

## Upstream copyright and license

The [canonical sample notice](../../../../Sample_Datasets/Supervised_Fine_Tuning/Text-Bug-Detection/README.md)
pins the published Microsoft source at Git revision
`cc042d12f6704be36c70f9f4b2457aeaa12b97c7` and lists its source Git blobs.
The complete upstream MIT notice, including
`Copyright (c) 2025 Azure AI Foundry`, is retained beside these renamed copies
in [LICENSE.md](LICENSE.md). Keep it with redistributed data. This license
addition changes neither the records nor the integrity contract above.
