# Dataset provenance

These files preserve the exact demonstration records, including prompts and answers. Only checkout line endings differ from some source Git blobs. The notebook validates canonical LF byte counts and SHA-256 values without rewriting the checked-in files.

| Cookbook file | Source | Rows | Bytes | SHA-256 |
|---|---|---:|---:|---|
| `bug-detection-train.jsonl` | `Sample_Datasets/Supervised_Fine_Tuning/Text-Bug-Detection/training_data.jsonl (the file used by Demos/SFT_Bug_Detection)` | 224 | 213931 | `5f0514a7ae2f529e0f78f7ba2a00beb54c525b16832030e616894025dbe3f1df` |
| `bug-detection-validation.jsonl` | `Sample_Datasets/Supervised_Fine_Tuning/Text-Bug-Detection/validation_data.jsonl (the file used by Demos/SFT_Bug_Detection)` | 20 | 20895 | `dd89ca979020003573bbf0b199b93fa712168ba56d953881648e90cff41e924c` |

The source demo also tracked a 10-row held-out scale. Those rows are not copied, uploaded, or used here.

Each non-empty JSONL line is parsed before upload and must contain a non-empty `messages` array. The validated canonical LF bytes are uploaded without changing records, prompts, or answers.
