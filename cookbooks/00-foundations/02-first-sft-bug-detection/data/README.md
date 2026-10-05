# Data provenance: First SFT: Code Bug Detection

These files were copied without transformation. Their byte content and split membership are preserved.

| Target file | Original source | Split | Rows | Bytes | SHA-256 |
|---|---|---|---:|---:|---|
| `bug-detection-train.jsonl` | `Sample_Datasets/Supervised_Fine_Tuning/Text-Bug-Detection/training_data.jsonl` | training | 224 | 214155 | `9d7a3c11a6ba7cf0a644702557665e79882cbf1ceb074fabf3866e914e7969ca` |
| `bug-detection-validation.jsonl` | `Sample_Datasets/Supervised_Fine_Tuning/Text-Bug-Detection/validation_data.jsonl` | validation | 20 | 20915 | `781c1e3a44e8bdb814beb6f2443b9307bc1698d1c76c2653566f2095a8e98229` |

## Schema

Each JSONL line is one object with a `messages` array in this exact order:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

The notebook verifies JSON parsing, role order, non-empty content, row count, duplicate rows, and SHA-256
before upload. The validated target files are the exact upload inputs.

## Split and holdout policy

The training and validation memberships are unchanged. No independent holdout was present in the source
material, and none was fabricated during migration. The notebook uses a fixed subset of validation rows for
an instructional base-versus-fine-tuned comparison. Create a separately governed test set for a real project.

## Source, license, and safety

Repository sample dataset; original external license is not recorded.

Confirm that your intended use complies with the original dataset terms, privacy obligations, and applicable
policies. Repository inclusion and source attribution do not grant additional rights. Review samples for PII,
secrets, harmful content, domain risk, and representation gaps before adapting this material.

Do not edit these preserved files to work around a preprocessing or safety failure. Instead, create a new,
versioned dataset with documented transforms, filters, approvals, and fresh hashes.
