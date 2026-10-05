# Data provenance: SFT Practice: News Summarization

These files were copied without transformation. Their byte content and split membership are preserved.

| Target file | Original source | Split | Rows | Bytes | SHA-256 |
|---|---|---|---:|---:|---|
| `news-summarization-train.jsonl` | `Demos/SFT_CNN_DailyMail/training.jsonl` | training | 1992 | 9059757 | `cf7e7cc1ed8b02f9959c269113bf396546c1e959599376c17095cee7a14ef7fd` |
| `news-summarization-validation.jsonl` | `Demos/SFT_CNN_DailyMail/validation.jsonl` | validation | 229 | 986588 | `25901f4b80de3f6c052f026b5184fca4057dfc6be4ca2be61e2c41f18efe92b7` |

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

CNN/DailyMail-derived files preserved from Demos/SFT_CNN_DailyMail; source attribution is not a license grant.

Confirm that your intended use complies with the original dataset terms, privacy obligations, and applicable
policies. Repository inclusion and source attribution do not grant additional rights. Review samples for PII,
secrets, harmful content, domain risk, and representation gaps before adapting this material.

Do not edit these preserved files to work around a preprocessing or safety failure. Instead, create a new,
versioned dataset with documented transforms, filters, approvals, and fresh hashes.
