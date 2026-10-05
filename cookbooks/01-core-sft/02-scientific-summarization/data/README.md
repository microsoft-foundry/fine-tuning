# Data provenance: Domain SFT: Scientific Summarization

These files were copied without transformation. Their byte content and split membership are preserved.

| Target file | Original source | Split | Rows | Bytes | SHA-256 |
|---|---|---|---:|---:|---|
| `scientific-summarization-train.jsonl` | `Demos/SFT_PubMed_Summarization/training.jsonl` | training | 1000 | 19203661 | `7e578bf92c922f849853e79e12f1bf23de9b24b56d64dea261093b826a1d5f8e` |
| `scientific-summarization-validation.jsonl` | `Demos/SFT_PubMed_Summarization/validation.jsonl` | validation | 100 | 2248766 | `17349f913142c46e4c28cb10e5d326430d37567fa3ecabc851ccc09f00e08494` |

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

PubMed-derived files preserved from Demos/SFT_PubMed_Summarization; source attribution is not a license grant.

Confirm that your intended use complies with the original dataset terms, privacy obligations, and applicable
policies. Repository inclusion and source attribution do not grant additional rights. Review samples for PII,
secrets, harmful content, domain risk, and representation gaps before adapting this material.

Do not edit these preserved files to work around a preprocessing or safety failure. Instead, create a new,
versioned dataset with documented transforms, filters, approvals, and fresh hashes.
