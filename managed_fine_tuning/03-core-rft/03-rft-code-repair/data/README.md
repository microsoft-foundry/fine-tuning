# Code-repair RFT data

The dataset contains Python code-repair tasks for reinforcement fine-tuning:

| Split | File | Rows | Bytes |
|---|---|---:|---:|
| Training | [code-repair-rft-training.jsonl](preserved/code-repair-rft-training.jsonl) | 50 | 47,678 |
| Validation | [code-repair-rft-validation.jsonl](preserved/code-repair-rft-validation.jsonl) | 15 | 15,177 |

[hashes.json](hashes.json) records the sample file paths, SHA-256 values, and
counts. The files are UTF-8 JSONL with LF line endings, no BOM, and no blank
records. Git text conversion is disabled so checkout bytes stay unchanged.

Every record has exactly four fields:

| Field | Used by |
|---|---|
| `messages` | Model input: a developer instruction and a user repair request with Python code. |
| `reference_patch` | Grader-only reference unified diff. |
| `test_expectations` | Grader-only natural-language expectations, not executable tests. |
| `requirements` | Grader-only constraints and edge cases. |

The developer instruction requests JSON with `patch`, `explanation`, and
`tests`; `patch` must be a unified diff. There is no stored assistant response:
RFT generates candidate repairs and rewards them rather than imitating the
reference patch. The grader's `{{sample.output_text}}` receives the complete
candidate response as text rather than parsed JSON fields.

The repository's offline tests check every record, exact hashes, duplicate
records, duplicate user prompts, and train/validation isolation. The notebook
loads the sample files and displays an example. These checks do not prove that
the reference patches are correct, the prompt states every graded requirement,
or the model grader recognizes all valid alternative repairs. Review those
relationships when adapting the task.

When adapting the task to another dataset, review its license, usage rights,
and privacy requirements before uploading. Keep custom inputs separately
versioned using [the adaptation guide](../../../ADAPT_YOUR_DATA.md).

The validation split is used by the RFT job, not reserved as an independent
test set. Use a separate test set for independent evaluation.
