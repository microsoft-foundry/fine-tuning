# Notice

These are preserved, synthetic natural-language-to-Python training examples,
not a newly generated or independently validated dataset.

## Dataset Summary

Each JSONL record contains a system instruction, a user programming task and
an assistant code response. The maintained code-distillation
[source manifest](../../../managed_training/05-data-and-distillation/02-code-distillation/data/SOURCE_DATA_SHA256SUMS.txt)
records the following original assets:

| Bundled file | Source rows | CRLF-to-LF SHA-256 |
| --- | ---: | --- |
| [chat_train.jsonl](chat_train.jsonl) | 1,576 | `af16c47387056f9e597deeef495d444702f125d4dd7ab91dee4e4836296cc913` |
| [chat_validation.jsonl](chat_validation.jsonl) | 83 | `558f93d59fe1d9c063f0922aec264aad3f26e96561c9ab5bcc514afb30265756` |

The sample Git blobs use LF while the original archived demo files use CRLF.
They contain identical records, not identical raw file bytes on every checkout.
The audit normalizes only physical CRLF to LF, records each file's actual raw
hash, and retains the historical archive digests in the source manifest and
[validation-status.json](../../validation-status.json). Curation still retains
each selected input line's exact bytes and records its raw-line hash.

These are the **same source assets**, not a distinct 85/4 dataset. The source
paths were `Demos/NL_to_Python_Distillation/training_data.jsonl` and
`Demos/NL_to_Python_Distillation/validation_data.jsonl`. The historical generation
notebook is not included. Its generator, judge scores, diversity coverage and
semantic correctness cannot be independently established from these chat files.
Do not infer that all examples passed AST parsing or a quality threshold.

## Syntax audit and explicit curation

**Do not treat the preserved files as syntax-validated training targets.** The
offline audit confirmed 95 training and 5 validation AST failures after removing
conventional surrounding fences. Preparation tests produced 1,481 training and
78 validation records, retaining accepted source lines unchanged. The preserved
source bytes are not rewritten; prepared files are created explicitly on demand.

[../../dataset_integrity.py](../../dataset_integrity.py) provides offline
`audit-code` and explicit `curate-code --output-dir <new-directory>` commands.
Run it with Python from the repository root. It:

1. Checks both sample assets against the source counts/hashes above **and** the
  maintained code-distillation copies; mismatches stop preparation.
2. Parses each assistant response with `ast.parse`, targeting Python 3.11 syntax.
  Only a single enclosing backtick fence labelled `python`, `py`, or unlabelled
  is stripped for this check. Prose, interior fences and broken code are not
  guessed away. Code is never imported or executed.
3. Writes accepted source records unchanged into separate curated files. Invalid
  records are excluded only through this explicit command, never silently on
  upload. It does not fabricate or repair Python answers.
4. Writes a manifest containing every source line number and raw-line SHA-256,
  the accept/reject decision, syntax diagnostics, source/output counts and byte
  hashes, parser version and the curation rule. The original files remain intact.

AST acceptance is **not** task correctness, dependency availability, safety or
runtime validation. Inspect the generated manifest and run the offline tests
before adopting any curated output. The manifest records exact output hashes
for the selected parser and verified source bytes.

## Format

Standard chat messages JSONL — each line is a JSON object with a `messages` array:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant that writes clean, well-structured Python code to solve the user's request."},
    {"role": "user", "content": "Write a function that..."},
    {"role": "assistant", "content": "def solve(...):\n    ..."}
  ]
}
```

## Related Demo

The maintained [code-distillation demo](../../../managed_training/05-data-and-distillation/02-code-distillation/README.md)
uses these same original, hash-pinned assets. Its byte/chat guards do not prove
Python syntax or task correctness, and it does **not** automatically consume
curated outputs. Changing that demo's training contract requires a separate,
reviewed update by its owner. It covers upload, training and service metrics,
not data generation, deployment or task-quality evaluation.

## Languages

The text and code in the dataset are in English/Python. The associated BCP-47 code is `en`.
