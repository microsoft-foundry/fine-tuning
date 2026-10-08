# Preserved sarcasm data

`SOURCE_DATA_SHA256SUMS.txt` records exact source bytes from
`Demos/DistillingSarcasm`. `qa.jsonl` contains 500 source questions;
`baseline.jsonl` is retained source data but is not consumed by this training-only
notebook.

The source recipe shuffles the questions with Python `random.Random(42)`, selects
the first 50, uses the first 40 as teacher-label candidates and reserves the last
10. The original teacher/judge pass accepted 26 labels, retained here unchanged
as 20 `sarcasm-training.jsonl` rows and 6 `sarcasm-validation.jsonl` rows. Every
accepted example uses `Clippy is a factual chatbot that is also sarcastic.`
as its system prompt. Labels are neither regenerated nor normalized.

The notebook checks hashes, unique accepted questions, membership in the 40
candidates and separation from the 10 held-out questions. The original source
workflow remains in `Demos/DistillingSarcasm/sarcasm.ipynb`; no source-demo runtime
files are required to execute the cookbook.
