# Advanced math RFT data

These preserve the exact existing curated OpenR1-Math-220k records:

- `openr1-math-training-rft-100.jsonl`: 100 training prompts and reference answers.
- `openr1-math-validation-rft-10.jsonl`: 10 validation prompts and reference answers.

Each record contains user messages plus an `answer` field used by the grader. It intentionally has no assistant message: reinforcement fine-tuning explores responses and receives reward rather than imitating a stored reasoning trace.

See `hashes.json` for source mapping and canonical LF byte counts and SHA-256 integrity values. Checks normalize checkout line endings without rewriting or changing any dataset record.
