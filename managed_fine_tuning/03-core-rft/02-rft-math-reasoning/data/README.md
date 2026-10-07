# Advanced math RFT data

The notebook uses 100 training and 10 validation records under [preserved](preserved/), with separate, non-overlapping problem prompts.

The source dataset is [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/tree/e4e141ec9dea9f8326f4d347be56105859b2bd68), revision `e4e141ec9dea9f8326f4d347be56105859b2bd68`, `default/train` (Apache-2.0). The [manifest](hashes.json) records provenance, row counts, canonical-LF byte counts, and SHA-256 integrity values.

Reference answers include both equation roots (training 8), all four real pairs (60), both side-length branches (90; discard zero when `a=b`), and both sphere radii (92). Training 58 uses real parameters with complex roots allowed, yielding `p = ±2^(1/8)`; training 17 uses integers `k ≥ 1`. Training 71 specifies one valid magic square; the text grader does not recognize equivalent rotations/reflections. Multipart prompts use the final-subproblem reference convention.

Hash, schema, split, and boxed-answer checks do not prove every answer is mathematically correct or every equivalent expression receives credit.

## Dataset files

These are the notebook's upload inputs:

- [Training data](preserved/openr1-math-training-rft-100.jsonl): 100 training prompts and reference answers.
- [Validation data](preserved/openr1-math-validation-rft-10.jsonl): 10 validation prompts and reference answers.

Each record contains user messages plus an `answer` field used by the grader. It intentionally has no assistant message: reinforcement fine-tuning explores responses and receives reward rather than imitating a stored reasoning trace.

Integrity checks normalize checkout line endings without modifying dataset records.
