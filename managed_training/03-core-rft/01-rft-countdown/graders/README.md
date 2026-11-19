# Countdown graders

- `countdown_python_grader.py` mirrors the deterministic arithmetic checks used by the Python training grader.
- `preserved/countdown-model-score-grader.json` and its prompt preserve the exact source content for the model/score variant.
- The preserved solver prompt and response schema define the shared output contract.

The notebook validates both successful training-grader contracts before submission.

Both hash manifests use canonical LF byte counts and hashes, so platform checkout line endings do not alter the integrity contract.

The deterministic standalone and embedded implementations share the same source
and reject expressions longer than 4096 characters, ASTs larger than 256 nodes,
or AST depth greater than 64 before recursive arithmetic evaluation. Parser or
evaluator recursion failures earn zero, as do nonfinite values; valid arithmetic,
reward tiers, and exact input-number multiplicities are unchanged.

The modified standalone entry in [hashes.json](hashes.json) is explicitly marked
as awaiting canonical-LF byte-count and SHA-256 recomputation. Its old pins have
not been fabricated or silently accepted: the existing integrity test must pass
after the parent validation step refreshes them. Preserved artifacts are unchanged.
