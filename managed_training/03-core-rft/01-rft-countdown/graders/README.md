# Countdown graders

- `countdown_python_grader.py` mirrors the deterministic arithmetic checks used by the Python training grader.
- `preserved/countdown-model-score-grader.json` and its prompt preserve the exact source content for the model/score variant.
- The preserved solver prompt and response schema define the shared output contract.

The notebook validates both successful training-grader contracts before submission.

Both hash manifests use canonical LF byte counts and hashes, so platform checkout line endings do not alter the integrity contract.
