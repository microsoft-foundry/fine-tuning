# Math reasoning grader

`boxed_math_grader.py` is the extracted and hardened version of the grader embedded in the source notebook. It preserves last-box selection, balanced-brace parsing, LaTeX answers, and order-insensitive comma-separated answer lists while adding explicit whitespace normalization.

Only top-level commas separate unordered answers. Parentheses, brackets, and
braces keep coordinate components and nested LaTeX expressions grouped, so
reordering whole solution pairs is allowed but cross-pairing their components
does not receive credit. The offline cookbook-contract suite includes the
adversarial training-row-60 case and independent checks of curated reference
answers. This remains an exact-text grader, not a symbolic-equivalence checker.

The notebook tests missing boxes, unclosed boxes, earlier-answer leakage, a wrong final box, and answer text placed outside the final box. `hashes.json` pins the exact canonical implementation used by this cookbook.

