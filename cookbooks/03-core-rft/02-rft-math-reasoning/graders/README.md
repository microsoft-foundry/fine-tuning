# Math reasoning grader

`boxed_math_grader.py` is the extracted and hardened version of the grader embedded in the source notebook. It preserves last-box selection, balanced-brace parsing, LaTeX answers, and order-insensitive comma-separated answer lists while adding explicit whitespace normalization.

The notebook tests missing boxes, unclosed boxes, earlier-answer leakage, a wrong final box, and answer text placed outside the final box. `hashes.json` pins the exact canonical implementation used by this cookbook.

