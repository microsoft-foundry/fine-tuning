# Countdown data

`preserved/` contains byte-identical copies of every JSONL dataset from `Demos/RFT_Countdown/data`. Names identify the split, role, and row count. The canonical notebook uses the 100-row RFT training set, 50-row RFT validation set, and independent 100-row evaluation set; the smaller and pre-conversion splits remain available for reproducibility and comparison.

`hashes.json` is the integrity contract. Do not reformat these JSONL files: even semantically equivalent formatting changes their hashes.

