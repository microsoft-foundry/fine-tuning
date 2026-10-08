# Preserved code-distillation data

`training_data.jsonl` (1,576 rows) and `validation_data.jsonl` (83 rows) are
byte-for-byte copies of the corresponding
`Demos/NL_to_Python_Distillation` files. `SOURCE_DATA_SHA256SUMS.txt` records
their source paths and byte hashes.

These exact source files, including generated code examples, are the training
contract. The cookbook does not substitute a smaller sample or rerun generation,
filtering or judging. It checks JSON/chat shape and exact row overlap before
upload. The original workflow remains in
`Demos/NL_to_Python_Distillation/Text_to_Python_Fine_Tuning.ipynb`.

The preserved source training recipe is 1 epoch, batch size 1 and learning-rate
multiplier 1.3 with supervised `GlobalStandard` training.
