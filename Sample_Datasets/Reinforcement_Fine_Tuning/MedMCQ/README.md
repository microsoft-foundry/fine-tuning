# Notice

These are preserved samples curated from [MedMCQA](https://github.com/medmcqa/medmcqa),
along with a sample grader. **The bundled validation file is not usable with
that grader: all 6,150 records lack `reference_answer`.** The offline audit
verified every row. No labels have been guessed or reconstructed.

The grader reads `item.reference_answer`; missing labels are not empty answers
and must not be treated as successful validation. Do not upload the validation
file, use it for RFT, or claim held-out grading until all labels have been
restored from a verified source and checked. JSON validity alone is insufficient.

For a ready-to-prepare smaller example, see the
[Countdown dataset and demo](../../../managed_training/03-core-rft/01-rft-countdown/README.md).
This medical dataset is also large enough to cause RFT service timeouts.

## Fail-closed checks and label restoration

[../../dataset_integrity.py](../../dataset_integrity.py) includes an offline
`check-medmcq` command that checks both bundled files and exits unsuccessfully
when any answer is missing or outside `A`–`D`.

The official [MedMCQA data description](https://github.com/medmcqa/medmcqa#data-download-and-preprocessing)
links the original archive and documents the original `question`, ordered
`opa`/`opb`/`opc`/`opd`, `id`, `choice_type` and **one-based** `cop` fields.
Its published split table alone does not establish that this bundled file is
the same split; the question text **and all four options in order** must match.
The publisher explicitly states that the official 6,150-question test set's
ground truth is withheld. Its public labelled development set contains 4,183
questions; replacing one with the other is a new split, not restoration of these
missing labels. Some mirrors also re-encode answers as zero-based classes;
those are not inputs to the original-format preparation path below.

An authorized maintainer can prepare labels without generating any answers:

1. Obtain an authentic original labelled JSONL from the publisher's archive,
	independently verify its origin, and retain its immutable revision/archive
	identifier and exact SHA-256. No downloader or credentialed lookup is provided.
2. Run `prepare-medmcq --labels <local-original-jsonl> --labels-sha256 <64-hex>
	--source-url <publisher-url> --source-revision <immutable-id>
	--output-dir <new-directory>` with the offline preparation script from the
	repository root. The supplied digest pins bytes; it does not independently
	authenticate the publisher.
3. The script requires an exact rendered question/options match to **one** source
	record with a stable ID, `choice_type="single"` and integer `cop` in `1..4`.
	It refuses unmatched/duplicate/ambiguous records, conflicting existing answers,
	multi-answer labels and malformed data. It never matches by row position,
	fuzzy text, medical knowledge or model predictions.
4. Only after **every** target row matches does it write a separate labelled
	validation file and a manifest mapping every target line/hash to its source
	line/ID/hash and label. Input/output counts and byte hashes are recorded.
	The bundled source files are not overwritten.

No authentic pinned label corpus was available in the inspected checked-in
copies; those copies repeat the missing-label problem. Label reconstruction is
therefore **blocked**, not completed. Unmatched data require obtaining the correct
source, not weakening the exact-match check or fabricating a fallback answer.

## Dataset Summary

MedMCQA is a large-scale, Multiple-Choice Question Answering (MCQA) dataset designed to address real-world medical entrance exam questions.

MedMCQA has more than 194k high-quality AIIMS & NEET PG entrance exam MCQs covering 2.4k healthcare topics and 21 medical subjects are collected with an average token length of 12.77 and high topical diversity.

Each sample contains a question, correct answer(s), and other options which require a deeper language understanding as it tests the 10+ reasoning abilities of a model across a wide range of medical subjects & topics. A detailed explanation of the solution, along with the above information, is provided in this study.

MedMCQA provides an open-source dataset for the Natural Language Processing community.
It is expected that this dataset would facilitate future research toward achieving better QA systems.
The dataset contains questions about the following topics:

- Anesthesia
- Anatomy
- Biochemistry
- Dental
- ENT
- Forensic Medicine (FM)
- Obstetrics and Gynecology (O&G)
- Medicine
- Microbiology
- Ophthalmology
- Orthopedics
- Pathology
- Pediatrics
- Pharmacology
- Physiology
- Psychiatry
- Radiology
- Skin
- Preventive & Social Medicine (PSM)
- Surgery

## Grader
The [string check grader](stringcheck_grader.json) compares the predicted answer
with the reference answer. Both training and validation records must first pass
the label preflight above; the grader cannot recover missing answers.
