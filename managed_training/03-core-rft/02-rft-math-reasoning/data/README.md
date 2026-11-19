# Advanced math RFT data

The preserved source contains 100 training and 10 validation records with
separate, non-overlapping problem prompts. Three preserved training prompts are
explicitly labeled local, self-contained adaptations; they are not recovered
upstream figures. The notebook now uploads a separately prepared 100/10 runtime
split with one additional, explicitly labeled answer-only adaptation of source
training row 1. Preserved bytes, row numbers, and source integrity pins are unchanged.

The source dataset is [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/tree/e4e141ec9dea9f8326f4d347be56105859b2bd68), revision `e4e141ec9dea9f8326f4d347be56105859b2bd68`, `default/train` (reported upstream license: Apache-2.0). The [manifest](hashes.json) records source attribution, local curation, row counts, canonical-LF byte counts, and SHA-256 integrity values. That attribution does not claim that upstream contains these corrections/adaptations or establish additional provenance for the missing figures.

Reference answers include both equation roots (training 8), all four real pairs (60), both side-length branches (90; discard zero when `a=b`), and both sphere radii (92). Training 58 uses real parameters with complex roots allowed, yielding `p = ±2^(1/8)`; training 17 uses integers `k ≥ 1`. Training 71 specifies one valid magic square; the text grader does not recognize equivalent rotations/reflections. Multipart prompts use the final-subproblem reference convention.

The manifest also records these additional curated training-answer corrections;
these five problem prompts and answers are retained unchanged by the later
curation below, and the 100/10 split is unchanged:

| Training row | Reference answer | Check |
|---|---|---|
| 68 | `-2, -1, 4, 5` | For `abs((2x-9)(2x+3))` to be prime, one integer factor must have magnitude 1. |
| 74 | `8, 9, 10, 11, 12, 13, 14, 15` | Eight consecutive integers beginning at 8 sum to 92. |
| 76 | `729` | Alternating squared/cubed digit sums reach 324 at odd steps and 729 at even steps. |
| 79 | `9, 23, 37` | Enumerate equal-sum partitions of the four whole buns. |
| 100 | `a=5, b=3, c=7` | The unique distinct-digit solution is `533 * 7 = 3731`. |

These are explicit answer curation, not a claim that the pinned upstream
revision already contains the corrected labels. The offline cookbook-contract
tests check the five labels, independent calculations, and artifact hashes.

## Self-contained prompt curation and complete lamp counts

Rows are one-based positions in the training file. The original problem text
for each adapted row is retained in the manifest, but is not an upload input.
The source figure/flowchart is absent in each case. The added geometry and
program choices are local task definitions, **not** reconstructions of an
original diagram, recovered source text, or new upstream-license evidence.
All other prompts remain unchanged.

| Training row | Change | Independent verification |
|---|---|---|
| 14 | Keep the lamp prompt; require both counts in its stated order: `(20, 30)` for 40-watt and 75-watt lamps. | `x+y=50`, `40x+75y=3050` uniquely give `x=20`, `y=30`. A lone `30` is incomplete. |
| 34 | Replace the absent-figure reference with coordinates `C=(0,0)`, `A=(1,0)`, `B=(0,1)` and explicit sides for the two equilateral constructions. | With `t=(1+sqrt(3))/2`, `D=(t,t)` and `E=(-1/2,1+sqrt(3)/2)`, the requested length satisfies `BE^2=1`. |
| 55 | Replace the unspecified flowchart with the newly defined program `Y=10X+15` and explicitly uniform `X` on `[1,9]`. | `Y>=55` iff `X>=4`, giving probability `5/8`, option `D`. This transformation is chosen for the adaptation, not inferred from the source. |
| 96 | Specify cyclic trapezoid vertices, `F` beyond `C` on ray `DC`, and the right angle at `F` with hypotenuse `DE`. | Set `D=(0,0)`, `C=(10,0)`, `A=(3,4)`, `B=(7,4)`. Then `E=2B-D=(14,8)`, `F=(14,0)`, so `CF=4`, option `D`. |

The three adapted prompts retain their prior reference values, but now provide
all assumptions needed to derive them. Focused offline tests verify these
constructions, the exact adaptation labels, and all 110 boxed reference
answers. The existing coordinate-grader and five answer-correction checks are
not relaxed. Historical metrics and charts are not recomputed or relabeled as
results for the adapted data.

Hash, schema, split, and boxed-answer checks do not prove every answer is mathematically correct or every equivalent expression receives credit.

## Runtime row 1 curation and independent proof

The **full exact source prompt** is retained in
[preserved training row 1](preserved/openr1-math-training-rft-100.jsonl#L1).
It defines a scalene disguisable triangle, then asks **(a) for all its side
lengths at minimum integral perimeter and (b) for a proof that a primitive
integral example has at least two square side lengths**. The stored `9` gives
neither all the sides nor the proof. Feeding `9` back to the same text grader is
only a label/parser check, not mathematical verification. The general historical
final-subproblem convention cannot make that label complete.

Here is an independent derivation for both subparts. Write the sorted sides as
$0<a<b<c$. Similarity to the larger triangle in the definition preserves order,
so a scale factor $r>1$ satisfies $ra=b$ and $rb=c$. Thus $b^2=ac$.
Write $r=n/m$ in lowest terms, with positive integers $m<n$. The integrality of
$b=an/m$ implies $a=mt$, and that of $c=tn^2/m$ implies $t=km$. Hence every
integral example has

$$
(a,b,c)=k(m^2,mn,n^2),\qquad \gcd(m,n)=1,
\qquad m^2+mn>n^2.
$$

Conversely these conditions give a scalene triangle, and scaling by $n/m$ gives
its two shorter sides equal to the previous two longer sides, as required.
For $m=1$, every $n\ge2$ fails $1+n>n^2$. For $m=2$, $n=3$ works and gives
$(4,6,9)$, perimeter $19$. For $m\ge3$, $n\ge m+1$ gives perimeter at least
$m^2+m(m+1)+(m+1)^2\ge37>19$ even with $k=1$; $k>1$ increases it. The unique
sorted minimum is therefore **$(4,6,9)$**, not the single largest side `9`.

For part (b), the three integers $m^2,mn,n^2$ have gcd 1. A triangle with
$k>1$ has a smaller integral similar triangle obtained by division by $k$,
contrary to the hypothesis. Therefore $k=1$ and the shortest and longest sides
are the perfect squares $m^2$ and $n^2$. Conversely an integral scaling of a
gcd-1 triple must have an integral scale factor (Bezout's identity); no smaller
positive integral scaling exists. This proves the primitive classification,
not merely a search over a few examples.

The boxed-answer grader cannot assess that proof. Accordingly,
[prepare_math_data.py](../prepare_math_data.py) uses the versioned policy
`openr1-row1-answer-only-v1` to adapt **only runtime training row 1** to part (a),
explicitly excluding part (b) from the graded prompt and requiring the full
boxed tuple `(4, 6, 9)` in increasing order. This is not a reconstructed source
answer/proof, an upstream correction claim, or proof-quality reward. The full
two-part prompt, incomplete source label `9`, earlier curation, and original
manifest artifact hashes remain immutable. No source row moves between splits;
the runtime split remains 100/10 with zero prompt overlap. The other 99 training
lines and the validation file retain their canonical-LF bytes exactly.

Preparation validates the preserved manifest first, then writes separate inputs
and a provenance report only under ignored outputs. The report records actual
canonical-LF source/runtime byte counts, row counts, and SHA-256 values computed
from the prepared bytes; no runtime hash is invented or substituted into the
preserved source manifest. A different existing prepared file fails closed
rather than being overwritten. Submission equivalence binds the new data, so
old job state must be retained/reconciled, never relabeled to force reuse.

Offline regressions independently enumerate all integral triangles of perimeter
at most 19 using $b^2=ac$, verify the unique minimum, check primitive square-side
structure, and verify that only the declared runtime line changes. These
mathematical assertions do not obtain their expected values from the stored
reference or grader result.

## Dataset files

These are immutable preparation sources, **not direct upload inputs**:

- [Training data](preserved/openr1-math-training-rft-100.jsonl): 100 training prompts and reference answers.
- [Validation data](preserved/openr1-math-validation-rft-10.jsonl): 10 validation prompts and reference answers.

Each record contains user messages plus an `answer` field used by the grader. It intentionally has no assistant message: reinforcement fine-tuning explores responses and receives reward rather than imitating a stored reasoning trace.

The preparation helper returns the actual training and validation upload paths
under ignored outputs, plus its preparation report. The notebook uses those
same paths for inspection, content-addressed naming, hashing, and submission.

Integrity checks and remote content names use the exact same CRLF-to-LF byte
normalization as upload. Uploads send canonical LF bytes in memory, independent
of Windows checkout settings; JSON string escapes inside the records are not
changed. Download verification checks the service's raw byte count against the
raw download, then compares the canonical content/hash with the manifest. Thus
an older CRLF upload can be verified by content without pretending its raw byte
count is the canonical-LF count. No Git checkout conversion is required.
