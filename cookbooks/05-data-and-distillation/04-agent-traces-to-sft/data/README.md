# Preserved trace-distillation data

The exact 100 transformed traces and their seed-42 80/10/10 split are retained
as customer training data. `SOURCE_DATA_SHA256SUMS.txt` records source paths and
hashes. The cookbook reads only these bundled files, not ignored source-demo
output directories. Test data are retained for lineage and never uploaded.

The source workflow in `Demos/TracesDistillation/notebook.ipynb` requests
approximately 100 traces from the configured source agent/time window using
the Foundry Data Generation API. Its transformation:

1. Removes overlapping snapshots by first-occurrence message deduplication using role, content, tool-call ID, and function-call identity.
2. Discards fragments without assistant tool-call supervision.
3. Merges adjacent assistant tool-call turns and omits assistant tool-call content.
4. Restores missing function-call IDs deterministically as `call_trace_<row>_<message>_<call>`, associating tool replies in order.
5. Restores the exact Zava system prompt, tool catalog and `parallel_tool_calls=true`.

The resulting rows are serialized unchanged. Python `random.Random(42)`
shuffles row indices, with the first 80 going to `train.jsonl`, the next 10 to
`val.jsonl`, and the final 10 to `test.jsonl`. The cookbook verifies this exact
membership and the prompt/tools fixture agreement read-only. It does not
repeat the transform or silently repair invalid rows.

The source implementation can skip orphan tool-response rows during
transformation; that historical recipe is not a new fallback in the cookbook.
Only the already transformed, hash-guarded data are used here. No hosted agent
calls or fresh generation are needed.

The preserved supervised training recipe is 3 epochs, learning-rate multiplier
1.0 and batch size 1 with `GlobalStandard`.
