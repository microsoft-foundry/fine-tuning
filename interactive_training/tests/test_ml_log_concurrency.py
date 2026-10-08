"""Concurrency regression tests for JSONL metric logging."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import chz

from interactive_training.utils.ml_log import JsonLogger, dump_config


class _RuntimeHandle:
    def __deepcopy__(self, memo):
        raise TypeError("runtime handles cannot be copied")


@chz.chz
class _ConfigWithRuntimeHandle:
    handles: list[object]


def test_dump_config_does_not_copy_runtime_handles():
    config = _ConfigWithRuntimeHandle(handles=[_RuntimeHandle()])

    assert dump_config(config) == {"handles": [{}]}


def test_json_logger_concurrent_appends_produce_complete_records(tmp_path):
    logger = JsonLogger(tmp_path)
    record_count = 200

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(
            executor.map(
                lambda step: logger.log_metrics(
                    {"payload": "x" * 4096, "record": step},
                    step=step,
                ),
                range(record_count),
            )
        )

    rows = [
        json.loads(line)
        for line in logger.metrics_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == record_count
    assert {row["record"] for row in rows} == set(range(record_count))
