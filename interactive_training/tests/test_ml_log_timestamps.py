"""Regression tests for timestamped cookbook log handlers."""

import logging
import re

from interactive_training.utils.ml_log import configure_logging_module

_TS_PREFIX = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")


def test_configure_logging_module_file_output_is_timestamped(tmp_path):
    log_path = tmp_path / "run.log"
    root = configure_logging_module(str(log_path))
    try:
        logging.getLogger("demo").info("hello world")
        for handler in root.handlers:
            handler.flush()
        line = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    finally:
        for handler in list(root.handlers):
            handler.close()
            root.removeHandler(handler)

    assert _TS_PREFIX.match(line)
    assert "demo:" in line and "hello world" in line


def test_both_handlers_include_asctime(tmp_path):
    root = configure_logging_module(str(tmp_path / "fmt.log"))
    try:
        assert root.handlers
        for handler in root.handlers:
            fmt = handler.formatter._fmt if handler.formatter else ""
            assert "%(asctime)s" in fmt
    finally:
        for handler in list(root.handlers):
            handler.close()
            root.removeHandler(handler)
