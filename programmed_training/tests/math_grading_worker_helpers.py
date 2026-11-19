"""Importable, finite helpers for the isolated math-grading regressions."""

import os
import threading
from pathlib import Path


def grade_after_marker(given_answer, ground_truth, *, marker_path):
    from programmed_training.recipes.math_rl.math_grading import grade_answer

    Path(marker_path).write_text(str(os.getpid()), encoding="utf-8")
    threading.Event().wait(0.35)
    return grade_answer(given_answer, ground_truth)