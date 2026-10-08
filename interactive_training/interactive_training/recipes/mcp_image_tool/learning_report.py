"""Report measured learning separately from successful training execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def build_learning_report(
    metrics: Sequence[dict[str, Any]], *, expected_episodes: int
) -> dict[str, Any]:
    evaluations = [row for row in metrics if "test/env/all/correct" in row]
    if len(evaluations) < 2:
        return {"accepted": False, "reason": "Initial and final evaluations are required"}

    initial, final = evaluations[0], evaluations[-1]
    before_rl = next(
        (row for row in evaluations if row.get("stage") == "before_rl"),
        None,
    )
    initial_accuracy = initial["test/env/all/correct"]
    final_accuracy = final["test/env/all/correct"]
    complete = all(
        row.get("test/env/all/total_episodes") == expected_episodes
        for row in (initial, final)
    )
    updated = final["step"] > initial["step"]
    post_sft_accuracy = (
        before_rl["test/env/all/correct"] if before_rl is not None else None
    )
    rl_accuracy_gain = (
        final_accuracy - post_sft_accuracy
        if post_sft_accuracy is not None
        else None
    )
    report = {
        "accepted": bool(
            complete
            and updated
            and initial_accuracy < 0.1
            and final_accuracy > 0.3
            and rl_accuracy_gain is not None
            and rl_accuracy_gain > 0.0
        ),
        "initial_step": initial["step"],
        "final_step": final["step"],
        "initial_accuracy": initial_accuracy,
        "final_accuracy": final_accuracy,
        "accuracy_gain": final_accuracy - initial_accuracy,
        "post_sft_accuracy": post_sft_accuracy,
        "rl_accuracy_gain": rl_accuracy_gain,
        "complete_evaluation": complete,
        "expected_episodes": expected_episodes,
        "target": (
            "initial accuracy < 0.10, final accuracy > 0.30, and RL accuracy "
            "strictly improves"
        ),
        "by_color_count": {},
    }
    for count in (1, 2, 3):
        key = f"test/env/colors_{count}/correct"
        report["by_color_count"][str(count)] = {
            "initial_accuracy": initial.get(key),
            "final_accuracy": final.get(key),
        }
    return report