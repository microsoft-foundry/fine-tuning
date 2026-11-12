"""Render local RL numerics logs as a self-contained HTML evidence report."""

from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


DEPTH_BINS = ("0_8k", "8_16k", "16_32k", "32_64k", "over_64k")
COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7")


def _finite_metric(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def chart(
    title: str, series: list[tuple[str, list[tuple[float, float]]]],
    *, depth: bool = False,
) -> str:
    points = [point for _, values in series for point in values]
    if not points:
        return f"<h2>{html.escape(title)}</h2><p>No recorded data; not reconstructed.</p>"
    if any(not math.isfinite(value) for point in points for value in point):
        raise ValueError(f"Non-finite chart data: {title}")
    logarithmic = all(y > 0 for _, y in points)
    transform = math.log10 if logarithmic else float
    ymin, ymax = min(transform(y) for _, y in points), max(transform(y) for _, y in points)
    padding = max((ymax - ymin) * 0.1, 0.1 if logarithmic else 0.0001)
    ymin, ymax = ymin - padding, ymax + padding
    xmax = 4 if depth else max(1, max(x for x, _ in points))

    def xy(x: float, y: float) -> tuple[float, float]:
        return 80 + x / xmax * 640, 275 - (transform(y) - ymin) / (ymax - ymin) * 235

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 {max(360, 330 + len(series) * 14)}" role="img">',
        f"<title>{html.escape(title)}</title>",
        '<path d="M80 40 V275 H720" fill="none" stroke="#333"/>',
    ]
    for index in range(5):
        value = ymin + (ymax - ymin) * index / 4
        y = 275 - index * 235 / 4
        label = 10 ** value if logarithmic else value
        elements.append(
            f'<text x="72" y="{y + 4:.2f}" text-anchor="end">{label:.3g}</text>'
        )
    labels = ["0-8K", "8-16K", "16-32K", "32-64K", "&gt;64K"] if depth else None
    for index in range(5):
        x = 80 + index * 160
        label = labels[index] if labels else f"{xmax * index / 4:g}"
        elements.append(f'<text x="{x}" y="298" text-anchor="middle">{label}</text>')
    for index, (name, values) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        if not depth:
            path = " ".join(
                f"{'M' if i == 0 else 'L'}{x:.2f},{y:.2f}"
                for i, (x, y) in enumerate(xy(x, y) for x, y in values)
            )
            elements.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in values:
            px, py = xy(x, y)
            if depth:
                px += (index - (len(series) - 1) / 2) * 18
                elements.append(
                    f'<path d="M{px:.2f} 275 V{py:.2f}" stroke="{color}" stroke-width="14"/>'
                )
            elements.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3" fill="{color}"/>')
        elements.append(
            f'<text x="80" y="{320 + index * 14}" fill="{color}">{html.escape(name)}</text>'
        )
    elements.append("</svg>")
    scale = "logarithmic" if logarithmic else "linear (includes zero or negative estimates)"
    return f"<h2>{html.escape(title)}</h2><p>Y scale: {scale}.</p>" + "".join(elements)


def build_report(log_paths: list[Path]) -> str:
    panels: list[str] = []
    step_series = []
    depth_series = []
    columns = (
        "optim/kl_sample_train_v1", "optim/kl_sample_train_v2",
        "optim/kl_sample_train_max_action_position",
        "env/all/reward/total", "env/all/exact_match", "env/all/parse_error",
        "env/all/context_overflow", "env/all/ac_tokens_frac_at_max",
        "env/all/by_group/frac_mixed", "env/all/max_context_tokens",
        "env/all/max_sampled_prompt_tokens", "env/all/target_context_tokens",
        "env/all/calibrated_context_tokens",
        "skyrl.ai/grad_norm",
    )
    for path in log_paths:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        meta = json.loads((path / "run_meta.json").read_text(encoding="utf-8"))
        rows: list[dict[str, Any]] = [
            json.loads(line) for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError(f"No completed step metrics in {path}")
        steps = [row["step"] for row in rows]
        if steps != sorted(set(steps)):
            raise ValueError(f"Step indices must be unique and increasing in {path}")
        name = f"{config['model_name']} / {meta['azure']['session_id']}"
        step_series.append((name, [
            (row["step"], row["optim/kl_sample_train_v1"])
            for row in rows if _finite_metric(row.get("optim/kl_sample_train_v1"))
        ]))
        depth_values = []
        initial = next((row for row in rows if row["step"] == 0), None)
        if initial is not None:
            for index, bucket in enumerate(DEPTH_BINS):
                prefix = f"optim/kl_sample_train_by_depth/{bucket}"
                tokens = initial.get(f"{prefix}/tokens")
                if (
                    _finite_metric(tokens) and tokens > 0
                    and _finite_metric(initial.get(f"{prefix}/v1"))
                ):
                    depth_values.append((index, initial[f"{prefix}/v1"]))
        depth_series.append((name, depth_values))
        completed = len(rows)
        durations = [row["time/total"] for row in rows if _finite_metric(row.get("time/total"))]
        timing = "Remaining-time estimate unavailable: no configured step limit or complete per-step timing."
        target_steps = config.get("max_steps")
        if len(durations) == completed and target_steps is not None:
            elapsed = sum(durations)
            mean = elapsed / completed
            remaining = max(0, target_steps - rows[-1]["step"] - 1)
            timing = (
                f"Measured completed-step time: {elapsed / 60:.1f} min; "
                f"mean: {mean / 60:.1f} min/update; "
                f"estimated remaining step time: {remaining * mean / 3600:.2f} h. "
                "This excludes dataset preparation, unmeasured setup/finalization and capacity wait; "
                "the estimate assumes unchanged workload and resource contention."
            )
        wall_time = ""
        timing_path = path / "timing.json"
        if timing_path.exists():
            saved_timing = json.loads(timing_path.read_text(encoding="utf-8"))
            wall_time = (
                f"Recorded training-loop wall time: "
                f"{saved_timing['e2e_wall_clock_seconds'] / 60:.1f} min."
            )
        elif "dataset_preparation_seconds" in meta and "started_at" in meta:
            elapsed = (
                datetime.now().astimezone() - datetime.fromisoformat(meta["started_at"])
            ).total_seconds()
            wall_time = (
                f"Elapsed since launcher start at report generation: {elapsed / 60:.1f} min; "
                f"dataset preparation: {meta['dataset_preparation_seconds'] / 60:.1f} min. "
                "No final timing record exists; this is not proof the process is still alive."
            )
        headers = "".join(f"<th>{html.escape(key)}</th>" for key in ("step",) + columns)
        table = []
        for row in rows:
            cells = []
            for key in ("step",) + columns:
                value = row.get(key)
                if value is None:
                    cells.append("<td>not recorded</td>")
                elif _finite_metric(value):
                    cells.append(f"<td>{value:.6g}</td>")
                else:
                    cells.append(f"<td>invalid measurement: {html.escape(str(value))}</td>")
            table.append("<tr>" + "".join(cells) + "</tr>")
        checkpoint_path = path / "checkpoints.jsonl"
        checkpoints = (
            [json.loads(line) for line in checkpoint_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if checkpoint_path.exists() else []
        )
        panels.append(
            f"<h2>{html.escape(name)}</h2><p>Completed updates: {completed}; "
            f"third-update milestone: {'reached' if completed >= 3 else 'not reached'}.</p>"
            f"<p>{html.escape(wall_time)}</p><p>{html.escape(timing)}</p>"
            f"<details><summary>Resolved configuration</summary><pre>{html.escape(json.dumps(config, indent=2))}</pre></details>"
            f"<details><summary>Recorded checkpoints</summary><pre>{html.escape(json.dumps(checkpoints, indent=2))}</pre></details>"
            f"<div class='table'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(table)}</tbody></table></div>"
        )
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>Loom RL numerics evidence</title>"
        "<style>body{font:15px system-ui;margin:32px;max-width:1200px}svg{width:100%;max-width:800px}"
        "svg text{font:12px system-ui}table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:6px}"
        ".table{overflow:auto}pre{white-space:pre-wrap}</style>"
        "<h1>Loom RL numerics evidence</h1>"
        "<p>Primary metric: sampler/trainer KL v1 (optim/kl_sample_train_v1), "
        "the mean of logp_sampler - logp_trainer over evaluated action tokens. "
        "This finite-sample estimator can be negative; it is not clamped or replaced by reward. "
        "KL v2, reward, accuracy and token depth are supporting diagnostics, not passing scores.</p>"
        "<p>Observed local logs only. Missing depth buckets are not zero. "
        "Invalid measurements remain in the step table and are excluded from numeric charts; "
        "they are not replaced by previous finite values. "
        "No historical buggy control is fabricated. A configured 256K limit does not prove actual 256K execution. "
        "max_context_tokens is a mean of episode peaks, not the deepest observed episode. "
        "kl_sample_train_max_action_position is the largest evaluated action-token position, not a configured limit. "
        "Compare resolved configurations before interpreting model differences.</p>"
        + chart("Primary: sampler/trainer KL v1 by optimizer-step index", step_series)
        + chart("Step-0 KL v1 by context token position", depth_series, depth=True)
        + "".join(panels) + "</html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_paths", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-recovery-attempts", action="store_true")
    args = parser.parse_args()
    paths = args.log_paths
    missing = []
    if args.include_recovery_attempts:
        from programmed_training.auto_resume import attempt_log_paths

        paths = list(dict.fromkeys(
            Path(path) for root in paths for path in attempt_log_paths(str(root))
        ))
        missing = [
            path for path in paths
            if not (path / "metrics.jsonl").is_file()
            or not (path / "metrics.jsonl").read_text(encoding="utf-8").strip()
        ]
        paths = [path for path in paths if path not in missing]
        if not paths:
            raise ValueError("No completed step metrics in recorded recovery attempts")
    report = build_report(paths)
    if missing:
        report = report.replace(
            "</html>",
            "<p>Recorded attempts without completed metrics (not reconstructed): "
            + html.escape(", ".join(str(path) for path in missing)) + "</p></html>",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
