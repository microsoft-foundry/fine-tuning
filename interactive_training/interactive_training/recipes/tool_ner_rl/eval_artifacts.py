from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from interactive_training.recipes.tool_ner_rl.ner_env import NerLoggingEnv
from interactive_training.recipes.tool_ner_rl.ner_task import GoldEntity, NerTask
from interactive_training.recipes.tool_ner_rl.private_files import private_text_writer
from interactive_training.rl.metric_util import ValidationRecord
from interactive_training.rl.types import Env, Trajectory

_IDENTIFIER_CUES = {
    "CREDITCARDNUMBER": ("credit card", "payment card", " card "),
    "DRIVERLICENSENUM": ("driver", "licence", "license"),
    "IDCARDNUM": ("national id", "id card", "id number", " id "),
    "PASSPORTNUM": ("passport",),
    "SOCIALNUM": ("social security", "social insurance"),
    "TAXNUM": ("tax",),
}


def _entity_dict(entity: GoldEntity, source: str) -> dict[str, Any]:
    return {
        "start": entity.start,
        "end": entity.end,
        "label": entity.label,
        "text": source[entity.start : entity.end],
    }


def _exact_counts(
    predictions: set[GoldEntity],
    gold: set[GoldEntity],
) -> tuple[int, int, int, float, float, float]:
    true_positives = len(predictions & gold)
    false_positives = len(predictions - gold)
    false_negatives = len(gold - predictions)
    precision = true_positives / len(predictions) if predictions else float(not gold)
    recall = true_positives / len(gold) if gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return true_positives, false_positives, false_negatives, precision, recall, f1


def _counts_to_metrics(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> dict[str, float | int]:
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 1.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 1.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _binary_pii_exact_span_metrics(
    predictions: set[GoldEntity],
    gold: set[GoldEntity],
) -> dict[str, float | int]:
    """Score exact entity boundaries while ignoring the 19-way label."""
    predicted_spans = {(entity.start, entity.end) for entity in predictions}
    gold_spans = {(entity.start, entity.end) for entity in gold}
    true_positives = len(predicted_spans & gold_spans)
    false_positives = len(predicted_spans - gold_spans)
    false_negatives = len(gold_spans - predicted_spans)
    exact_labeled_spans = {
        (entity.start, entity.end)
        for entity in predictions & gold
    }
    wrong_label_spans = (predicted_spans & gold_spans) - exact_labeled_spans
    return {
        **_counts_to_metrics(true_positives, false_positives, false_negatives),
        "wrong_label_on_exact_span": len(wrong_label_spans),
        "label_accuracy_on_exact_spans": (
            len(exact_labeled_spans) / true_positives if true_positives else 1.0
        ),
    }


def _binary_pii_metrics(
    predictions: set[GoldEntity],
    gold: set[GoldEntity],
    source: str,
) -> dict[str, float | int]:
    """Score label- and segmentation-agnostic PII character coverage.

    Whitespace is excluded so a predicted ``John Smith`` span receives full
    credit against adjacent gold ``John`` and ``Smith`` spans.
    """
    predicted_characters = {
        index
        for entity in predictions
        for index in range(entity.start, entity.end)
        if not source[index].isspace()
    }
    gold_characters = {
        index
        for entity in gold
        for index in range(entity.start, entity.end)
        if not source[index].isspace()
    }
    return _counts_to_metrics(
        len(predicted_characters & gold_characters),
        len(predicted_characters - gold_characters),
        len(gold_characters - predicted_characters),
    )


def _overlap_metrics(
    predictions: set[GoldEntity],
    gold: set[GoldEntity],
    *,
    require_same_label: bool,
) -> dict[str, float | int]:
    """Score one-to-one entity matches with any positive span overlap."""
    predicted_entities = sorted(predictions)
    gold_entities = sorted(gold)
    adjacency = [
        [
            gold_index
            for gold_index, gold_entity in enumerate(gold_entities)
            if (not require_same_label or prediction.label == gold_entity.label)
            and max(prediction.start, gold_entity.start)
            < min(prediction.end, gold_entity.end)
        ]
        for prediction in predicted_entities
    ]
    matched_prediction_by_gold: dict[int, int] = {}

    def match(prediction_index: int, visited_gold: set[int]) -> bool:
        for gold_index in adjacency[prediction_index]:
            if gold_index in visited_gold:
                continue
            visited_gold.add(gold_index)
            previous_prediction = matched_prediction_by_gold.get(gold_index)
            if previous_prediction is None or match(previous_prediction, visited_gold):
                matched_prediction_by_gold[gold_index] = prediction_index
                return True
        return False

    true_positives = sum(
        match(prediction_index, set())
        for prediction_index in range(len(predicted_entities))
    )
    return _counts_to_metrics(
        true_positives,
        len(predicted_entities) - true_positives,
        len(gold_entities) - true_positives,
    )


def _error_kind(
    entity: GoldEntity,
    others: set[GoldEntity],
) -> str:
    if any(
        other.start == entity.start and other.end == entity.end and other.label != entity.label
        for other in others
    ):
        return "label_mismatch"
    if any(
        other.label == entity.label
        and max(other.start, entity.start) < min(other.end, entity.end)
        for other in others
    ):
        return "boundary_mismatch"
    return "unmatched"


def _write_private_json(path: Path, value: Any) -> None:
    with private_text_writer(path) as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def _write_private_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with private_text_writer(path) as file:
        for value in values:
            file.write(json.dumps(value, ensure_ascii=False) + "\n")


def _bootstrap_micro_f1(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    samples: int = 2_000,
) -> dict[str, float | int]:
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [rows[rng.randrange(len(rows))] for _ in rows]
        tp = sum(row["metrics"]["true_positives"] for row in selected)
        fp = sum(row["metrics"]["false_positives"] for row in selected)
        fn = sum(row["metrics"]["false_negatives"] for row in selected)
        denominator = 2 * tp + fp + fn
        estimates.append(2 * tp / denominator if denominator else 0.0)
    estimates.sort()
    return {
        "samples": samples,
        "seed": seed,
        "p2_5": estimates[int(0.025 * samples)],
        "p50": estimates[int(0.5 * samples)],
        "p97_5": estimates[int(0.975 * samples)],
    }


def _tab_reference_rows(
    sampled_rows: list[dict[str, Any]], tasks: tuple[NerTask, ...]
) -> list[dict[str, Any]]:
    if not tasks:
        return sampled_rows
    sampled = {(row["document_uid"], row["source_start"]): row for row in sampled_rows}
    expected = {(task.document_uid, task.source_start) for task in tasks}
    if len(sampled) != len(sampled_rows) or set(sampled) != expected:
        raise ValueError("TAB evaluation requires exactly one prediction per source window")
    rows = []
    for task in tasks:
        row = sampled[(task.document_uid, task.source_start)]
        if row["source_text"] != task.source_text:
            raise ValueError("TAB reference source does not match sampled window")
        rows.append({
            **row,
            "uid": task.uid,
            "annotation_layer": task.annotation_layer,
            "gold": [_entity_dict(entity, task.source_text) for entity in sorted(task.gold_entities)],
            "gold_attributes": [asdict(attribute) for attribute in task.gold_attributes],
        })
    return rows


def _tab_document_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    tab_rows = [row for row in rows if row.get("dataset_name") == "tab"]
    if not tab_rows:
        return None

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in tab_rows:
        key = (row["document_uid"], row["annotation_layer"])
        group = groups.setdefault(
            key,
            {
                "predictions": set(),
                "gold": set(),
                "attributes": set(),
                "predicted_characters": set(),
                "gold_characters": set(),
            },
        )
        source_start = row["source_start"]
        source = row["source_text"]
        for item in row["predictions"]:
            entity = GoldEntity(
                source_start + item["start"],
                source_start + item["end"],
                item["label"],
            )
            group["predictions"].add(entity)
            group["predicted_characters"].update(
                source_start + index
                for index in range(item["start"], item["end"])
                if not source[index].isspace()
            )
        for item in row["gold"]:
            entity = GoldEntity(
                source_start + item["start"],
                source_start + item["end"],
                item["label"],
            )
            group["gold"].add(entity)
            group["gold_characters"].update(
                source_start + index
                for index in range(item["start"], item["end"])
                if not source[index].isspace()
            )
        group["attributes"].update(
            (
                source_start + item["start"],
                source_start + item["end"],
                item["label"],
                item["identifier_type"],
                item["entity_id"],
            )
            for item in row["gold_attributes"]
        )

    exact_totals: Counter[str] = Counter()
    character_totals: Counter[str] = Counter()
    span_totals = {name: Counter() for name in ("binary_exact", "labeled_overlap", "binary_overlap")}
    identifier_totals: dict[str, Counter[str]] = {
        "DIRECT": Counter(),
        "QUASI": Counter(),
    }
    complete_entities = total_entities = 0
    document_layer_f1: list[float] = []
    for group in groups.values():
        predictions = group["predictions"]
        gold = group["gold"]
        tp, fp, fn, _, _, f1 = _exact_counts(predictions, gold)
        exact_totals.update({"tp": tp, "fp": fp, "fn": fn})
        document_layer_f1.append(f1)
        for name, metrics in (
            ("binary_exact", _binary_pii_exact_span_metrics(predictions, gold)),
            ("labeled_overlap", _overlap_metrics(predictions, gold, require_same_label=True)),
            ("binary_overlap", _overlap_metrics(predictions, gold, require_same_label=False)),
        ):
            span_totals[name].update({
                key: metrics[key] for key in ("true_positives", "false_positives", "false_negatives")
            })
        character_totals.update(
            {
                "tp": len(group["predicted_characters"] & group["gold_characters"]),
                "fp": len(group["predicted_characters"] - group["gold_characters"]),
                "fn": len(group["gold_characters"] - group["predicted_characters"]),
            }
        )
        predicted_spans = {
            (entity.start, entity.end) for entity in predictions
        }
        mentions_by_entity: dict[str, set[tuple[int, int]]] = {}
        for start, end, label, identifier_type, entity_id in group["attributes"]:
            matched = GoldEntity(start, end, label) in predictions
            identifier_totals[identifier_type]["gold"] += 1
            identifier_totals[identifier_type]["tp"] += int(matched)
            mentions_by_entity.setdefault(entity_id, set()).add((start, end))
        total_entities += len(mentions_by_entity)
        complete_entities += sum(
            mentions <= predicted_spans for mentions in mentions_by_entity.values()
        )

    exact = _counts_to_metrics(
        exact_totals["tp"], exact_totals["fp"], exact_totals["fn"]
    )
    characters = _counts_to_metrics(
        character_totals["tp"],
        character_totals["fp"],
        character_totals["fn"],
    )
    return {
        "documents": len({row["document_uid"] for row in tab_rows}),
        "document_annotation_layers": len(groups),
        "exact": exact,
        **{
            name: _counts_to_metrics(counts["true_positives"], counts["false_positives"], counts["false_negatives"])
            for name, counts in span_totals.items()
        },
        "mean_document_layer_f1": sum(document_layer_f1) / len(document_layer_f1),
        "character_coverage": characters,
        "identifier_recall": {
            identifier_type.lower(): {
                "gold": counts["gold"],
                "tp": counts["tp"],
                "recall": counts["tp"] / counts["gold"] if counts["gold"] else 1.0,
            }
            for identifier_type, counts in identifier_totals.items()
        },
        "coreference_complete_masking": {
            "entities": total_entities,
            "complete_entities": complete_entities,
            "recall": complete_entities / total_entities if total_entities else 1.0,
        },
    }


@dataclass
class NerEvalArtifactObserver:
    log_path: Path
    seed: int
    error_sample_size: int = 12
    turn_token_limits: tuple[int, ...] = (768, 640)
    name: str = "base"
    max_tokens: int | None = None
    tab_reference_tasks: tuple[NerTask, ...] = ()
    summary_metrics: dict[str, float] = field(default_factory=dict, init=False)

    def create_record(
        self,
        step: int,
        trajectory: Trajectory,
        env: Env,
        tags: tuple[str, ...],
    ) -> ValidationRecord:
        if not isinstance(env, NerLoggingEnv):
            raise TypeError(f"Expected NerLoggingEnv, got {type(env).__name__}")
        task = env.episode.task
        predictions = sorted(env.episode.predictions or frozenset())
        gold = sorted(task.gold_entities)
        parse_error = any(transition.metrics.get("parse_error") for transition in trajectory.transitions)
        hit_token_cap = any(
            len(transition.ac.tokens)
            >= self.turn_token_limits[min(index, len(self.turn_token_limits) - 1)] - 4
            for index, transition in enumerate(trajectory.transitions)
        )
        tool_errors = []
        for message in env.message_env.history:
            if message.get("role") != "tool" or not isinstance(message.get("content"), str):
                continue
            try:
                tool_payload = json.loads(message["content"])
            except json.JSONDecodeError:
                continue
            if isinstance(tool_payload, dict) and "error" in tool_payload:
                tool_errors.append(
                    {"tool": message.get("name", "unknown"), "error": tool_payload["error"]}
                )
        payload = {
            "uid": task.uid,
            "dataset_name": task.dataset_name,
            "document_uid": task.document_uid,
            "annotation_layer": task.annotation_layer,
            "source_start": task.source_start,
            "source_text": task.source_text,
            "predictions": [_entity_dict(entity, task.source_text) for entity in predictions],
            "gold": [_entity_dict(entity, task.source_text) for entity in gold],
            "gold_attributes": [
                {
                    "start": attribute.start,
                    "end": attribute.end,
                    "label": attribute.label,
                    "identifier_type": attribute.identifier_type,
                    "entity_id": attribute.entity_id,
                }
                for attribute in task.gold_attributes
            ],
            "turns": len(trajectory.transitions),
            "generated_tokens": sum(len(transition.ac.tokens) for transition in trajectory.transitions),
            "finalized": env.episode.predictions is not None,
            "parse_error": parse_error,
            "hit_token_cap": hit_token_cap,
            "tool_errors": tool_errors,
            "tool_counts": {
                "ground_entities": getattr(env.episode, "ground_entities_calls", 0),
                "find_text": getattr(env.episode, "find_text_calls", 0),
                "grep": getattr(env.episode, "grep_calls", 0),
                "regex_grep": getattr(env.episode, "regex_grep_calls", 0),
                "inspect": getattr(env.episode, "inspect_calls", 0),
                "select_occurrence": getattr(env.episode, "select_occurrence_calls", 0),
                "submit": getattr(env.episode, "submit_calls", 0),
                "finalize": getattr(env.episode, "finalize_calls", 0),
            },
        }
        return ValidationRecord(
            step=step,
            input=task.uid,
            response=json.dumps(payload, ensure_ascii=False),
            ground_truth=None,
            tags=tags,
        )

    def on_validation_complete(self, records: list[ValidationRecord]) -> None:
        sampled_rows = [json.loads(record.response) for record in records]
        reference_rows = _tab_reference_rows(sampled_rows, self.tab_reference_tasks)
        rows: list[dict[str, Any]] = []
        per_label: dict[str, Counter[str]] = {}
        error_counts: Counter[str] = Counter()
        density_slices: dict[str, Counter[str]] = {}
        identifier_cue_slices: dict[str, Counter[str]] = {}
        total_tp = total_fp = total_fn = 0
        binary_totals: Counter[str] = Counter()
        binary_exact_span_totals: Counter[str] = Counter()
        labeled_overlap_totals: Counter[str] = Counter()
        binary_overlap_totals: Counter[str] = Counter()

        for row in reference_rows:
            predictions = {
                GoldEntity(item["start"], item["end"], item["label"])
                for item in row["predictions"]
            }
            gold = {
                GoldEntity(item["start"], item["end"], item["label"])
                for item in row["gold"]
            }
            tp, fp, fn, precision, recall, f1 = _exact_counts(predictions, gold)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            row["metrics"] = {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            row["binary_pii"] = _binary_pii_metrics(
                predictions,
                gold,
                row["source_text"],
            )
            binary_totals.update(
                {
                    key: int(row["binary_pii"][key])
                    for key in (
                        "true_positives",
                        "false_positives",
                        "false_negatives",
                    )
                }
            )
            row["binary_pii_exact_span"] = _binary_pii_exact_span_metrics(
                predictions,
                gold,
            )
            binary_exact_span_totals.update(
                {
                    key: int(row["binary_pii_exact_span"][key])
                    for key in (
                        "true_positives",
                        "false_positives",
                        "false_negatives",
                        "wrong_label_on_exact_span",
                    )
                }
            )
            row["labeled_overlap"] = _overlap_metrics(
                predictions,
                gold,
                require_same_label=True,
            )
            labeled_overlap_totals.update(
                {
                    key: int(row["labeled_overlap"][key])
                    for key in (
                        "true_positives",
                        "false_positives",
                        "false_negatives",
                    )
                }
            )
            row["binary_overlap"] = _overlap_metrics(
                predictions,
                gold,
                require_same_label=False,
            )
            binary_overlap_totals.update(
                {
                    key: int(row["binary_overlap"][key])
                    for key in (
                        "true_positives",
                        "false_positives",
                        "false_negatives",
                    )
                }
            )
            row["errors"] = {
                "false_positives": [
                    {
                        **_entity_dict(entity, row["source_text"]),
                        "kind": _error_kind(entity, gold),
                    }
                    for entity in sorted(predictions - gold)
                ],
                "false_negatives": [
                    {
                        **_entity_dict(entity, row["source_text"]),
                        "kind": _error_kind(entity, predictions),
                    }
                    for entity in sorted(gold - predictions)
                ],
            }
            for items in row["errors"].values():
                error_counts.update(item["kind"] for item in items)

            labels = {entity.label for entity in predictions | gold}
            for label in labels:
                counts = per_label.setdefault(label, Counter())
                predicted_label = {entity for entity in predictions if entity.label == label}
                gold_label = {entity for entity in gold if entity.label == label}
                counts["tp"] += len(predicted_label & gold_label)
                counts["fp"] += len(predicted_label - gold_label)
                counts["fn"] += len(gold_label - predicted_label)

            density = "1-4" if len(gold) <= 4 else "5-8" if len(gold) <= 8 else "9+"
            density_counts = density_slices.setdefault(density, Counter())
            density_counts.update({"documents": 1, "tp": tp, "fp": fp, "fn": fn})

            source_lower = row["source_text"].lower()
            for entity in gold:
                cues = _IDENTIFIER_CUES.get(entity.label)
                if cues is None:
                    continue
                context = source_lower[
                    max(0, entity.start - 55) : min(len(source_lower), entity.end + 55)
                ]
                cue_bucket = "cued" if any(cue in context for cue in cues) else "uncued"
                cue_counts = identifier_cue_slices.setdefault(
                    f"{entity.label}:{cue_bucket}", Counter()
                )
                cue_counts["gold"] += 1
                cue_counts["tp"] += int(entity in predictions)
            rows.append(row)

        label_summary: dict[str, dict[str, float | int]] = {}
        label_f1_values: list[float] = []
        for label, counts in sorted(per_label.items()):
            precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
            recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            label_f1_values.append(f1)
            label_summary[label] = {
                "tp": counts["tp"],
                "fp": counts["fp"],
                "fn": counts["fn"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 1.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if micro_precision + micro_recall
            else 0.0
        )
        binary_summary = {
            "label": "PII",
            **_counts_to_metrics(
                binary_totals["true_positives"],
                binary_totals["false_positives"],
                binary_totals["false_negatives"],
            ),
            "unit": "non_whitespace_source_character",
        }
        exact_span_metrics = _counts_to_metrics(
            binary_exact_span_totals["true_positives"],
            binary_exact_span_totals["false_positives"],
            binary_exact_span_totals["false_negatives"],
        )
        exact_span_count = binary_exact_span_totals["true_positives"]
        binary_exact_span_summary = {
            "label": "PII",
            "tp": exact_span_metrics["true_positives"],
            "fp": exact_span_metrics["false_positives"],
            "fn": exact_span_metrics["false_negatives"],
            "precision": exact_span_metrics["precision"],
            "recall": exact_span_metrics["recall"],
            "f1": exact_span_metrics["f1"],
            "unit": "entity_span",
            "wrong_label_on_exact_span": binary_exact_span_totals[
                "wrong_label_on_exact_span"
            ],
            "label_accuracy_on_exact_spans": (
                (
                    exact_span_count
                    - binary_exact_span_totals["wrong_label_on_exact_span"]
                )
                / exact_span_count
                if exact_span_count
                else 1.0
            ),
        }
        labeled_overlap_summary = {
            **_counts_to_metrics(
                labeled_overlap_totals["true_positives"],
                labeled_overlap_totals["false_positives"],
                labeled_overlap_totals["false_negatives"],
            ),
            "unit": "entity",
            "match": "positive_span_overlap_and_same_label",
            "assignment": "maximum_cardinality_one_to_one",
        }
        binary_overlap_summary = {
            "label": "PII",
            **_counts_to_metrics(
                binary_overlap_totals["true_positives"],
                binary_overlap_totals["false_positives"],
                binary_overlap_totals["false_negatives"],
            ),
            "unit": "entity",
            "match": "positive_span_overlap",
            "assignment": "maximum_cardinality_one_to_one",
        }

        density_summary = {}
        for name, counts in sorted(density_slices.items()):
            precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
            recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 1.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            density_summary[name] = {
                "documents": counts["documents"],
                "tp": counts["tp"],
                "fp": counts["fp"],
                "fn": counts["fn"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        identifier_summary = {
            name: {
                "gold": counts["gold"],
                "tp": counts["tp"],
                "recall": counts["tp"] / counts["gold"],
            }
            for name, counts in sorted(identifier_cue_slices.items())
        }
        tab_summary = _tab_document_summary(rows)
        summary = {
            "documents": len(rows),
            "totals": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
            "micro": {
                "precision": micro_precision,
                "recall": micro_recall,
                "f1": micro_f1,
            },
            "micro_f1_document_bootstrap_95ci": _bootstrap_micro_f1(
                rows,
                seed=self.seed,
            ),
            "macro_label_f1": sum(label_f1_values) / len(label_f1_values) if label_f1_values else 0.0,
            "binary_pii": binary_summary,
            "binary_pii_exact_span": binary_exact_span_summary,
            "labeled_overlap": labeled_overlap_summary,
            "binary_overlap": binary_overlap_summary,
            "per_label": label_summary,
            "error_kinds": dict(sorted(error_counts.items())),
            "density_slices": density_summary,
            "identifier_cue_slices": identifier_summary,
            "protocol": {
                "finalized_documents": sum(bool(row["finalized"]) for row in rows),
                "parse_error_documents": sum(bool(row["parse_error"]) for row in rows),
                "token_cap_documents": sum(bool(row["hit_token_cap"]) for row in rows),
                "mean_turns": sum(row["turns"] for row in rows) / len(rows),
                "mean_generated_tokens": sum(row["generated_tokens"] for row in rows)
                / len(rows),
                "max_generated_tokens": max(row["generated_tokens"] for row in rows),
            },
        }
        if tab_summary is not None:
            summary["tab_document_aggregation"] = tab_summary
            summary["scoring_unit"] = "window_annotation_layer; use tab_document_aggregation for document metrics"
            summary["protocol"] = {
                "sampled_windows": len(sampled_rows),
                "finalized_documents": sum(bool(row["finalized"]) for row in sampled_rows),
                "parse_error_documents": sum(bool(row["parse_error"]) for row in sampled_rows),
                "token_cap_documents": sum(bool(row["hit_token_cap"]) for row in sampled_rows),
                "mean_turns": sum(row["turns"] for row in sampled_rows) / len(sampled_rows),
                "mean_generated_tokens": sum(row["generated_tokens"] for row in sampled_rows) / len(sampled_rows),
                "max_generated_tokens": max(row["generated_tokens"] for row in sampled_rows),
            }
            summary.pop("micro_f1_document_bootstrap_95ci")

        erroneous = [
            row
            for row in rows
            if row["metrics"]["false_positives"]
            or row["metrics"]["false_negatives"]
        ]
        semantic_errors = [row for row in erroneous if row["finalized"]]
        protocol_errors = [row for row in rows if not row["finalized"]]
        sampled_errors = sorted(
            semantic_errors,
            key=lambda row: (
                row["metrics"]["f1"],
                hashlib.sha256(f"{self.seed}:{row['uid']}".encode()).digest(),
            ),
        )[: self.error_sample_size]

        _write_private_jsonl(self.log_path / "per_document.jsonl", rows)
        _write_private_json(self.log_path / "error_summary.json", summary)
        _write_private_json(self.log_path / "error_review_sample.json", sampled_errors)
        _write_private_json(self.log_path / "protocol_errors.json", protocol_errors)
        self.summary_metrics = {
            "base/eval/micro_precision": micro_precision,
            "base/eval/micro_recall": micro_recall,
            "base/eval/micro_f1": micro_f1,
            "base/eval/macro_label_f1": summary["macro_label_f1"],
            "base/eval/binary_pii_precision": binary_summary["precision"],
            "base/eval/binary_pii_recall": binary_summary["recall"],
            "base/eval/binary_pii_f1": binary_summary["f1"],
            "base/eval/binary_pii_exact_span_f1": binary_exact_span_summary["f1"],
            "base/eval/labeled_overlap_f1": labeled_overlap_summary["f1"],
            "base/eval/binary_overlap_f1": binary_overlap_summary["f1"],
            "base/eval/label_accuracy_on_exact_spans": binary_exact_span_summary[
                "label_accuracy_on_exact_spans"
            ],
            "base/eval/error_documents": float(len(erroneous)),
            "base/eval/protocol_error_documents": float(len(protocol_errors)),
            "base/eval/finalized_rate": summary["protocol"]["finalized_documents"]
            / len(rows),
            "base/eval/parse_error_rate": summary["protocol"]["parse_error_documents"]
            / len(rows),
            "base/eval/token_cap_rate": summary["protocol"]["token_cap_documents"]
            / len(rows),
        }
        if tab_summary is not None:
            self.summary_metrics.update(
                {
                    "base/eval/finalized_rate": sum(row["finalized"] for row in sampled_rows) / len(sampled_rows),
                    "base/eval/parse_error_rate": sum(row["parse_error"] for row in sampled_rows) / len(sampled_rows),
                    "base/eval/token_cap_rate": sum(row["hit_token_cap"] for row in sampled_rows) / len(sampled_rows),
                    "base/eval/protocol_error_documents": float(sum(not row["finalized"] for row in sampled_rows)),
                    "base/eval/tab_document_binary_exact_f1": tab_summary["binary_exact"]["f1"],
                    "base/eval/tab_document_labeled_overlap_f1": tab_summary["labeled_overlap"]["f1"],
                    "base/eval/tab_document_binary_overlap_f1": tab_summary["binary_overlap"]["f1"],
                    "base/eval/tab_document_character_f1": tab_summary["character_coverage"]["f1"],
                    "base/eval/tab_document_micro_precision": tab_summary["exact"][
                        "precision"
                    ],
                    "base/eval/tab_document_micro_recall": tab_summary["exact"]["recall"],
                    "base/eval/tab_document_micro_f1": tab_summary["exact"]["f1"],
                    "base/eval/tab_mean_document_layer_f1": tab_summary[
                        "mean_document_layer_f1"
                    ],
                    "base/eval/tab_direct_recall": tab_summary["identifier_recall"][
                        "direct"
                    ]["recall"],
                    "base/eval/tab_quasi_recall": tab_summary["identifier_recall"][
                        "quasi"
                    ]["recall"],
                    "base/eval/tab_coreference_complete_recall": tab_summary[
                        "coreference_complete_masking"
                    ]["recall"],
                }
            )