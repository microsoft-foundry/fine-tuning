"""BillSum loading, deterministic splitting, and token-budget filtering."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, cast

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

BILLSUM_DATASET_NAME = "FiscalNote/billsum"
BILLSUM_REVISION = "3d8510441c06a3d9dfb32eb0d7f80151730bcc4f"


@dataclass(frozen=True)
class BillSumExample:
    """One immutable summarization example."""

    example_id: str
    source_split: str
    source_index: int
    title: str
    text: str
    reference_summary: str


@dataclass(frozen=True)
class DatasetFilterStats:
    rows_seen: int
    rows_invalid: int
    rows_policy_overlength: int
    rows_judge_overlength: int
    train_retained: int
    validation_retained: int

    def as_metrics(self) -> dict[str, float]:
        return {
            "data/rows_seen": float(self.rows_seen),
            "data/rows_invalid": float(self.rows_invalid),
            "data/rows_policy_overlength": float(self.rows_policy_overlength),
            "data/rows_judge_overlength": float(self.rows_judge_overlength),
            "data/train_retained": float(self.train_retained),
            "data/validation_retained": float(self.validation_retained),
        }


@dataclass(frozen=True)
class PreparedBillSumData:
    train: tuple[BillSumExample, ...]
    validation: tuple[BillSumExample, ...]
    stats: DatasetFilterStats


def stable_example_id(
    *, source_split: str, source_index: int, title: str, text: str
) -> str:
    """Build an ID that remains stable for a pinned dataset revision."""

    digest = hashlib.sha256(f"{title}\0{text}".encode("utf-8")).hexdigest()[:16]
    return f"billsum:{source_split}:{source_index}:{digest}"


def _example_from_row(
    row: dict[str, Any], *, default_split: str, default_index: int
) -> BillSumExample | None:
    title = row.get("title")
    text = row.get("text")
    summary = row.get("summary")
    source_split = row.get("source_split", default_split)
    source_index = row.get("source_index", default_index)
    if (
        not isinstance(title, str)
        or not title.strip()
        or not isinstance(text, str)
        or not text.strip()
        or not isinstance(summary, str)
        or not summary.strip()
        or not isinstance(source_split, str)
        or type(source_index) is not int
    ):
        return None
    return BillSumExample(
        example_id=stable_example_id(
            source_split=source_split,
            source_index=source_index,
            title=title,
            text=text,
        ),
        source_split=source_split,
        source_index=source_index,
        title=title,
        text=text,
        reference_summary=summary,
    )


def load_synthetic_jsonl(path: str | Path) -> list[BillSumExample]:
    """Load the repository-owned synthetic smoke fixture."""

    examples: list[BillSumExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Synthetic row {index} is not a JSON object")
            if "source_split" not in row:
                raise ValueError(
                    f"Synthetic row {index} must declare source_split='train'"
                )
            if row["source_split"] != "train":
                raise ValueError(
                    f"Synthetic row {index} has non-train source_split: "
                    f"{row['source_split']!r}"
                )
            example = _example_from_row(
                row, default_split="train", default_index=index
            )
            if example is None:
                raise ValueError(f"Synthetic row {index} has an invalid schema")
            examples.append(example)
    if not examples:
        raise ValueError(f"Synthetic fixture is empty: {path}")
    return examples


def load_billsum_train(
    *,
    dataset_name: str = BILLSUM_DATASET_NAME,
    revision: str = BILLSUM_REVISION,
    dataset_path: str | None = None,
) -> list[BillSumExample]:
    """Load only BillSum's official train split or a local synthetic JSONL."""

    if dataset_path is not None:
        examples = load_synthetic_jsonl(dataset_path)
    else:
        dataset = cast(
            Dataset,
            load_dataset(
                dataset_name,
                name="default",
                split="train",
                revision=revision,
            ),
        )
        examples = []
        for index, item in enumerate(dataset):
            row = cast(dict[str, Any], item)
            example = _example_from_row(
                row, default_split="train", default_index=index
            )
            if example is not None:
                examples.append(example)

    leaked = sorted({example.source_split for example in examples} - {"train"})
    if leaked:
        raise ValueError(
            "Training input contains non-train BillSum splits: "
            + ", ".join(leaked)
        )
    return examples


def _split_rank(example: BillSumExample, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{example.example_id}".encode("utf-8")).hexdigest()


def deterministic_train_validation_split(
    examples: Iterable[BillSumExample],
    *,
    validation_size: int,
    seed: int,
) -> tuple[list[BillSumExample], list[BillSumExample]]:
    """Carve validation deterministically from official train without leakage."""

    ordered = sorted(examples, key=lambda example: _split_rank(example, seed))
    if validation_size <= 0:
        raise ValueError("validation_size must be positive")
    if validation_size >= len(ordered):
        raise ValueError(
            f"validation_size={validation_size} must be smaller than "
            f"the {len(ordered)} available train examples"
        )
    validation = ordered[:validation_size]
    train = ordered[validation_size:]
    return train, validation


def prepare_billsum_data(
    *,
    examples: Iterable[BillSumExample],
    validation_size: int,
    seed: int,
    policy_token_count: Callable[[BillSumExample], int],
    judge_base_token_count: Callable[[BillSumExample], int],
    policy_prompt_token_budget: int,
    judge_prompt_token_budget: int,
    judge_candidate_token_reserve: int,
    max_train_examples: int | None = None,
    max_validation_examples: int | None = None,
) -> PreparedBillSumData:
    """Split and filter rows; source text is never truncated."""

    source_examples = list(examples)
    train_candidates, validation_candidates = deterministic_train_validation_split(
        source_examples,
        validation_size=validation_size,
        seed=seed,
    )
    invalid = 0
    policy_overlength = 0
    judge_overlength = 0

    def keep(example: BillSumExample) -> bool:
        nonlocal invalid, policy_overlength, judge_overlength
        if (
            not example.title.strip()
            or not example.text.strip()
            or not example.reference_summary.strip()
        ):
            invalid += 1
            return False
        if policy_token_count(example) > policy_prompt_token_budget:
            policy_overlength += 1
            return False
        judge_tokens = judge_base_token_count(example) + judge_candidate_token_reserve
        if judge_tokens > judge_prompt_token_budget:
            judge_overlength += 1
            return False
        return True

    train = [example for example in train_candidates if keep(example)]
    validation = [example for example in validation_candidates if keep(example)]
    if max_train_examples is not None:
        train = train[: max(0, max_train_examples)]
    if max_validation_examples is not None:
        validation = validation[: max(0, max_validation_examples)]
    if not train:
        raise ValueError("No BillSum training examples remain after filtering")
    if not validation:
        raise ValueError("No BillSum validation examples remain after filtering")

    stats = DatasetFilterStats(
        rows_seen=len(source_examples),
        rows_invalid=invalid,
        rows_policy_overlength=policy_overlength,
        rows_judge_overlength=judge_overlength,
        train_retained=len(train),
        validation_retained=len(validation),
    )
    logger.info(
        "BillSum preprocessing: seen=%d invalid=%d policy_overlength=%d "
        "judge_overlength=%d train=%d validation=%d",
        stats.rows_seen,
        stats.rows_invalid,
        stats.rows_policy_overlength,
        stats.rows_judge_overlength,
        stats.train_retained,
        stats.validation_retained,
    )
    return PreparedBillSumData(tuple(train), tuple(validation), stats)
