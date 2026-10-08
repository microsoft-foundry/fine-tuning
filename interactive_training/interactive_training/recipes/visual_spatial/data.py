"""Shared contract for the Do-You-See-Me visual-spatial infrastructure task."""

import io
from typing import Any, cast

import datasets

from interactive_training.renderers import Message

DATASET_NAME = "microsoft/Do-You-See-Me"
DATASET_ID = "visual_spatial"


def is_visual_spatial(row: dict[str, Any]) -> bool:
    return row["dataset_id"] == DATASET_ID


def normalize_answer(answer: Any) -> str:
    return " ".join(str(answer).split()).casefold()


def answers_match(candidate: Any, expected: Any) -> bool:
    return normalize_answer(candidate) == normalize_answer(expected)


def normalize_image(image: Any) -> Any:
    if not isinstance(image, dict):
        return image

    image_bytes = image.get("bytes")
    image_path = image.get("path")
    if image_bytes is None and image_path is None:
        return image

    from PIL import Image

    source = io.BytesIO(image_bytes) if image_bytes is not None else image_path
    decoded = Image.open(source)
    decoded.load()
    return decoded


def user_message(row: dict[str, Any]) -> Message:
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": normalize_image(row["image"])},
            {
                "type": "text",
                "text": (
                    "Reply with only the exact short answer to this question: "
                    f"{str(row['question']).strip()}"
                ),
            },
        ],
    }


def grouped_user_message(rows: list[dict[str, Any]]) -> Message:
    if not rows:
        raise ValueError("grouped visual-spatial examples require at least one row")
    content = [
        {"type": "image", "image": normalize_image(row["image"])}
        for row in rows
    ]
    content.append(
        {
            "type": "text",
            "text": (
                f"Inspect all {len(rows)} images and reply with only the exact "
                f"short answer to this question about the first image: "
                f"{str(rows[0]['question']).strip()}"
            ),
        }
    )
    return {"role": "user", "content": content}


def absolute_difference_answer(rows: list[dict[str, Any]]) -> str:
    if len(rows) != 2:
        raise ValueError("absolute-difference examples require exactly two rows")
    try:
        counts = [int(str(row["answer"]).strip()) for row in rows]
    except (KeyError, ValueError) as exc:
        raise ValueError("absolute-difference answers must be integers") from exc
    return str(abs(counts[0] - counts[1]))


def absolute_difference_user_message(rows: list[dict[str, Any]]) -> Message:
    if len(rows) != 2:
        raise ValueError("absolute-difference examples require exactly two rows")
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": normalize_image(row["image"])}
            for row in rows
        ]
        + [
            {
                "type": "text",
                "text": (
                    "Answer each visual counting question, then reply with only "
                    "the absolute difference between the two counts as one integer.\n"
                    f"First image: {str(rows[0]['question']).strip()}\n"
                    f"Second image: {str(rows[1]['question']).strip()}"
                ),
            }
        ],
    }


def sft_messages(row: dict[str, Any]) -> list[Message]:
    return [
        user_message(row),
        {"role": "assistant", "content": str(row["answer"]).strip()},
    ]


def grouped_sft_messages(rows: list[dict[str, Any]]) -> list[Message]:
    if not rows:
        raise ValueError("grouped visual-spatial examples require at least one row")
    return [
        grouped_user_message(rows),
        {"role": "assistant", "content": str(rows[0]["answer"]).strip()},
    ]


def absolute_difference_sft_messages(rows: list[dict[str, Any]]) -> list[Message]:
    return [
        absolute_difference_user_message(rows),
        {"role": "assistant", "content": absolute_difference_answer(rows)},
    ]


COUNT_CATEGORY_CODES = {
    "0": "3",
    "1": "5",
    "2": "0",
    "3": "6",
    "4": "1",
    "5": "2",
    "6": "4",
}


def count_category_answer(row: dict[str, Any]) -> str:
    count = str(row["answer"]).strip()
    try:
        return COUNT_CATEGORY_CODES[count]
    except KeyError as exc:
        raise ValueError(f"unsupported count category: {count!r}") from exc


def count_category_user_message(row: dict[str, Any]) -> Message:
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": normalize_image(row["image"])},
            {
                "type": "text",
                "text": (
                    f"{str(row['question']).strip()}\n"
                    "Reply with only the learned category code as one integer."
                ),
            },
        ],
    }


def count_category_sft_messages(row: dict[str, Any]) -> list[Message]:
    return [
        count_category_user_message(row),
        {"role": "assistant", "content": count_category_answer(row)},
    ]


def split_dataset(
    dataset: datasets.Dataset,
    *,
    seed: int = 0,
    test_size: int | float = 0.1,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    visual = dataset.filter(
        lambda dataset_id: dataset_id == DATASET_ID,
        input_columns=["dataset_id"],
    )
    split = visual.train_test_split(test_size=test_size, seed=seed, shuffle=True)
    return split["train"], split["test"]


def load_dataset(
    *,
    seed: int = 0,
    test_size: int | float = 0.1,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    dataset = datasets.load_dataset(DATASET_NAME, split="train")
    return split_dataset(cast(datasets.Dataset, dataset), seed=seed, test_size=test_size)