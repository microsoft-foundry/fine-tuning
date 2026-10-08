"""Adapter for Azure OpenAI conversations JSONL vision datasets."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, cast

import datasets

from interactive_training.renderers import Message
from interactive_training.renderers import base as renderer_base
from interactive_training.recipes.visual_spatial.data import normalize_answer

logger = logging.getLogger(__name__)

MAX_IMAGES_PER_EXAMPLE = 64
MAX_TOTAL_IMAGE_BYTES_PER_EXAMPLE = 32_000_000
MAX_TOTAL_IMAGE_PIXELS_PER_EXAMPLE = 100_000_000
MAX_TEXT_CHARACTERS_PER_EXAMPLE = 1_000_000
MAX_JSONL_ROW_BYTES = 64_000_000
MAX_JSONL_FILE_BYTES = 512_000_000
MAX_DATASET_EXAMPLES = 50_000
MAX_MESSAGES_PER_EXAMPLE = 128


@dataclass(frozen=True)
class ImageDatasetSummary:
    examples: int
    images: int
    encoded_bytes: int
    decoded_pixels: int


def _image_references(messages: list[dict[str, Any]]) -> list[str]:
    references = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if not isinstance(image_url, dict) or not isinstance(
                image_url.get("url"), str
            ):
                raise ValueError("image_url parts must contain a string image_url.url")
            references.append(image_url["url"])
    return references


def _validate_reference_syntax(reference: str) -> None:
    if reference.startswith("data:"):
        renderer_base._load_image_reference(reference)
    else:
        renderer_base._validate_image_url_syntax(reference)


def _validate_jsonl_file(data_path: str) -> None:
    if not os.path.isfile(data_path):
        raise ValueError(f"data_path must be a local JSONL file: {data_path}")
    file_bytes = os.path.getsize(data_path)
    if file_bytes > MAX_JSONL_FILE_BYTES:
        raise ValueError(
            f"JSONL file exceeds the {MAX_JSONL_FILE_BYTES}-byte limit"
        )

    examples = 0
    with open(data_path, "rb") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            if len(line) > MAX_JSONL_ROW_BYTES:
                raise ValueError(
                    f"JSONL row {line_number} exceeds the "
                    f"{MAX_JSONL_ROW_BYTES}-byte limit"
                )
            if line.strip():
                examples += 1
                if examples > MAX_DATASET_EXAMPLES:
                    raise ValueError(
                        f"JSONL file exceeds the {MAX_DATASET_EXAMPLES}-example limit"
                    )
    if examples < 2:
        raise ValueError("JSONL data must contain at least two non-empty examples")


def _text_content(content: Any, *, field_name: str) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text = " ".join(
            str(part["text"]).strip()
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "text"
            and part.get("text") is not None
        ).strip()
        if text:
            return text
    raise ValueError(f"{field_name} must contain text")


def adapt_conversation(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    if len(messages) > MAX_MESSAGES_PER_EXAMPLE:
        raise ValueError(
            f"an example may contain at most {MAX_MESSAGES_PER_EXAMPLE} messages"
        )
    if not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant":
        raise ValueError("the final message must contain the assistant reference answer")

    prompt_messages = messages[:-1]
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("every message must be an object")
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("message roles must be system, user, assistant, or tool")
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict) or part.get("type") not in {
                    "text",
                    "image_url",
                }:
                    raise ValueError(
                        "message content parts must be text or image_url objects"
                    )
                if part["type"] == "text" and not isinstance(part.get("text"), str):
                    raise ValueError("text parts must contain string text")
        elif not isinstance(content, str):
            raise ValueError("message content must be a string or a list of content parts")

    user_messages = [
        message
        for message in prompt_messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if not user_messages:
        raise ValueError("messages must contain a user prompt")

    if not _image_references(user_messages):
        raise ValueError("the user prompt must contain an image_url content part")
    image_references = _image_references(prompt_messages)
    if len(image_references) > MAX_IMAGES_PER_EXAMPLE:
        raise ValueError(
            f"an example may contain at most {MAX_IMAGES_PER_EXAMPLE} images"
        )
    for reference in image_references:
        _validate_reference_syntax(reference)

    text_characters = sum(
        len(str(part.get("text", "")))
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "text"
    ) + sum(
        len(message["content"])
        for message in messages
        if isinstance(message.get("content"), str)
    )
    if text_characters > MAX_TEXT_CHARACTERS_PER_EXAMPLE:
        raise ValueError(
            f"example text exceeds the {MAX_TEXT_CHARACTERS_PER_EXAMPLE}-character limit"
        )

    return {
        "prompt_messages": prompt_messages,
        "question": _text_content(
            user_messages[-1].get("content"), field_name="the final user message"
        ),
        "answer": _text_content(
            messages[-1].get("content"), field_name="the final assistant message"
        ),
    }


def load_dataset(
    data_path: str,
    *,
    seed: int = 0,
    test_size: int | float = 0.1,
    image_cache_dir: str | None = None,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    _validate_jsonl_file(data_path)
    dataset = cast(
        datasets.Dataset,
        datasets.load_dataset("json", data_files=data_path, split="train"),
    )
    adapted = dataset.map(
        adapt_conversation,
        remove_columns=dataset.column_names,
        desc="Adapting Azure OpenAI conversations",
    )
    if image_cache_dir is not None:
        prepare_image_cache(adapted, image_cache_dir)
    split = adapted.train_test_split(
        test_size=test_size,
        seed=seed,
        shuffle=True,
    )
    return split["train"], split["test"]


def prepare_image_cache(
    dataset: datasets.Dataset,
    image_cache_dir: str,
) -> ImageDatasetSummary:
    total_images = 0
    total_bytes = 0
    total_pixels = 0
    for example_index, row in enumerate(dataset):
        references = _image_references(row["prompt_messages"])
        example_bytes = 0
        example_pixels = 0
        for reference in references:
            try:
                image_data = renderer_base._load_image_reference(
                    reference,
                    cache_dir=image_cache_dir,
                )
                example_bytes += len(image_data)
                if example_bytes > MAX_TOTAL_IMAGE_BYTES_PER_EXAMPLE:
                    raise ValueError(
                        "aggregate encoded image data exceeds the "
                        f"{MAX_TOTAL_IMAGE_BYTES_PER_EXAMPLE}-byte limit"
                    )
                with renderer_base._open_verified_image(image_data) as image:
                    example_pixels += image.width * image.height
                if example_pixels > MAX_TOTAL_IMAGE_PIXELS_PER_EXAMPLE:
                    raise ValueError(
                        "aggregate decoded image size exceeds the "
                        f"{MAX_TOTAL_IMAGE_PIXELS_PER_EXAMPLE}-pixel limit"
                    )
            except ValueError as exc:
                raise ValueError(
                    f"Dataset example {example_index} image input is invalid: {exc}"
                ) from None
        total_images += len(references)
        total_bytes += example_bytes
        total_pixels += example_pixels

    summary = ImageDatasetSummary(
        examples=len(dataset),
        images=total_images,
        encoded_bytes=total_bytes,
        decoded_pixels=total_pixels,
    )
    logger.info(
        "Prepared image dataset: examples=%d images=%d encoded_bytes=%d decoded_pixels=%d",
        summary.examples,
        summary.images,
        summary.encoded_bytes,
        summary.decoded_pixels,
    )
    return summary


def prompt_messages(row: dict[str, Any]) -> list[Message]:
    return cast(list[Message], row["prompt_messages"])


def sft_messages(row: dict[str, Any]) -> list[Message]:
    return [
        *prompt_messages(row),
        {"role": "assistant", "content": str(row["answer"]).strip()},
    ]


def answers_match(candidate: Any, expected: Any) -> bool:
    return normalize_answer(candidate) == normalize_answer(expected)