from __future__ import annotations

import hashlib
import json
import os
import random
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from interactive_training.recipes.tool_ner_rl.ner_task import (
    GoldEntity,
    GoldEntityAttribute,
    NerTask,
)

TAB_DATASET_NAME = "NorskRegnesentral/text-anonymization-benchmark"
TAB_REVISION = "558e09e26d6b36f5f78440074e6a233946d98bd9"
TAB_DEV_SHA256 = "8c3c7306f46b8d54debeb38ae11d8b0b8bcf4bdccbc3b6f13c12ad7be16893ec"
TAB_TRAIN_SHA256 = "4aba41f8ac305ff9e93dd6f0bbc16756e57e9ace396c827931fab70e18d8c6a6"
TAB_DEV_URL = (
    "https://raw.githubusercontent.com/NorskRegnesentral/"
    f"text-anonymization-benchmark/{TAB_REVISION}/echr_dev.json"
)
TAB_LABELS = (
    "CODE",
    "DATETIME",
    "DEM",
    "LOC",
    "MISC",
    "ORG",
    "PERSON",
    "QUANTITY",
)
TAB_MASK_IDENTIFIER_TYPES = frozenset({"DIRECT", "QUASI"})


@dataclass(frozen=True)
class TabTaskCollection:
    tasks: tuple[NerTask, ...]
    selected_document_uids: tuple[str, ...]
    total_documents: int
    chunk_count: int
    annotation_layer_count: int
    max_gold_entities: int
    excluded_documents: tuple[tuple[str, str], ...] = ()


def _cached_dev_path(split: str = "dev") -> Path:
    cache_root = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    )
    return cache_root / "interactive-post-training" / "tab" / f"echr_{split}-{TAB_REVISION}.json"


def _read_dev_bytes(source_path: str | Path | None, split: str = "dev") -> bytes:
    if split not in {"train", "dev"}:
        raise ValueError("Only TAB train and dev are supported")
    if source_path is not None:
        payload = Path(source_path).expanduser().read_bytes()
    else:
        cache_path = _cached_dev_path(split)
        if cache_path.exists():
            payload = cache_path.read_bytes()
        else:
            with urllib.request.urlopen(TAB_DEV_URL.replace("echr_dev", f"echr_{split}"), timeout=60) as response:
                payload = response.read()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_path.with_suffix(".tmp")
            temporary_path.write_bytes(payload)
            temporary_path.replace(cache_path)

    digest = hashlib.sha256(payload).hexdigest()
    expected = TAB_DEV_SHA256 if split == "dev" else TAB_TRAIN_SHA256
    if digest != expected:
        raise ValueError(
            f"TAB {split} checksum mismatch: expected {expected}, got {digest}"
        )
    return payload


def _chunk_ranges(
    text: str,
    *,
    max_words: int,
    overlap_words: int,
) -> list[tuple[int, int]]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be between 0 and max_words - 1")

    words = list(re.finditer(r"\S+", text))
    if len(words) <= max_words:
        return [(0, len(text))]

    ranges: list[tuple[int, int]] = []
    stride = max_words - overlap_words
    for word_start in range(0, len(words), stride):
        word_end = min(word_start + max_words, len(words))
        start = 0 if word_start == 0 else words[word_start].start()
        end = len(text) if word_end == len(words) else words[word_end].start()
        ranges.append((start, end))
        if word_end == len(words):
            break
    return ranges


def _validate_mention(mention: dict[str, Any], text: str) -> None:
    start = mention["start_offset"]
    end = mention["end_offset"]
    if not 0 <= start < end <= len(text):
        raise ValueError(f"Invalid TAB offsets: {start}:{end}")
    if text[start:end] != mention["span_text"]:
        raise ValueError(
            f"TAB span text mismatch at {start}:{end}: {mention['span_text']!r}"
        )
    if mention["entity_type"] not in TAB_LABELS:
        raise ValueError(f"Unknown TAB entity type: {mention['entity_type']}")


def _select_documents(
    documents: list[dict[str, Any]],
    *,
    max_documents: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if max_documents is None:
        return documents
    if not 1 <= max_documents <= len(documents):
        raise ValueError(
            f"max_documents must be between 1 and {len(documents)}"
        )
    selected = random.Random(seed).sample(documents, max_documents)
    return sorted(selected, key=lambda document: document["doc_id"])


def load_tab_tasks(
    *,
    split: Literal["train", "dev"] = "dev",
    source_path: str | Path | None = None,
    max_documents: int | None = None,
    seed: int = 42,
    max_words: int = 128,
    overlap_words: int = 16,
) -> TabTaskCollection:
    documents = json.loads(_read_dev_bytes(source_path, split))
    if not isinstance(documents, list) or not documents:
        raise ValueError("TAB dev data must be a non-empty JSON list")
    selected_documents = _select_documents(
        documents,
        max_documents=max_documents,
        seed=seed,
    )

    tasks: list[NerTask] = []
    excluded_documents = []
    retained_documents = []
    chunk_count = 0
    annotation_layer_count = 0
    for document in selected_documents:
        document_uid = str(document["doc_id"])
        if document.get("dataset_type", split) != split:
            raise ValueError(f"TAB split mismatch for {document_uid}")
        text = document["text"]
        chunk_ranges = _chunk_ranges(
            text,
            max_words=max_words,
            overlap_words=overlap_words,
        )
        if split == "train":
            unsupported = []
            for annotation in document["annotations"].values():
                for mention in annotation["entity_mentions"]:
                    _validate_mention(mention, text)
                    if mention["identifier_type"] in TAB_MASK_IDENTIFIER_TYPES and (
                        len(mention["span_text"]) > 256
                        or not any(start <= mention["start_offset"] and mention["end_offset"] <= end
                                   for start, end in chunk_ranges)
                    ):
                        unsupported.append(mention["entity_mention_id"])
            if unsupported:
                excluded_documents.append((document_uid, "unsupported_masked_spans:" + ",".join(unsupported)))
                continue
        retained_documents.append(document_uid)
        chunk_count += len(chunk_ranges)
        for annotation_layer, annotation in sorted(document["annotations"].items()):
            annotation_layer_count += 1
            mentions = annotation["entity_mentions"]
            for mention in mentions:
                _validate_mention(mention, text)

            owned_mentions: list[list[dict[str, Any]]] = [
                [] for _ in chunk_ranges
            ]
            for mention in mentions:
                if mention["identifier_type"] not in TAB_MASK_IDENTIFIER_TYPES:
                    continue
                contained = False
                for chunk_index, (chunk_start, chunk_end) in enumerate(chunk_ranges):
                    if (
                        chunk_start <= mention["start_offset"]
                        and mention["end_offset"] <= chunk_end
                    ):
                        owned_mentions[chunk_index].append(mention)
                        contained = True
                if not contained:
                    raise ValueError(
                        "TAB mention is not fully contained by any chunk: "
                        f"{document_uid}/{annotation_layer}/"
                        f"{mention['entity_mention_id']}"
                    )

            for chunk_index, (chunk_start, chunk_end) in enumerate(chunk_ranges):
                chunk_mentions = owned_mentions[chunk_index]
                gold_entities = frozenset(
                    GoldEntity(
                        start=mention["start_offset"] - chunk_start,
                        end=mention["end_offset"] - chunk_start,
                        label=mention["entity_type"],
                    )
                    for mention in chunk_mentions
                )
                gold_attributes = tuple(
                    GoldEntityAttribute(
                        start=mention["start_offset"] - chunk_start,
                        end=mention["end_offset"] - chunk_start,
                        label=mention["entity_type"],
                        identifier_type=mention["identifier_type"],
                        entity_id=mention["entity_id"],
                    )
                    for mention in chunk_mentions
                )
                tasks.append(
                    NerTask(
                        uid=(
                            f"tab-{split}:{document_uid}:{annotation_layer}:"
                            f"chunk-{chunk_index:03d}"
                        ),
                        source_text=text[chunk_start:chunk_end],
                        labels=TAB_LABELS,
                        gold_entities=gold_entities,
                        dataset_name="tab",
                        document_uid=document_uid,
                        annotation_layer=annotation_layer,
                        source_start=chunk_start,
                        gold_attributes=gold_attributes,
                        task_instruction=document["task"],
                        document_context=text,
                    )
                )

    return TabTaskCollection(
        tasks=tuple(tasks),
        selected_document_uids=tuple(retained_documents),
        total_documents=len(documents),
        chunk_count=chunk_count,
        annotation_layer_count=annotation_layer_count,
        max_gold_entities=max(len(task.gold_entities) for task in tasks),
        excluded_documents=tuple(excluded_documents),
    )


def load_tab_dev_tasks(**kwargs: Any) -> TabTaskCollection:
    return load_tab_tasks(split="dev", **kwargs)


def mix_tab_training_tasks(
    openpii: list[NerTask], tab: tuple[NerTask, ...], *, fraction: float, seed: int
) -> list[NerTask]:
    if not 0 <= fraction < 1:
        raise ValueError("TAB fraction must be in [0, 1)")
    if not fraction:
        return list(openpii)
    if not openpii or not tab:
        raise ValueError("Both datasets are required for mixed training")
    by_document: dict[str, dict[str, list[NerTask]]] = {}
    for task in tab:
        if not task.uid.startswith("tab-train:"):
            raise ValueError("Only TAB train tasks may enter the training mixture")
        by_document.setdefault(task.document_uid, {}).setdefault(task.annotation_layer, []).append(task)
    rng = random.Random(seed)
    documents = sorted(by_document)
    tab_count = round(len(openpii) * fraction / (1 - fraction))
    selected = []
    for _ in range(tab_count):
        layers = by_document[rng.choice(documents)]
        selected.append(rng.choice(layers[rng.choice(sorted(layers))]))
    mixed = list(openpii) + selected
    rng.shuffle(mixed)
    return mixed