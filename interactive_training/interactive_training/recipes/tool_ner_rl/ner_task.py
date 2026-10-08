from __future__ import annotations

import hashlib
import json
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, Field

from interactive_training.renderers.base import Message
from interactive_training.tool_use import ToolResult, simple_tool_result, tool

Normalization = Literal[
    "exact",
    "case_insensitive",
    "whitespace",
    "punctuation_whitespace",
]
EntityLabel = Literal[
    "AGE",
    "BUILDINGNUM",
    "CITY",
    "CREDITCARDNUMBER",
    "DATE",
    "DRIVERLICENSENUM",
    "EMAIL",
    "GENDER",
    "GIVENNAME",
    "IDCARDNUM",
    "PASSPORTNUM",
    "SEX",
    "SOCIALNUM",
    "STREET",
    "SURNAME",
    "TAXNUM",
    "TELEPHONENUM",
    "TITLE",
    "ZIPCODE",
]
ToolVariant = Literal["basic", "grounded", "fast"]
ENTITY_LABELS = tuple(get_args(EntityLabel))


@dataclass(frozen=True, order=True)
class GoldEntity:
    start: int
    end: int
    label: str


@dataclass(frozen=True, order=True)
class GoldEntityAttribute:
    start: int
    end: int
    label: str
    identifier_type: str
    entity_id: str


@dataclass(frozen=True)
class NerTask:
    uid: str
    source_text: str
    labels: tuple[str, ...]
    gold_entities: frozenset[GoldEntity]
    language: str = "en"
    dataset_name: str = "openpii"
    document_uid: str | None = None
    annotation_layer: str | None = None
    source_start: int = 0
    gold_attributes: tuple[GoldEntityAttribute, ...] = ()
    task_instruction: str = ""
    document_context: str = ""

    def __post_init__(self) -> None:
        if not self.uid:
            raise ValueError("uid must not be empty")
        if not self.source_text:
            raise ValueError("source_text must not be empty")
        if not self.labels:
            raise ValueError("labels must not be empty")
        if self.source_start < 0:
            raise ValueError("source_start must not be negative")
        allowed_labels = set(self.labels)
        for entity in self.gold_entities:
            if not 0 <= entity.start < entity.end <= len(self.source_text):
                raise ValueError(f"Invalid entity offsets: {entity}")
            if entity.label not in allowed_labels:
                raise ValueError(f"Unknown gold label: {entity.label}")
        gold_keys = {
            (entity.start, entity.end, entity.label) for entity in self.gold_entities
        }
        for attribute in self.gold_attributes:
            if (attribute.start, attribute.end, attribute.label) not in gold_keys:
                raise ValueError(f"Gold attribute has no matching entity: {attribute}")


@dataclass(frozen=True)
class Span:
    span_id: str
    text: str
    start: int
    end: int

    def as_dict(self) -> dict[str, str | int]:
        return {
            "span_id": self.span_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
        }


class EntitySubmission(BaseModel):
    label: str = Field(
        description="The entity label from the supplied schema.",
        json_schema_extra={"enum": list(ENTITY_LABELS)},
    )
    span_ids: list[str] = Field(
        min_length=1,
        max_length=50,
        description="One or more span IDs returned by discovery tools in this rollout.",
    )


class GroundingProposal(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=256,
        description="The exact entity text copied from the source document.",
    )
    label: str = Field(
        description="The proposed entity label from the supplied schema.",
        json_schema_extra={"enum": list(ENTITY_LABELS)},
    )
    normalization: Normalization = Field(
        default="exact",
        description=(
            "Matching mode. Prefer exact. Use case_insensitive, whitespace, or "
            "punctuation_whitespace only when the copied text differs in that specific way."
        ),
    )


def _normalized_with_source_map(
    text: str,
    normalization: Normalization,
) -> tuple[str, list[int], list[int]]:
    if normalization == "exact":
        return text, list(range(len(text))), list(range(1, len(text) + 1))

    output: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    collapse_whitespace = normalization in {"whitespace", "punctuation_whitespace"}
    collapse_punctuation = normalization == "punctuation_whitespace"
    casefold = normalization == "case_insensitive"

    for index, character in enumerate(text):
        is_separator = character.isspace() or (
            collapse_punctuation and unicodedata.category(character).startswith("P")
        )
        if collapse_whitespace and is_separator:
            if output and output[-1] == " ":
                ends[-1] = index + 1
            else:
                output.append(" ")
                starts.append(index)
                ends.append(index + 1)
            continue

        transformed = character.casefold() if casefold else character
        for normalized_character in transformed:
            output.append(normalized_character)
            starts.append(index)
            ends.append(index + 1)

    return "".join(output), starts, ends


def _find_source_ranges(
    source: str,
    query: str,
    normalization: Normalization,
) -> list[tuple[int, int]]:
    if not query:
        return []

    normalized_source, starts, ends = _normalized_with_source_map(source, normalization)
    normalized_query, _, _ = _normalized_with_source_map(query, normalization)
    if not normalized_query:
        return []

    ranges: list[tuple[int, int]] = []
    search_from = 0
    while True:
        match_start = normalized_source.find(normalized_query, search_from)
        if match_start < 0:
            break
        match_end = match_start + len(normalized_query)
        source_start = starts[match_start]
        source_end = ends[match_end - 1]
        if not query[0].isspace():
            while source_start < source_end and source[source_start].isspace():
                source_start += 1
        if not query[-1].isspace():
            while source_end > source_start and source[source_end - 1].isspace():
                source_end -= 1
        source_range = (source_start, source_end)
        if not ranges or ranges[-1] != source_range:
            ranges.append(source_range)
        search_from = match_start + 1
    return ranges


@dataclass(frozen=True)
class NerTaskTool:
    delegate: Any
    task: NerTask
    finalization_hint: bool = False

    @property
    def name(self) -> str:
        return self.delegate.name

    @property
    def description(self) -> str:
        return self.delegate.description

    @property
    def parameters_schema(self) -> dict[str, Any]:
        schema = deepcopy(self.delegate.parameters_schema)
        for definition in schema.get("$defs", {}).values():
            label = definition.get("properties", {}).get("label")
            if label is not None:
                label["enum"] = list(self.task.labels)
        proposals = schema.get("properties", {}).get("proposals")
        if proposals is not None:
            proposals["maxItems"] = 64 if self.task.dataset_name == "tab" else 40
        return schema

    def to_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    async def run(self, tool_input: Any) -> ToolResult:
        result = await self.delegate.run(tool_input)
        if self.finalization_hint and self.name == "ground_entities":
            for message in result.messages:
                payload = json.loads(message["content"])
                payload["next_action"] = (
                    "Final turn: call finalize now using only returned span IDs. "
                    "Do not call ground_entities again. Group IDs by label; "
                    "omit empty groups. Use entities=[] if nothing should be masked."
                )
                message["content"] = json.dumps(payload, ensure_ascii=False)
        return result


class NerEpisode:
    """State and tools for one NER rollout."""

    def __init__(
        self,
        task: NerTask,
        *,
        max_results: int = 50,
        max_query_chars: int = 256,
        max_inspect_ids: int = 20,
        max_context_chars: int = 200,
    ) -> None:
        self.task = task
        self.max_results = max_results
        self.max_query_chars = max_query_chars
        self.max_inspect_ids = max_inspect_ids
        self.max_context_chars = max_context_chars
        self._spans: dict[str, Span] = {}
        self._exposed_span_ids: set[str] = set()
        self.predictions: frozenset[GoldEntity] | None = None
        self.invalid_actions = 0
        self.invalid_finalizations = 0
        self.tool_calls = 0
        self.ground_entities_calls = 0
        self.find_text_calls = 0
        self.inspect_calls = 0
        self.select_occurrence_calls = 0
        self.finalize_calls = 0
        self.candidate_spans_returned = 0

    def tools_for_variant(self, variant: ToolVariant) -> list[Any]:
        if variant == "basic":
            tools = [self.find_text, self.inspect, self.finalize]
        elif variant == "grounded":
            tools = [
                self.ground_entities,
                self.find_text,
                self.inspect,
                self.select_occurrence,
                self.finalize,
            ]
        elif variant == "fast":
            tools = [self.ground_entities, self.finalize]
        else:
            raise ValueError(f"Unknown tool variant: {variant}")
        return [
            NerTaskTool(delegate, self.task, variant == "fast" and self.task.dataset_name == "tab")
            for delegate in tools
        ]

    def _span_for(self, start: int, end: int) -> Span:
        digest = hashlib.sha256(
            f"{self.task.uid}\0{start}\0{end}".encode("utf-8")
        ).hexdigest()[:12]
        span_id = f"s_{digest}"
        span = Span(
            span_id=span_id,
            text=self.task.source_text[start:end],
            start=start,
            end=end,
        )
        self._spans[span_id] = span
        return span

    def _error(self, message: str) -> ToolResult:
        self.invalid_actions += 1
        return simple_tool_result(json.dumps({"error": message}))

    def _expose_ranges(self, ranges: list[tuple[int, int]]) -> dict[str, Any]:
        spans = [self._span_for(start, end) for start, end in ranges[: self.max_results]]
        self._exposed_span_ids.update(span.span_id for span in spans)
        self.candidate_spans_returned += len(spans)
        return {
            "matches": [span.as_dict() for span in spans],
            "match_count": len(ranges),
            "truncated": len(ranges) > self.max_results,
        }

    @tool
    async def ground_entities(
        self,
        proposals: Annotated[
            list[GroundingProposal],
            Field(
                max_length=64,
                description=(
                    "Candidate entities identified semantically in the document. Each proposal "
                    "contains copied source text, one schema label, and an optional normalization. "
                    "Use an empty list only when no entities are predicted."
                ),
            ),
        ],
    ) -> ToolResult:
        """Ground many proposed entities in one call; this does not finalize predictions.

        Each result is unique, ambiguous, or not_found. Unique and ambiguous matches expose
        span IDs that may be inspected, selected, or submitted to finalize later.
        """
        self.tool_calls += 1
        self.ground_entities_calls += 1
        parsed = [
            proposal
            if isinstance(proposal, GroundingProposal)
            else GroundingProposal.model_validate(proposal)
            for proposal in proposals
        ]
        results: list[dict[str, Any]] = []
        for index, proposal in enumerate(parsed):
            if proposal.label not in self.task.labels:
                self.invalid_actions += 1
                results.append(
                    {
                        "proposal_index": index,
                        "text": proposal.text,
                        "label": proposal.label,
                        "status": "invalid_label",
                        "matches": [],
                        "match_count": 0,
                        "truncated": False,
                    }
                )
                continue
            ranges = _find_source_ranges(
                self.task.source_text,
                proposal.text,
                proposal.normalization,
            )
            match_payload = self._expose_ranges(ranges)
            match_count = match_payload["match_count"]
            if match_count > 1:
                contextual_matches = []
                for match in match_payload["matches"]:
                    left = max(0, match["start"] - 60)
                    right = min(len(self.task.source_text), match["end"] + 60)
                    contextual_matches.append(
                        {**match, "context": self.task.source_text[left:right]}
                    )
                match_payload["matches"] = contextual_matches
            results.append(
                {
                    "proposal_index": index,
                    "text": proposal.text,
                    "label": proposal.label,
                    "status": (
                        "not_found"
                        if match_count == 0
                        else "unique"
                        if match_count == 1
                        else "ambiguous"
                    ),
                    **match_payload,
                }
            )
        return simple_tool_result(json.dumps({"results": results}, ensure_ascii=False))

    @tool
    async def find_text(
        self,
        text: Annotated[
            str,
            Field(
                min_length=1,
                max_length=256,
                description="Exact entity text copied from the immutable source document.",
            ),
        ],
        normalization: Annotated[
            Normalization,
            "Matching mode. Prefer exact; alternatives preserve original source boundaries.",
        ] = "exact",
    ) -> ToolResult:
        """Find every source span matching text and return opaque span IDs."""
        self.tool_calls += 1
        self.find_text_calls += 1
        if not text:
            return self._error("text must not be empty")
        if len(text) > self.max_query_chars:
            return self._error(f"text exceeds {self.max_query_chars} characters")
        return simple_tool_result(
            json.dumps(
                self._expose_ranges(
                    _find_source_ranges(self.task.source_text, text, normalization)
                ),
                ensure_ascii=False,
            )
        )

    @tool
    async def inspect(
        self,
        span_ids: Annotated[
            list[str],
            Field(
                min_length=1,
                max_length=20,
                description="Span IDs returned by discovery tools in this rollout.",
            ),
        ],
        context_chars: Annotated[
            int,
            Field(
                default=50,
                ge=0,
                le=200,
                description="Number of source characters shown on each side of each span.",
            ),
        ],
    ) -> ToolResult:
        """Show bounded source context around previously returned spans."""
        self.tool_calls += 1
        self.inspect_calls += 1
        if len(span_ids) > self.max_inspect_ids:
            return self._error(f"at most {self.max_inspect_ids} span IDs may be inspected")
        if not 0 <= context_chars <= self.max_context_chars:
            return self._error(
                f"context_chars must be between 0 and {self.max_context_chars}"
            )
        unknown = [span_id for span_id in span_ids if span_id not in self._exposed_span_ids]
        if unknown:
            return self._error(f"unknown or unexposed span IDs: {unknown}")

        contexts = []
        for span_id in span_ids:
            span = self._spans[span_id]
            left = max(0, span.start - context_chars)
            right = min(len(self.task.source_text), span.end + context_chars)
            contexts.append(
                {
                    **span.as_dict(),
                    "context": self.task.source_text[left:right],
                    "span_start_in_context": span.start - left,
                    "span_end_in_context": span.end - left,
                }
            )
        return simple_tool_result(json.dumps({"spans": contexts}, ensure_ascii=False))

    @tool
    async def select_occurrence(
        self,
        span_ids: Annotated[
            list[str],
            Field(
                min_length=1,
                max_length=20,
                description="Ambiguous span IDs previously returned for the same proposed entity.",
            ),
        ],
        context_contains: Annotated[
            str,
            Field(
                min_length=1,
                max_length=256,
                description="Exact nearby source phrase that identifies the intended occurrence.",
            ),
        ],
        context_chars: Annotated[
            int,
            Field(
                default=80,
                ge=0,
                le=200,
                description="Number of source characters examined on each side of each span.",
            ),
        ],
    ) -> ToolResult:
        """Select exposed occurrences whose local context contains an exact phrase.

        Use this only after a discovery result is ambiguous. It filters the supplied IDs and
        returns matching spans with context. It never searches for new entity text.
        """
        self.tool_calls += 1
        self.select_occurrence_calls += 1
        unknown = [span_id for span_id in span_ids if span_id not in self._exposed_span_ids]
        if unknown:
            return self._error(f"unknown or unexposed span IDs: {unknown}")

        matches: list[dict[str, Any]] = []
        for span_id in span_ids:
            span = self._spans[span_id]
            left = max(0, span.start - context_chars)
            right = min(len(self.task.source_text), span.end + context_chars)
            context = self.task.source_text[left:right]
            if context_contains in context:
                matches.append({**span.as_dict(), "context": context})
        return simple_tool_result(
            json.dumps(
                {
                    "matches": matches,
                    "match_count": len(matches),
                    "context_contains": context_contains,
                },
                ensure_ascii=False,
            )
        )

    @tool
    async def finalize(
        self,
        entities: Annotated[
            list[EntitySubmission],
            Field(
                max_length=100,
                description=(
                    "Complete final set of labeled span IDs. Call exactly once, after discovery, "
                    "and include every predicted entity in this one submission."
                ),
            ),
        ],
    ) -> ToolResult:
        """Submit the complete prediction exactly once and end the episode.

        Every span ID must have been returned by a discovery tool in this rollout. This tool
        validates grounding but does not reveal correctness or gold annotations.
        """
        self.tool_calls += 1
        self.finalize_calls += 1
        if self.predictions is not None:
            self.invalid_finalizations += 1
            return self._error("the episode was already finalized")

        parsed = [
            entity if isinstance(entity, EntitySubmission) else EntitySubmission.model_validate(entity)
            for entity in entities
        ]
        allowed_labels = set(self.task.labels)
        invalid_labels = sorted({entity.label for entity in parsed} - allowed_labels)
        unknown_ids = sorted({
            span_id
            for entity in parsed
            for span_id in entity.span_ids
            if span_id not in self._exposed_span_ids
        })
        if invalid_labels or unknown_ids:
            self.invalid_finalizations += 1
            self.invalid_actions += 1

        self.predictions = frozenset(
            GoldEntity(
                start=self._spans[span_id].start,
                end=self._spans[span_id].end,
                label=entity.label,
            )
            for entity in parsed
            for span_id in entity.span_ids
            if entity.label in allowed_labels and span_id in self._exposed_span_ids
        )
        rejected: dict[str, Any] = {}
        if invalid_labels:
            rejected["invalid_labels"] = invalid_labels
        if unknown_ids:
            rejected["unknown_span_ids"] = unknown_ids
        return simple_tool_result(
            json.dumps(
                {
                    "accepted": True,
                    "entity_count": len(self.predictions),
                    "rejected": rejected,
                }
            ),
            should_stop=True,
        )

    async def reward(self, _history: list[Message]) -> tuple[float, dict[str, float]]:
        """Grade the finalized prediction set against exact source spans and labels."""
        finalized = self.predictions is not None
        predictions = self.predictions or frozenset()
        true_positives = len(predictions & self.task.gold_entities)
        precision = (
            true_positives / len(predictions)
            if predictions
            else 1.0 if not self.task.gold_entities else 0.0
        )
        recall = true_positives / len(self.task.gold_entities) if self.task.gold_entities else 1.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        penalty = 0.1 if not finalized or self.invalid_finalizations else 0.0
        reward = f1 - penalty
        return reward, {
            "exact_f1": f1,
            "exact_precision": precision,
            "exact_recall": recall,
            "exact_true_positives": float(true_positives),
            "predicted_entities": float(len(predictions)),
            "gold_entities": float(len(self.task.gold_entities)),
            "phi_false_negative_rate": 1.0 - recall,
            "finalize_valid": float(finalized and self.invalid_finalizations == 0),
            "invalid_actions": float(self.invalid_actions),
            "invalid_finalizations": float(self.invalid_finalizations),
            "tool_calls": float(self.tool_calls),
            "ground_entities_calls": float(self.ground_entities_calls),
            "find_text_calls": float(self.find_text_calls),
            "inspect_calls": float(self.inspect_calls),
            "select_occurrence_calls": float(self.select_occurrence_calls),
            "finalize_calls": float(self.finalize_calls),
            "candidate_spans_returned": float(self.candidate_spans_returned),
        }