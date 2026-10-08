from __future__ import annotations

import json
import hashlib
import logging
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import chz
from datasets import load_dataset

from interactive_training import model_info, tokenizer_utils
from interactive_training.recipes.tool_ner_rl.ner_task import (
    ENTITY_LABELS,
    GoldEntity,
    NerEpisode,
    NerTask,
    ToolVariant,
)
from interactive_training.renderers import get_renderer
from interactive_training.renderers.base import Message, Renderer
from interactive_training.rl.message_env import EnvFromMessageEnv
from interactive_training.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from interactive_training.tool_use import AgentToolMessageEnv

logger = logging.getLogger(__name__)

DATASET_NAME = "ai4privacy/pii-masking-openpii-1.5m"
DEFAULT_DATASET_REVISION = "f6a1e0a543a22693bd541f86f36c74ab2ca8eaad"
MAX_SAFE_VALIDATION_EXAMPLES = 500
MAX_SAFE_TRAIN_SCORING_EXAMPLES = 10_000
MAX_SAFE_EVAL_EXAMPLES = MAX_SAFE_VALIDATION_EXAMPLES
MAX_SAFE_POOL_EXAMPLES = 12_500
_PARQUET_SHARDS = {"train": 8, "validation": 2}
FEWSHOT_TRAIN_UIDS = frozenset({"24854336", "24790226", "24829919"})

LABEL_DEFINITIONS: dict[str, str] = {
    "AGE": "The age value only, excluding words such as 'age' and 'years old'.",
    "BUILDINGNUM": "The leading building number in an address, separate from STREET.",
    "CITY": "The complete annotated city/locality phrase; it may contain several words.",
    "CREDITCARDNUMBER": "The complete payment-card number, preserving visible separators.",
    "DATE": "The complete date expression, including ordinal words, month, year, and time when present.",
    "DRIVERLICENSENUM": "A driver's-licence identifier, selected only when context identifies that document type.",
    "EMAIL": "The complete email address, excluding adjacent punctuation.",
    "GENDER": "A gender-identity descriptor such as Agender, Bigender, or Genderfluid; distinct from SEX.",
    "GIVENNAME": "The complete annotated given-name span; it may include multiple given or middle names.",
    "IDCARDNUM": "A national, government, or generic ID-card number; not a passport, tax, social, or licence number.",
    "PASSPORTNUM": "A passport identifier, selected only when passport context is present.",
    "SEX": "The recorded sex value, often M, F, Male, or Female; distinct from a GENDER identity descriptor.",
    "SOCIALNUM": "A social-security or national-social-insurance number.",
    "STREET": "The complete street/road name, excluding a separate leading BUILDINGNUM.",
    "SURNAME": "The complete annotated family-name span; it may be multiword or hyphenated.",
    "TAXNUM": "A tax-identification number, selected only when tax context is present.",
    "TELEPHONENUM": "The complete telephone number, preserving visible punctuation and internal spaces.",
    "TITLE": "A personal honorific or title such as Mr, Mrs, Miss, Mstr, Master, or Madame.",
    "ZIPCODE": "The complete postal or ZIP code, separate from CITY and STREET.",
}

TAB_LABEL_DEFINITIONS: dict[str, str] = {
    "CODE": "The identifying number or code itself, excluding introductory words such as 'no.'.",
    "DATETIME": "A complete date, time, duration, or age expression.",
    "DEM": "A demographic attribute such as nationality, ethnicity, religion, or occupation.",
    "LOC": "A complete location, address, country, city, or geographic feature.",
    "MISC": "An identifying entity that does not fit another TAB semantic type.",
    "ORG": "A complete organization, institution, company, court, or public body name.",
    "PERSON": "A complete person name, including titles or initials when part of the mention.",
    "QUANTITY": "A monetary amount, percentage, measurement, or other quantity.",
}
LABEL_DEFINITIONS = {label: LABEL_DEFINITIONS[label] for label in ENTITY_LABELS}

PromptVersion = Literal["fewshot_fast"]

_COMPACT_PROMPT = """You are an exact-span named entity recognition agent.
Extract every entity in the immutable source document using the supplied schema.

Act efficiently:
1. Call discovery tools immediately. Make independent find_text calls together in one turn.
2. Use inspect or select_occurrence only for ambiguous occurrences.
3. Call finalize exactly once in a later turn with all predictions.

Only finalize span IDs returned in this rollout. Use the narrowest exact source span and its correct label.
Do not invent or alter source text, offsets, span IDs, or labels. Do not repeat the document or schema in prose.
Keep reasoning brief so the complete extraction fits within the turn budget."""

# All examples below are exact excerpts from the pinned OpenPII training split,
# never validation. They teach dataset-specific labels and boundaries without
# exposing evaluation answers. UIDs: 24854336, 24790226, 24829919.
_ANNOTATION_EXAMPLES = """

OpenPII annotation examples:

Example 1 (training UID 24854336) demonstrates the complete ontology with explicit cues.
Source:
Registration Form:
Name: Miss Lahcène Gutberlet
Date of Birth: 18th March 2015
Age: 69
Sex: M
Gender: Agender
Email: SH14@tutanota.com
Phone: 016.469 5793
Address: 1 Jalan Kemajuan, Singapore Shell Paya Lebar Macpherson Shell Paya Lebar Macpherson 508986
ID Card No.: K6XXGF7XZ6
Passport No.: NQ3745598
Driver's License No.: V6MKMLIFGP
Tax ID: T4780987T
Social Security No.: 9915753619
Payment Method: Credit Card ending with 6761882948790

Correct entity texts and boundaries:
TITLE="Miss"; GIVENNAME="Lahcène"; SURNAME="Gutberlet"; DATE="18th March 2015";
AGE="69"; SEX="M"; GENDER="Agender"; EMAIL="SH14@tutanota.com";
TELEPHONENUM="016.469 5793"; BUILDINGNUM="1"; STREET="Jalan Kemajuan";
CITY="Singapore Shell Paya Lebar Macpherson Shell Paya Lebar Macpherson";
ZIPCODE="508986"; IDCARDNUM="K6XXGF7XZ6"; PASSPORTNUM="NQ3745598";
DRIVERLICENSENUM="V6MKMLIFGP"; TAXNUM="T4780987T"; SOCIALNUM="9915753619";
CREDITCARDNUMBER="6761882948790".

Example 2 (training UID 24790226) demonstrates multiword name boundaries.
Source excerpt: Mrs Sevina Tringe Samitsch Ghazraoui sends a reminder to 27A@yahoo.com.
Correct: TITLE="Mrs"; GIVENNAME="Sevina Tringe"; SURNAME="Samitsch Ghazraoui";
EMAIL="27A@yahoo.com". Do not split either multiword name.

Example 3 (training UID 24829919) distinguishes GENDER from SEX.
Source: According to the internal survey, Genderfluid representation rose to 48% and M distribution remained balanced.
Correct: GENDER="Genderfluid"; SEX="M". The percentage is not part of either span."""

_FEWSHOT_PROMPT = _COMPACT_PROMPT + _ANNOTATION_EXAMPLES + """

For a new document, first identify candidate text and labels semantically. Prefer one
ground_entities call containing all proposals. Use returned unique span IDs directly;
disambiguate repeated text with inspect or select_occurrence; then call finalize once."""

_FEWSHOT_FAST_PROMPT = _FEWSHOT_PROMPT + """

Strict latency contract for this run: exactly two assistant turns are available.
Turn 1: call ground_entities exactly once with every candidate proposal.
Turn 2: call finalize exactly once using the returned span IDs.
No intermediate lookup or inspection turn is available. Ambiguous grounding results include
local context; choose the intended IDs directly in finalize. If a proposal is not_found, omit
it rather than retrying. Keep both calls concise and emit no prose."""

_TAB_PROMPT = """You anonymize a legal document for the person specified in the task.

TAB anonymization policy:
- DIRECT identifiers uniquely identify the target, such as their name or case number.
- QUASI identifiers can identify the target in combination: personal dates, locations,
    occupations, relationships, organizations, events, or amounts linked to their case.
- NO_MASK mentions do not help identify the target. Do not mask an entity merely because
    it is a named entity. General laws, article numbers, treaty names, and generic court or
    country references are normally background, unless they identify the target in context.
- Use the full document to resolve the target's circumstances and relationships. Extract
    only complete mentions present in the active source window, including repeated mentions.
- Preserve the complete mention as written, including a person's title when present.
- Use only the supplied native TAB labels. Do not split PERSON into given and family names.

Ground copied source text with ground_entities, then finalize the returned span IDs.
Grounding verifies location, not whether masking is correct. Select IDs based on context.
Group IDs with the same label into one finalize entry. Omit empty label groups.
For no predictions, use ground_entities(proposals=[]) then finalize(entities=[]).
Never invent offsets or span IDs. Emit only tool calls, without prose."""

_TAB_TRAIN_EXAMPLE = """

Annotation example from TAB TRAIN document 001-90194 (not dev):
Target: Henrik Hasslund.
Source excerpts:
The case originated in an application (no. 36244/06) against the Kingdom of Denmark
lodged with the Court under Article 34 of the Convention for the Protection of Human
Rights and Fundamental Freedoms by a Danish national, Mr Henrik Hasslund, on 31 August 2006.
The applicant was represented by Mr Tyge Trier, a lawyer practising in Copenhagen.
The Danish Government were represented by their Agent, Ms Nina Holst-Christensen
of the Ministry of Justice.
Mask: CODE="36244/06"; PERSON="Mr Henrik Hasslund"; DATETIME="31 August 2006";
PERSON="Mr Tyge Trier"; PERSON="Ms Nina Holst-Christensen".
Do not mask in this example: Kingdom of Denmark; Danish; Copenhagen; Danish Government;
Ministry of Justice; generic Court, Convention and Article 34 references.
Do not include 'no.' in CODE. Do include the title in a named PERSON mention.
Generic 'the applicant' is not a person name. Named people linked to the case may be
quasi-identifiers even when they are not the target. Decide using the actual document;
the unmasked categories in this example are not universal exclusions.
"""

_TAB_FAST_PROMPT = _TAB_PROMPT + _TAB_TRAIN_EXAMPLE + """

Strict latency contract for this run: exactly two assistant turns are available.
Turn 1: call ground_entities exactly once with every candidate proposal.
Turn 2: call finalize exactly once using the returned span IDs.
No intermediate lookup or inspection turn is available. Ambiguous grounding results include
local context; choose the intended IDs directly in finalize. If a proposal is not_found, omit
it rather than retrying. Keep both calls concise and emit no prose."""

def _parse_privacy_mask(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [cast(dict[str, Any], item) for item in value if isinstance(item, dict)]


def openpii_row_to_task(row: dict[str, Any]) -> NerTask | None:
    source_text = row.get("source_text")
    raw_uid = row.get("uid") or row.get("id")
    language = row.get("language", "")
    if (
        not isinstance(source_text, str)
        or not source_text
        or not isinstance(raw_uid, (str, int))
    ):
        return None
    uid = str(raw_uid)

    entities: set[GoldEntity] = set()
    for item in _parse_privacy_mask(row.get("privacy_mask")):
        start = item.get("start")
        end = item.get("end")
        label = item.get("label")
        value = item.get("value")
        if not isinstance(start, int) or not isinstance(end, int) or not isinstance(label, str):
            return None
        if label not in LABEL_DEFINITIONS or not 0 <= start < end <= len(source_text):
            return None
        if isinstance(value, str) and source_text[start:end] != value:
            return None
        entities.add(GoldEntity(start=start, end=end, label=label))

    if not entities:
        return None
    return NerTask(
        uid=uid,
        source_text=source_text,
        labels=tuple(LABEL_DEFINITIONS),
        gold_entities=frozenset(entities),
        language=str(language),
    )


def tasks_from_rows(
    rows: Iterable[dict[str, Any]],
    *,
    max_examples: int,
    language: str = "en",
    max_documents_scanned: int | None = None,
) -> list[NerTask]:
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    scan_limit = max_documents_scanned or max(100, max_examples * 100)
    tasks: list[NerTask] = []
    scanned = 0
    rejected = 0
    for row in rows:
        scanned += 1
        if scanned > scan_limit:
            break
        if row.get("language") != language:
            continue
        task = openpii_row_to_task(row)
        if task is None:
            rejected += 1
            continue
        tasks.append(task)
        if len(tasks) >= max_examples:
            break

    logger.info(
        "OpenPII scan complete: scanned=%d retained=%d rejected=%d language=%s",
        scanned,
        len(tasks),
        rejected,
        language,
    )
    if len(tasks) < max_examples:
        raise RuntimeError(
            f"Only found {len(tasks)} valid {language!r} rows after scanning {scanned}; "
            f"requested {max_examples}"
        )
    return tasks


def load_openpii_tasks(
    split: Literal["train", "validation"],
    *,
    max_examples: int,
    language: str = "en",
    revision: str = DEFAULT_DATASET_REVISION,
    max_documents_scanned: int | None = None,
) -> list[NerTask]:
    if max_examples > MAX_SAFE_POOL_EXAMPLES:
        raise ValueError(
            f"OpenPII loading is capped at {MAX_SAFE_POOL_EXAMPLES} examples; "
            f"requested {max_examples}"
        )
    logger.info(
        "Streaming filtered parquet for %s split=%s revision=%s; retaining %d %s rows",
        DATASET_NAME,
        split,
        revision,
        max_examples,
        language,
    )
    base_url = (
        f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/{revision}"
        f"/default/{split}"
    )
    data_files = [
        f"{base_url}/{index:04d}.parquet" for index in range(_PARQUET_SHARDS[split])
    ]
    rows = load_dataset(
        "parquet",
        data_files={split: data_files},
        split=split,
        streaming=True,
        filters=[("language", "==", language)],
    )
    return tasks_from_rows(
        cast(Iterable[dict[str, Any]], rows),
        max_examples=max_examples,
        language=language,
        max_documents_scanned=max_documents_scanned,
    )


def stratified_sample_tasks(
    tasks: list[NerTask],
    *,
    sample_size: int,
    seed: int,
) -> list[NerTask]:
    """Select a deterministic sample across entity-count quartiles."""
    if not 1 <= sample_size <= len(tasks):
        raise ValueError("sample_size must be positive and no larger than the task pool")

    ordered = sorted(tasks, key=lambda task: (len(task.gold_entities), task.uid))
    quartiles = [
        ordered[(index * len(ordered)) // 4 : ((index + 1) * len(ordered)) // 4]
        for index in range(4)
    ]
    base, remainder = divmod(sample_size, 4)
    selected: list[NerTask] = []
    remaining: list[NerTask] = []
    for index, quartile in enumerate(quartiles):
        count = base + int(index < remainder)
        ranked = sorted(
            quartile,
            key=lambda task: hashlib.sha256(f"{seed}:{task.uid}".encode()).digest(),
        )
        selected.extend(ranked[:count])
        remaining.extend(ranked[count:])
    remaining.sort(key=lambda task: hashlib.sha256(f"{seed}:{task.uid}".encode()).digest())
    selected.extend(remaining[:sample_size - len(selected)])
    assert len(selected) == sample_size
    return sorted(
        selected,
        key=lambda task: hashlib.sha256(f"manifest:{seed}:{task.uid}".encode()).digest(),
    )


def system_prompt(version: PromptVersion, *, dataset_name: str = "openpii") -> str:
    if version != "fewshot_fast":
        raise ValueError(f"Unknown prompt version: {version}")
    if dataset_name == "tab":
        return _TAB_FAST_PROMPT
    return _FEWSHOT_FAST_PROMPT


def task_prompt(task: NerTask) -> str:
    definitions = TAB_LABEL_DEFINITIONS if task.dataset_name == "tab" else LABEL_DEFINITIONS
    schema = "\n".join(
        f"- {label}: {definitions[label]}" for label in task.labels
    )
    if task.dataset_name == "tab":
        scope = (
            f"{task.task_instruction}\n\n"
            "Task scope: Extract every DIRECT or QUASI identifier that should be masked "
            "under TAB policy. Exclude NO_MASK mentions.\n\n"
            f"Full document (context only, not a discovery source):\n"
            f"<context>\n{task.document_context}\n</context>\n\n"
            "The source document below is the active window. Only submit complete "
            "mentions from that window.\n\n"
        )
    else:
        scope = "Task scope: Extract every entity matching the complete schema.\n\n"
    return (
        f"{scope}Entity schema:\n{schema}\n\n"
        f"Source document:\n<document>\n{task.source_text}\n</document>"
    )


def initial_messages(
    task: NerTask,
    renderer: Renderer,
    episode: Any,
    *,
    prompt_version: PromptVersion,
    tool_variant: ToolVariant,
    tools: list[Any] | None = None,
) -> list[Message]:
    if tools is None:
        tools = episode.tools_for_variant(tool_variant)
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=[tool.to_spec() for tool in tools],
        system_prompt=system_prompt(prompt_version, dataset_name=task.dataset_name),
    )
    return prefix + [{"role": "user", "content": task_prompt(task)}]


class NerLoggingEnv(EnvFromMessageEnv):
    """Add decoded assistant/tool messages to trajectory diagnostics."""

    message_env: AgentToolMessageEnv
    episode: Any

    def __init__(
        self,
        *args: Any,
        episode: Any,
        second_turn_max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.episode = episode
        self.second_turn_max_tokens = second_turn_max_tokens
        self.turn_count = 0

    async def step(self, action: Action) -> StepResult:
        self.turn_count += 1
        history_start = len(self.message_env.history)
        result = await super().step(action)
        if (
            not result.episode_done
            and self.turn_count >= 1
            and self.second_turn_max_tokens is not None
        ):
            result.next_max_tokens = self.second_turn_max_tokens
        if result.metrics.get("parse_error"):
            _, reward_metrics = await self.message_env.reward_fn(self.message_env.history)
            result.metrics.update(reward_metrics)
            result.logs["raw_response"] = self.renderer.tokenizer.decode(action)
        result.logs["messages"] = json.dumps(
            self.message_env.history[history_start:],
            ensure_ascii=False,
            default=str,
        )
        return result


@chz.chz
class NerEnvGroupBuilder(EnvGroupBuilder):
    task: NerTask
    model_name: str
    renderer_name: str | None
    group_size: int
    max_turns: int = 2
    prompt_version: PromptVersion = "fewshot_fast"
    tool_variant: ToolVariant = "fast"
    max_trajectory_tokens: int = 16 * 1024
    max_results: int = 50
    second_turn_max_tokens: int | None = None

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name, prefer_non_thinking=True
        )
        renderer = get_renderer(renderer_name, tokenizer)
        envs: list[Env] = []
        for _ in range(self.group_size):
            episode = NerEpisode(self.task, max_results=self.max_results)
            tools = episode.tools_for_variant(self.tool_variant)
            messages = initial_messages(
                self.task,
                renderer,
                episode,
                prompt_version=self.prompt_version,
                tool_variant=self.tool_variant,
                tools=tools,
            )
            message_env = AgentToolMessageEnv(
                tools=tools,
                initial_messages=messages,
                max_turns=self.max_turns,
                reward_fn=episode.reward,
            )
            envs.append(
                NerLoggingEnv(
                    renderer=renderer,
                    message_env=message_env,
                    episode=episode,
                    second_turn_max_tokens=self.second_turn_max_tokens,
                    failed_parse_reward=-0.1,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                    tool_specs=[tool.to_spec() for tool in tools],
                )
            )
        return envs

    def logging_tags(self) -> list[str]:
        return [
            self.task.dataset_name,
            self.task.language,
            "full_document",
            self.prompt_version,
            self.tool_variant,
        ]


class NerRLDataset(RLDataset):
    def __init__(self, builders: list[NerEnvGroupBuilder], batch_size: int) -> None:
        if not builders:
            raise ValueError("NER dataset must contain at least one environment group")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        return self.builders[start : start + self.batch_size]

    def __len__(self) -> int:
        return (len(self.builders) + self.batch_size - 1) // self.batch_size


def build_eval_dataset(
    tasks: list[NerTask],
    *,
    model_name: str,
    renderer_name: str | None,
    batch_size: int,
    samples_per_task: int,
    max_turns: int,
    prompt_version: PromptVersion,
    tool_variant: ToolVariant,
    second_turn_max_tokens: int | None = None,
) -> NerRLDataset:
    builders = [
        NerEnvGroupBuilder(
            task=task,
            model_name=model_name,
            renderer_name=renderer_name,
            group_size=samples_per_task,
            max_turns=max_turns,
            prompt_version=prompt_version,
            tool_variant=tool_variant,
            second_turn_max_tokens=second_turn_max_tokens,
        )
        for task in tasks
    ]
    return NerRLDataset(builders, batch_size)


@chz.chz
class NerDatasetBuilder(RLDatasetBuilder):
    """Build filtered training groups and random, unfiltered validation groups."""

    model_name_for_tokenizer: str
    batch_size: int
    group_size: int
    train_filter_manifest_path: str
    renderer_name: str | None = None
    dataset_revision: str = DEFAULT_DATASET_REVISION
    language: str = "en"
    seed: int = 42
    train_source_pool_size: int = 2_000
    validation_pool_size: int = 1_000
    max_test_examples: int = 200
    max_train_examples: int | None = None
    max_turns: int = 2
    prompt_version: PromptVersion = "fewshot_fast"
    tool_variant: ToolVariant = "fast"
    second_turn_max_tokens: int = 768
    max_trajectory_tokens: int = 16 * 1024
    tab_fraction: float = 0.0
    mixture_manifest_path: str | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        from interactive_training.recipes.tool_ner_rl.offline_filter import eligible_uids

        manifest_path = Path(self.train_filter_manifest_path).expanduser()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        allowed_uids = eligible_uids(manifest)
        if not allowed_uids:
            raise ValueError("Training filter manifest contains no eligible UIDs")

        load_count = self.train_source_pool_size + len(FEWSHOT_TRAIN_UIDS)
        train_pool = load_openpii_tasks(
            "train",
            max_examples=load_count,
            language=self.language,
            revision=self.dataset_revision,
            max_documents_scanned=max(load_count + 500, load_count),
        )
        train_pool = [
            task for task in train_pool if task.uid not in FEWSHOT_TRAIN_UIDS
        ][: self.train_source_pool_size]
        train_by_uid = {task.uid: task for task in train_pool}
        missing = sorted(allowed_uids - set(train_by_uid))
        if missing:
            raise ValueError(
                f"Training filter contains {len(missing)} UIDs outside the loaded pool; "
                f"first missing UIDs: {missing[:5]}"
            )
        train_tasks = [train_by_uid[uid] for uid in sorted(allowed_uids)]
        random.Random(self.seed).shuffle(train_tasks)
        if self.max_train_examples is not None:
            train_tasks = train_tasks[:self.max_train_examples]
        if self.tab_fraction:
            from interactive_training.recipes.tool_ner_rl.tab_data import (
                TAB_REVISION, TAB_TRAIN_SHA256, load_tab_tasks, mix_tab_training_tasks,
            )
            tab_train = load_tab_tasks(split="train")
            train_tasks = mix_tab_training_tasks(
                train_tasks, tab_train.tasks, fraction=self.tab_fraction, seed=self.seed,
            )
            if self.mixture_manifest_path:
                Path(self.mixture_manifest_path).write_text(json.dumps({
                    "tab_revision": TAB_REVISION,
                    "tab_train_sha256": TAB_TRAIN_SHA256,
                    "tab_train_documents": len(tab_train.selected_document_uids),
                    "tab_excluded_documents": tab_train.excluded_documents,
                    "requested_tab_fraction": self.tab_fraction,
                    "sampling": "document_then_annotator_then_window",
                    "counts_before_pods": {
                        name: sum(task.dataset_name == name for task in train_tasks)
                        for name in ("openpii", "tab")
                    },
                    "task_uids": [task.uid for task in train_tasks],
                }, indent=2), encoding="utf-8")
        train_builders = [
            NerEnvGroupBuilder(
                task=task,
                model_name=self.model_name_for_tokenizer,
                renderer_name=self.renderer_name,
                group_size=self.group_size,
                max_turns=self.max_turns,
                prompt_version=self.prompt_version,
                tool_variant=self.tool_variant,
                second_turn_max_tokens=self.second_turn_max_tokens,
                max_trajectory_tokens=self.max_trajectory_tokens,
            )
            for task in train_tasks
        ]
        train_dataset = NerRLDataset(train_builders, self.batch_size)

        validation_pool = load_openpii_tasks(
            "validation",
            max_examples=self.validation_pool_size,
            language=self.language,
            revision=self.dataset_revision,
            max_documents_scanned=self.validation_pool_size + 500,
        )
        eval_count = min(self.max_test_examples, len(validation_pool))
        test_tasks = random.Random(self.seed + 1).sample(validation_pool, eval_count)
        test_dataset = build_eval_dataset(
            test_tasks,
            model_name=self.model_name_for_tokenizer,
            renderer_name=self.renderer_name,
            batch_size=self.batch_size,
            samples_per_task=1,
            max_turns=self.max_turns,
            prompt_version=self.prompt_version,
            tool_variant=self.tool_variant,
            second_turn_max_tokens=self.second_turn_max_tokens,
        )
        return train_dataset, test_dataset