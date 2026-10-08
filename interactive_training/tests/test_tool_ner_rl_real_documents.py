"""NER tool contracts against exact OpenPII training excerpts.

The embedded examples are synthetic PII from
``ai4privacy/pii-masking-openpii-1.5m`` (CC-BY-4.0), converted-parquet revision
``f6a1e0a543a22693bd541f86f36c74ab2ca8eaad``. UIDs are recorded beside each
fixture. Tests are network-free and never use validation rows.
"""

from __future__ import annotations

import json

import pytest

from interactive_training.recipes.tool_ner_rl.ner_env import LABEL_DEFINITIONS
from interactive_training.recipes.tool_ner_rl.ner_task import GoldEntity, NerEpisode, NerTask
from interactive_training.renderers.qwen3_5 import Qwen3_5DisableThinkingRenderer
from interactive_training.tool_use import ToolInput


# OpenPII training UID 24854336. This profile has explicit cues for all 19 labels.
PROFILE_SOURCE = """Registration Form:
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
Preferred Class: Beginner/Intermediate/Advanced (please tick)
Payment Method: Credit Card ending with 6761882948790
Signature: _____________________"""

PROFILE_ENTITIES = [
    ("TITLE", "Miss"),
    ("GIVENNAME", "Lahcène"),
    ("SURNAME", "Gutberlet"),
    ("DATE", "18th March 2015"),
    ("AGE", "69"),
    ("SEX", "M"),
    ("GENDER", "Agender"),
    ("EMAIL", "SH14@tutanota.com"),
    ("TELEPHONENUM", "016.469 5793"),
    ("BUILDINGNUM", "1"),
    ("STREET", "Jalan Kemajuan"),
    ("CITY", "Singapore Shell Paya Lebar Macpherson Shell Paya Lebar Macpherson"),
    ("ZIPCODE", "508986"),
    ("IDCARDNUM", "K6XXGF7XZ6"),
    ("PASSPORTNUM", "NQ3745598"),
    ("DRIVERLICENSENUM", "V6MKMLIFGP"),
    ("TAXNUM", "T4780987T"),
    ("SOCIALNUM", "9915753619"),
    ("CREDITCARDNUMBER", "6761882948790"),
]

# OpenPII training UID 24790226. The annotation keeps both name parts multiword.
MULTIWORD_NAME_SOURCE = (
    "Before the quarterly review, Mrs Sevina Tringe Samitsch Ghazraoui sends a reminder "
    "to 27A@yahoo.com and attaches the performance dashboard for the team based in 460546."
)

# OpenPII training UID 24670011. The same CITY surface occurs in distinct contexts.
REPEATED_CITY_SOURCE = (
    "Mstr Özgün Grigoletti writes: I was strolling along Tras Street on November/62 when I "
    "discovered an old leather‑bound diary tucked beneath a park bench. The first entry, dated "
    "10/05/1960, recalls a childhood memory of a monsoon‑filled night in Singapore that echoes "
    "a story my Not specified aunt, born on March 11th, 1946, used to tell. I am currently 40 "
    "years old, and the looping cursive reminds me of my late grandfather whose YS0ZZQCJMF "
    "ends in 7890. If this diary belongs to you, please contact me at 1998WIB@protonmail.com or "
    "call 0065 724659955; I would gladly arrange its safe return to an address in Singapore 420002."
)

# OpenPII training UID 24785854. Phone punctuation is part of the source span.
FORMATTED_PHONE_SOURCE = (
    "Dear Madame Swee Leyde Krishowkumar, we have reserved the ceremony at Jalan Mas Puteh 54, "
    "Singapore 799475 on 6th June 1969; please confirm via (00).3642.3020 or A@tutanota.com."
)


def _nth_range(source: str, text: str, occurrence: int = 0) -> tuple[int, int]:
    start = -1
    for _ in range(occurrence + 1):
        start = source.index(text, start + 1)
    return start, start + len(text)


def _profile_task() -> NerTask:
    entities = []
    for label, text in PROFILE_ENTITIES:
        if label == "SEX":
            start = PROFILE_SOURCE.index("Sex: M") + len("Sex: ")
            end = start + 1
        else:
            start, end = _nth_range(PROFILE_SOURCE, text)
        entities.append(GoldEntity(start, end, label))
    return NerTask(
        uid="24854336",
        source_text=PROFILE_SOURCE,
        labels=tuple(LABEL_DEFINITIONS),
        gold_entities=frozenset(entities),
    )


async def _run_tool(tool, **arguments):
    result = await tool.run(ToolInput(arguments=arguments, call_id="real-doc-call"))
    return result, json.loads(result.messages[0]["content"])


@pytest.mark.asyncio
async def test_ground_entities_and_finalize_complete_real_profile():
    episode = NerEpisode(_profile_task())
    _, grounded = await _run_tool(
        episode.ground_entities,
        proposals=[{"label": label, "text": text} for label, text in PROFILE_ENTITIES],
    )

    assert len(grounded["results"]) == 19
    assert {result["status"] for result in grounded["results"]} <= {"unique", "ambiguous"}
    assert next(result for result in grounded["results"] if result["label"] == "SEX")[
        "status"
    ] == "ambiguous"

    submissions = []
    for result in grounded["results"]:
        if result["label"] == "SEX":
            _, selected = await _run_tool(
                episode.select_occurrence,
                span_ids=[match["span_id"] for match in result["matches"]],
                context_contains="Sex: M",
                context_chars=8,
            )
            span_ids = [selected["matches"][0]["span_id"]]
        else:
            span_ids = [result["matches"][0]["span_id"]]
        submissions.append({"label": result["label"], "span_ids": span_ids})

    final_result, final_payload = await _run_tool(episode.finalize, entities=submissions)
    reward, metrics = await episode.reward([])

    assert final_result.should_stop is True
    assert final_payload == {"accepted": True, "entity_count": 19, "rejected": {}}
    assert reward == pytest.approx(1.0)
    assert metrics["exact_true_positives"] == 19.0
    assert metrics["ground_entities_calls"] == 1.0
    assert metrics["select_occurrence_calls"] == 1.0


@pytest.mark.asyncio
async def test_invalid_label_does_not_discard_other_grounding_proposals():
    episode = NerEpisode(_profile_task())

    _, grounded = await _run_tool(
        episode.ground_entities,
        proposals=[
            {"label": "EMAIL", "text": "SH14@tutanota.com"},
            {"label": "TELEPHONUM", "text": "016.469 5793"},
            {"label": "DATE", "text": "18th March 2015"},
        ],
    )

    assert [result["status"] for result in grounded["results"]] == [
        "unique",
        "invalid_label",
        "unique",
    ]
    assert grounded["results"][0]["matches"]
    assert grounded["results"][2]["matches"]


@pytest.mark.asyncio
async def test_find_text_and_inspect_preserve_real_multiword_name_boundary():
    task = NerTask(
        uid="24790226",
        source_text=MULTIWORD_NAME_SOURCE,
        labels=("GIVENNAME", "SURNAME"),
        gold_entities=frozenset(),
    )
    episode = NerEpisode(task)

    _, found = await _run_tool(episode.find_text, text="Sevina Tringe")
    span = found["matches"][0]
    _, inspected = await _run_tool(
        episode.inspect,
        span_ids=[span["span_id"]],
        context_chars=25,
    )

    assert found["match_count"] == 1
    assert span["text"] == "Sevina Tringe"
    assert MULTIWORD_NAME_SOURCE[span["start"] : span["end"]] == "Sevina Tringe"
    assert "Mrs Sevina Tringe Samitsch Ghazraoui" in inspected["spans"][0]["context"]


@pytest.mark.asyncio
async def test_select_occurrence_filters_repeated_real_city_by_context():
    task = NerTask(
        uid="24670011",
        source_text=REPEATED_CITY_SOURCE,
        labels=("CITY",),
        gold_entities=frozenset(),
    )
    episode = NerEpisode(task)
    _, found = await _run_tool(episode.find_text, text="Singapore")

    _, selected = await _run_tool(
        episode.select_occurrence,
        span_ids=[match["span_id"] for match in found["matches"]],
        context_contains="monsoon‑filled night",
        context_chars=40,
    )

    assert found["match_count"] == 2
    assert selected["match_count"] == 1
    assert selected["matches"][0]["start"] == REPEATED_CITY_SOURCE.index("Singapore")


@pytest.mark.asyncio
async def test_find_text_normalization_returns_original_real_phone_span():
    task = NerTask(
        uid="24785854",
        source_text=FORMATTED_PHONE_SOURCE,
        labels=("TELEPHONENUM",),
        gold_entities=frozenset(),
    )
    episode = NerEpisode(task)

    _, found = await _run_tool(
        episode.find_text,
        text="(00) 3642 3020",
        normalization="punctuation_whitespace",
    )

    assert found["match_count"] == 1
    assert found["matches"][0]["text"] == "(00).3642.3020"


def test_grounded_tool_schemas_are_explicit_and_qwen_compatible():
    episode = NerEpisode(_profile_task())
    specs = {tool.name: tool.to_spec() for tool in episode.tools_for_variant("grounded")}

    assert list(specs) == [
        "ground_entities",
        "find_text",
        "inspect",
        "select_occurrence",
        "finalize",
    ]
    assert "find_near" not in specs

    grounding_schema = specs["ground_entities"]["parameters"]
    proposal_schema = grounding_schema["$defs"]["GroundingProposal"]
    assert "minItems" not in grounding_schema["properties"]["proposals"]
    assert grounding_schema["properties"]["proposals"]["maxItems"] == 40
    assert proposal_schema["properties"]["label"]["enum"] == list(LABEL_DEFINITIONS)
    assert proposal_schema["properties"]["normalization"]["default"] == "exact"

    select_schema = specs["select_occurrence"]["parameters"]
    assert select_schema["required"] == ["span_ids", "context_contains"]
    assert select_schema["properties"]["context_chars"]["default"] == 80

    finalize_schema = specs["finalize"]["parameters"]
    submission_schema = finalize_schema["$defs"]["EntitySubmission"]
    assert submission_schema["properties"]["label"]["enum"] == list(LABEL_DEFINITIONS)
    assert submission_schema["properties"]["span_ids"]["minItems"] == 1
    assert "does not reveal correctness" in specs["finalize"]["description"]
    assert [tool.name for tool in episode.tools_for_variant("fast")] == [
        "ground_entities",
        "finalize",
    ]


def test_qwen36_system_message_declares_every_grounded_tool():
    episode = NerEpisode(_profile_task())
    tool_specs = [tool.to_spec() for tool in episode.tools_for_variant("grounded")]
    renderer = Qwen3_5DisableThinkingRenderer(tokenizer=None)  # type: ignore[arg-type]

    prefix = renderer.create_conversation_prefix_with_tools(
        tool_specs,
        system_prompt="Extract exact spans.",
    )

    assert len(prefix) == 1
    system_content = prefix[0]["content"]
    assert isinstance(system_content, str)
    tools_text = system_content.split("<tools>\n", 1)[1].split("\n</tools>", 1)[0]
    declared_specs = [json.loads(line) for line in tools_text.splitlines()]
    assert declared_specs == tool_specs
    assert "<function=example_function_name>" in system_content
    assert system_content.endswith("Extract exact spans.")