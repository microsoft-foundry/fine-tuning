import hashlib
import json

import pytest
from interactive_training.tool_use import ToolInput

from interactive_training.recipes.tool_ner_rl.ner_env import system_prompt, task_prompt
from interactive_training.recipes.tool_ner_rl.ner_task import ENTITY_LABELS, NerEpisode, NerTask
from interactive_training.recipes.tool_ner_rl.tab_data import TAB_LABELS, load_tab_dev_tasks, load_tab_tasks, mix_tab_training_tasks


def _document(doc_id="case-1"):
    text = " ".join(f"word{index}" for index in range(150))
    person = "word120 word121"
    start = text.index(person)
    return {
        "doc_id": doc_id,
        "task": "Task: Anonymise the following person: Example Applicant",
        "text": text,
        "annotations": {
            "annotator1": {
                "entity_mentions": [
                    {
                        "entity_type": "PERSON",
                        "entity_mention_id": f"{doc_id}-mention-1",
                        "start_offset": start,
                        "end_offset": start + len(person),
                        "span_text": person,
                        "identifier_type": "DIRECT",
                        "entity_id": f"{doc_id}-entity-1",
                    },
                    {
                        "entity_type": "ORG",
                        "entity_mention_id": f"{doc_id}-mention-2",
                        "start_offset": 0,
                        "end_offset": len("word0"),
                        "span_text": "word0",
                        "identifier_type": "NO_MASK",
                        "entity_id": f"{doc_id}-entity-2",
                    },
                ]
            },
            "annotator2": {"entity_mentions": []},
        },
    }


def _write_fixture(tmp_path, documents, monkeypatch):
    path = tmp_path / "echr_dev.json"
    path.write_text(json.dumps(documents), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr("interactive_training.recipes.tool_ner_rl.tab_data.TAB_DEV_SHA256", digest)
    monkeypatch.setattr("interactive_training.recipes.tool_ner_rl.tab_data.TAB_TRAIN_SHA256", digest)
    return path


def test_tab_loader_scores_every_visible_overlap_mention(tmp_path, monkeypatch):
    collection = load_tab_dev_tasks(
        source_path=_write_fixture(tmp_path, [_document()], monkeypatch),
        max_words=128,
        overlap_words=16,
    )

    assert collection.chunk_count == 2
    assert collection.annotation_layer_count == 2
    assert len(collection.tasks) == 4
    assert collection.max_gold_entities == 1
    assert all(task.labels == TAB_LABELS for task in collection.tasks)
    assert all(task.dataset_name == "tab" for task in collection.tasks)

    person_tasks = [task for task in collection.tasks if task.gold_entities]
    assert len(person_tasks) == 2
    assert len({
        (task.source_start + entity.start, task.source_start + entity.end, entity.label)
        for task in person_tasks for entity in task.gold_entities
    }) == 1
    task = person_tasks[0]
    entity = next(iter(task.gold_entities))
    attribute = task.gold_attributes[0]
    assert task.source_text[entity.start : entity.end] == "word120 word121"
    assert attribute.identifier_type == "DIRECT"
    assert attribute.entity_id == "case-1-entity-1"
    assert all(entity.label != "ORG" for task in collection.tasks for entity in task.gold_entities)


def test_tab_loader_samples_whole_documents_deterministically(tmp_path, monkeypatch):
    path = _write_fixture(tmp_path, [_document(f"case-{index}") for index in range(4)], monkeypatch)

    first = load_tab_dev_tasks(source_path=path, max_documents=2, seed=7)
    second = load_tab_dev_tasks(source_path=path, max_documents=2, seed=7)

    assert first.selected_document_uids == second.selected_document_uids
    assert len(first.selected_document_uids) == 2
    assert {task.document_uid for task in first.tasks} == set(first.selected_document_uids)


def test_tab_loader_rejects_bad_offsets(tmp_path, monkeypatch):
    document = _document()
    mention = document["annotations"]["annotator1"]["entity_mentions"][0]
    mention["span_text"] = "not the source text"

    with pytest.raises(ValueError, match="span text mismatch"):
        load_tab_dev_tasks(source_path=_write_fixture(tmp_path, [document], monkeypatch))


def test_tab_prompt_uses_native_policy_without_openpii_examples(tmp_path, monkeypatch):
    task = load_tab_dev_tasks(
        source_path=_write_fixture(tmp_path, [_document()], monkeypatch)
    ).tasks[0]

    prompt = system_prompt("fewshot_fast", dataset_name="tab")
    assert "TAB anonymization policy" in prompt
    assert "DIRECT or QUASI" in task_prompt(task)
    assert "OpenPII annotation examples" not in prompt
    assert "exactly two assistant turns" in prompt
    assert "Example Applicant" in task_prompt(task)
    assert "word149" in task_prompt(task)
    assert "Extract every entity in the immutable source" not in prompt


def test_native_tool_schemas_do_not_leak_labels_between_datasets(tmp_path, monkeypatch):
    tab_task = load_tab_dev_tasks(source_path=_write_fixture(tmp_path, [_document()], monkeypatch)).tasks[0]
    pii_task = NerTask("pii", "Jane", ENTITY_LABELS, frozenset())
    for task in (tab_task, pii_task, tab_task):
        specs = [tool.to_spec() for tool in NerEpisode(task).tools_for_variant("fast")]
        for spec in specs:
            for definition in spec["parameters"].get("$defs", {}).values():
                if "label" in definition.get("properties", {}):
                    assert definition["properties"]["label"]["enum"] == list(task.labels)
    assert len(ENTITY_LABELS) == 19
    assert set(ENTITY_LABELS).isdisjoint(TAB_LABELS)


@pytest.mark.asyncio
async def test_tab_fast_grounding_reminds_finalize_without_gold_feedback(tmp_path, monkeypatch):
    task = load_tab_dev_tasks(source_path=_write_fixture(tmp_path, [_document()], monkeypatch)).tasks[0]
    episode = NerEpisode(task)
    ground, finalize = episode.tools_for_variant("fast")
    result = await ground.run(ToolInput(arguments={"proposals": []}, call_id="ground"))
    payload = json.loads(result.messages[0]["content"])
    assert "call finalize now" in payload["next_action"]
    assert "gold" not in payload
    final = await finalize.run(ToolInput(arguments={"entities": []}, call_id="final"))
    assert final.should_stop


def test_mixture_is_deterministic_and_rejects_dev(tmp_path, monkeypatch):
    path = _write_fixture(tmp_path, [_document()], monkeypatch)
    train = load_tab_tasks(split="train", source_path=path)
    pii = [NerTask(str(index), "Jane", ENTITY_LABELS, frozenset()) for index in range(30)]
    mixed = mix_tab_training_tasks(pii, train.tasks, fraction=0.25, seed=42)
    assert len(mixed) == 40
    assert sum(task.dataset_name == "tab" for task in mixed) == 10
    assert mixed == mix_tab_training_tasks(pii, train.tasks, fraction=0.25, seed=42)
    with pytest.raises(ValueError, match="Only TAB train"):
        mix_tab_training_tasks(pii, load_tab_dev_tasks(source_path=path).tasks, fraction=0.25, seed=42)


def test_unsupported_training_document_is_recorded_not_silently_truncated(tmp_path, monkeypatch):
    bad = _document("unsupported")
    mention = bad["annotations"]["annotator1"]["entity_mentions"][0]
    mention.update(start_offset=0, end_offset=len(bad["text"]), span_text=bad["text"])
    path = _write_fixture(tmp_path, [_document(), bad], monkeypatch)
    train = load_tab_tasks(split="train", source_path=path)
    assert train.selected_document_uids == ("case-1",)
    assert train.excluded_documents[0][0] == "unsupported"
    with pytest.raises(ValueError, match="not fully contained"):
        load_tab_dev_tasks(source_path=path)


@pytest.mark.parametrize("split", ["train", "dev"])
def test_local_tab_source_requires_pinned_checksum(tmp_path, monkeypatch, split):
    path = _write_fixture(tmp_path, [_document()], monkeypatch)
    assert load_tab_tasks(split=split, source_path=path).tasks
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match=f"TAB {split} checksum mismatch"):
        load_tab_tasks(split=split, source_path=path)


@pytest.mark.asyncio
async def test_invalid_local_tab_source_does_not_publish_pinned_provenance(tmp_path):
    from interactive_training.recipes.tool_ner_rl.eval_azure import CLIConfig, cli_main

    source = tmp_path / "custom.json"
    source.write_text(json.dumps([_document()]), encoding="utf-8")
    output = tmp_path / "evaluation"
    config = CLIConfig(
        project_endpoint="https://example.invalid",
        dataset_name="tab",
        dataset_split="dev",
        tab_source_path=str(source),
        log_path=str(output),
    )
    with pytest.raises(ValueError, match="TAB dev checksum mismatch"):
        await cli_main(config)
    assert not (output / "run_meta.json").exists()
    assert not (output / "manifest.json").exists()