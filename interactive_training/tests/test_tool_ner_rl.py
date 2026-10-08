import json

import pytest

from interactive_training.recipes.tool_ner_rl.ner_env import (
    DEFAULT_DATASET_REVISION,
    FEWSHOT_TRAIN_UIDS,
    LABEL_DEFINITIONS,
    MAX_SAFE_EVAL_EXAMPLES,
    MAX_SAFE_POOL_EXAMPLES,
    load_openpii_tasks,
    openpii_row_to_task,
    stratified_sample_tasks,
    system_prompt,
    tasks_from_rows,
)
from interactive_training.recipes.tool_ner_rl.eval_azure import (
    CLIConfig,
    _parse_checkpoint_path,
    add_micro_metrics,
    prompt_seed,
)
from interactive_training.recipes.tool_ner_rl.ner_task import GoldEntity, NerEpisode, NerTask
from interactive_training.tool_use import ToolInput


def _task() -> NerTask:
    source = "Patient saw Jane Smith. Emergency contact: Jane Smith."
    return NerTask(
        uid="example-1",
        source_text=source,
        labels=("GIVENNAME",),
        gold_entities=frozenset(
            {GoldEntity(source.index("Jane Smith"), source.index("Jane Smith") + 10, "GIVENNAME")}
        ),
    )


async def _run_tool(tool, **arguments):
    result = await tool.run(ToolInput(arguments=arguments, call_id="call-1"))
    return result, json.loads(result.messages[0]["content"])


@pytest.mark.asyncio
async def test_duplicate_discovery_inspection_finalization_and_reward():
    episode = NerEpisode(_task())

    _, found = await _run_tool(episode.find_text, text="Jane Smith", normalization="exact")
    assert found["match_count"] == 2
    assert [match["start"] for match in found["matches"]] == [12, 43]

    provider_span_id = found["matches"][0]["span_id"]
    _, inspected = await _run_tool(
        episode.inspect,
        span_ids=[match["span_id"] for match in found["matches"]],
        context_chars=15,
    )
    assert "Patient saw Jane Smith" in inspected["spans"][0]["context"]

    finalized, payload = await _run_tool(
        episode.finalize,
        entities=[{"label": "GIVENNAME", "span_ids": [provider_span_id]}],
    )
    assert finalized.should_stop is True
    assert payload == {"accepted": True, "entity_count": 1, "rejected": {}}

    reward, metrics = await episode.reward([])
    assert reward == pytest.approx(1.0)
    assert metrics["exact_f1"] == pytest.approx(1.0)
    assert metrics["exact_recall"] == pytest.approx(1.0)
    assert metrics["exact_true_positives"] == 1.0
    assert metrics["predicted_entities"] == 1.0
    assert metrics["gold_entities"] == 1.0
    assert metrics["finalize_valid"] == 1.0
    assert metrics["find_text_calls"] == 1.0
    assert metrics["inspect_calls"] == 1.0
    assert metrics["ground_entities_calls"] == 0.0
    assert metrics["select_occurrence_calls"] == 0.0
    assert metrics["finalize_calls"] == 1.0


@pytest.mark.asyncio
async def test_normalized_lookup_preserves_original_offsets():
    source = "Name: John M.   Smith"
    task = NerTask(
        uid="normalized",
        source_text=source,
        labels=("NAME",),
        gold_entities=frozenset({GoldEntity(6, len(source), "NAME")}),
    )
    episode = NerEpisode(task)

    _, payload = await _run_tool(
        episode.find_text,
        text="John M Smith",
        normalization="punctuation_whitespace",
    )

    assert payload["match_count"] == 1
    assert payload["matches"][0]["text"] == "John M.   Smith"
    assert (payload["matches"][0]["start"], payload["matches"][0]["end"]) == (6, len(source))


@pytest.mark.asyncio
async def test_casefold_expansion_preserves_original_unicode_offsets():
    source = "Street: Straße"
    task = NerTask(
        uid="casefold-expansion",
        source_text=source,
        labels=("STREET",),
        gold_entities=frozenset({GoldEntity(8, 14, "STREET")}),
    )
    episode = NerEpisode(task)

    _, payload = await _run_tool(
        episode.find_text,
        text="STRASSE",
        normalization="case_insensitive",
    )

    assert payload["match_count"] == 1
    assert payload["matches"][0]["text"] == "Straße"
    assert (payload["matches"][0]["start"], payload["matches"][0]["end"]) == (8, 14)


@pytest.mark.asyncio
async def test_finalize_rejects_guessed_span_id_and_stops_with_empty_prediction():
    episode = NerEpisode(_task())

    result, payload = await _run_tool(
        episode.finalize,
        entities=[{"label": "GIVENNAME", "span_ids": ["s_guessed"]}],
    )

    assert result.should_stop is True
    assert payload["accepted"] is True
    assert payload["entity_count"] == 0
    assert payload["rejected"] == {"unknown_span_ids": ["s_guessed"]}
    reward, metrics = await episode.reward([])
    assert reward == pytest.approx(-0.1)
    assert metrics["finalize_valid"] == 0.0
    assert metrics["invalid_finalizations"] == 1.0


@pytest.mark.asyncio
async def test_empty_grounding_can_be_finalized_as_an_honest_empty_prediction():
    episode = NerEpisode(_task())

    _, grounded = await _run_tool(episode.ground_entities, proposals=[])
    finalized, payload = await _run_tool(episode.finalize, entities=[])

    assert grounded == {"results": []}
    assert finalized.should_stop is True
    assert payload == {"accepted": True, "entity_count": 0, "rejected": {}}
    reward, metrics = await episode.reward([])
    assert reward == 0.0
    assert metrics["finalize_valid"] == 1.0


@pytest.mark.asyncio
async def test_episode_state_is_isolated():
    first = NerEpisode(_task())
    second = NerEpisode(_task())

    _, found = await _run_tool(first.find_text, text="Jane Smith", normalization="exact")
    span_id = found["matches"][0]["span_id"]
    _, second_payload = await _run_tool(second.inspect, span_ids=[span_id], context_chars=10)

    assert "error" in second_payload
    assert second.candidate_spans_returned == 0


def test_openpii_row_normalization_accepts_json_mask():
    row = {
        "uid": "row-1",
        "language": "en",
        "source_text": "Call Jane at 555-0100.",
        "privacy_mask": json.dumps(
            [
                {"value": "Jane", "start": 5, "end": 9, "label": "GIVENNAME"},
                {"value": "555-0100", "start": 13, "end": 21, "label": "TELEPHONENUM"},
            ]
        ),
    }

    task = openpii_row_to_task(row)

    assert task is not None
    assert task.uid == "row-1"
    assert task.labels == tuple(LABEL_DEFINITIONS)
    assert len(task.gold_entities) == 2


def test_openpii_row_normalization_accepts_integer_uid():
    row = {
        "uid": 42,
        "language": "en",
        "source_text": "Jane",
        "privacy_mask": [{"value": "Jane", "start": 0, "end": 4, "label": "GIVENNAME"}],
    }

    task = openpii_row_to_task(row)

    assert task is not None
    assert task.uid == "42"


def test_openpii_row_rejects_offset_mismatch():
    row = {
        "uid": "bad-offset",
        "language": "en",
        "source_text": "Jane",
        "privacy_mask": [{"value": "John", "start": 0, "end": 4, "label": "GIVENNAME"}],
    }
    assert openpii_row_to_task(row) is None


def test_bounded_row_scan_filters_language_and_stops():
    valid = {
        "uid": "en-1",
        "language": "en",
        "source_text": "Jane",
        "privacy_mask": [{"value": "Jane", "start": 0, "end": 4, "label": "GIVENNAME"}],
    }
    rows = [
        {**valid, "uid": "fr-1", "language": "fr"},
        valid,
        {**valid, "uid": "en-2"},
    ]

    tasks = tasks_from_rows(rows, max_examples=2, max_documents_scanned=3)

    assert [task.uid for task in tasks] == ["en-1", "en-2"]


def test_loader_rejects_accidentally_large_eval_before_network(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("load_dataset must not be called")

    monkeypatch.setattr("interactive_training.recipes.tool_ner_rl.ner_env.load_dataset", fail_if_called)
    with pytest.raises(ValueError, match=str(MAX_SAFE_POOL_EXAMPLES)):
        load_openpii_tasks("validation", max_examples=MAX_SAFE_POOL_EXAMPLES + 1)


def test_loader_uses_pinned_parquet_and_language_pushdown(monkeypatch):
    row = {
        "uid": "en-1",
        "language": "en",
        "source_text": "Jane",
        "privacy_mask": [{"value": "Jane", "start": 0, "end": 4, "label": "GIVENNAME"}],
    }
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return iter([row])

    monkeypatch.setattr("interactive_training.recipes.tool_ner_rl.ner_env.load_dataset", fake_load_dataset)
    tasks = load_openpii_tasks("validation", max_examples=1)

    assert [task.uid for task in tasks] == ["en-1"]
    args, kwargs = calls[0]
    assert args == ("parquet",)
    assert kwargs["streaming"] is True
    assert kwargs["filters"] == [("language", "==", "en")]
    assert all(DEFAULT_DATASET_REVISION in url for url in kwargs["data_files"]["validation"])


def test_prompt_variants_enforce_grounded_terminal_submission():
    fewshot = system_prompt("fewshot_fast")
    assert "exactly two assistant turns" in fewshot
    assert "training UID 24854336" in fewshot
    assert 'GENDER="Genderfluid"; SEX="M"' in fewshot
    assert "ground_entities call containing all proposals" in fewshot


@pytest.mark.asyncio
async def test_negative_task_rewards_honest_empty_prediction():
    task = NerTask(
        uid="negative",
        source_text="Jane Smith",
        labels=("AGE",),
        gold_entities=frozenset(),
    )
    episode = NerEpisode(task)

    await _run_tool(episode.ground_entities, proposals=[])
    await _run_tool(episode.finalize, entities=[])
    reward, metrics = await episode.reward([])

    assert reward == pytest.approx(1.0)
    assert metrics["exact_f1"] == pytest.approx(1.0)


def test_eval_config_defaults_are_small_and_training_free(monkeypatch):
    monkeypatch.setenv("PROJECT_ENDPOINT", "https://example.invalid/projects/test")
    config = CLIConfig(project_endpoint="https://example.invalid/projects/test")

    assert config.model_name == "Qwen/Qwen3.8-27B"
    assert config.renderer_name is None
    assert config.lora_seed == 42
    assert config.max_examples == 8
    assert config.samples_per_task == 1
    assert config.max_turns == 2
    assert config.max_tokens == 768
    assert config.second_turn_max_tokens == 768
    assert config.max_tokens + config.second_turn_max_tokens == 1536
    assert config.prompt_version == "fewshot_fast"
    assert config.tool_variant == "fast"
    assert config.dataset_split == "validation"
    assert config.uid_filter_manifest is None
    assert config.selection_method == "stratified"
    assert config.load_checkpoint_path is None
    assert FEWSHOT_TRAIN_UIDS == {"24854336", "24790226", "24829919"}
    assert not hasattr(config, "learning_rate")
    assert not hasattr(config, "max_steps")


def test_eval_config_rejects_large_rollout_matrix():
    with pytest.raises(ValueError, match="max_examples"):
        CLIConfig(
            project_endpoint="https://example.invalid/projects/test",
            max_examples=MAX_SAFE_EVAL_EXAMPLES + 1,
        )
    with pytest.raises(ValueError, match="samples_per_task"):
        CLIConfig(
            project_endpoint="https://example.invalid/projects/test",
            samples_per_task=9,
        )

    with pytest.raises(ValueError, match="max_turns"):
        CLIConfig(
            project_endpoint="https://example.invalid/projects/test",
            max_turns=4,
        )
    with pytest.raises(ValueError, match="max_tokens"):
        CLIConfig(
            project_endpoint="https://example.invalid/projects/test",
            max_tokens=1025,
        )


def test_add_micro_metrics_uses_raw_corpus_counts():
    metrics = {
        "base/env/all/exact_true_positives": 3.75,
        "base/env/all/predicted_entities": 5.375,
        "base/env/all/gold_entities": 7.5,
    }

    add_micro_metrics(metrics)

    assert metrics["base/env/all/micro_precision"] == pytest.approx(30 / 43)
    assert metrics["base/env/all/micro_recall"] == pytest.approx(30 / 60)
    assert metrics["base/env/all/micro_f1"] == pytest.approx(60 / 103)


def test_stratified_sample_is_deterministic_and_spans_complexity_quartiles():
    tasks = [
        NerTask(
            uid=f"task-{index:03d}",
            source_text="x" * 50,
            labels=("AGE",),
            gold_entities=frozenset(
                GoldEntity(offset, offset + 1, "AGE") for offset in range(index + 1)
            ),
        )
        for index in range(40)
    ]

    first = stratified_sample_tasks(tasks, sample_size=8, seed=42)
    second = stratified_sample_tasks(tasks, sample_size=8, seed=42)

    assert [task.uid for task in first] == [task.uid for task in second]
    entity_counts = sorted(len(task.gold_entities) for task in first)
    assert sum(count <= 10 for count in entity_counts) == 2
    assert sum(11 <= count <= 20 for count in entity_counts) == 2
    assert sum(21 <= count <= 30 for count in entity_counts) == 2
    assert sum(count >= 31 for count in entity_counts) == 2


@pytest.mark.parametrize("pool_size", range(1, 17))
def test_stratified_sample_fills_small_and_uneven_pools(pool_size):
    tasks = [
        NerTask(str(index), "Example", ("AGE",), frozenset())
        for index in range(pool_size)
    ]
    for sample_size in range(1, pool_size + 1):
        sample = stratified_sample_tasks(tasks, sample_size=sample_size, seed=42)
        assert len(sample) == sample_size
        assert len({task.uid for task in sample}) == sample_size
        assert sample == stratified_sample_tasks(
            list(reversed(tasks)), sample_size=sample_size, seed=42,
        )


def test_prompt_seed_depends_only_on_prompt_and_base_seed():
    assert prompt_seed([1, 2, 3], 42) == prompt_seed([1, 2, 3], 42)
    assert prompt_seed([1, 2, 3], 42) != prompt_seed([1, 2, 4], 42)
    assert prompt_seed([1, 2, 3], 42) != prompt_seed([1, 2, 3], 43)
    assert prompt_seed([1, 2, 3], 42, 0) != prompt_seed([1, 2, 3], 42, 1)


def test_eval_checkpoint_path_parsing():
    assert _parse_checkpoint_path("model_abc/195") == (
        "model_abc",
        "195",
    )
    assert _parse_checkpoint_path("abc/final") == ("model_abc", "final")
    with pytest.raises(ValueError, match="load_checkpoint_path"):
        _parse_checkpoint_path("invalid")