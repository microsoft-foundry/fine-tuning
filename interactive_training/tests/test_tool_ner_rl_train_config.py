import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from interactive_training.recipes.tool_ner_rl.train_azure import (
    CLIConfig,
    _parse_checkpoint_path,
    _read_manifest_summary,
)


def _manifest(threshold=0.95, sources=2):
    return {
        "config": {"f1_threshold": threshold},
        "summary": {
            "eligible_documents": 949,
            "eligible_gold_entities": 7805,
        },
        "sources": [{"path": str(index)} for index in range(sources)],
        "eligible": [{"uid": "one"}],
    }


def test_initial_training_defaults_are_bounded_and_use_pods():
    config = CLIConfig(
        project_endpoint="https://example.invalid/projects/test",
        train_filter_manifest_path="/tmp/manifest.json",
    )

    assert config.group_size == 8
    assert config.groups_per_batch == 16
    assert config.pods_keep == 4
    assert config.max_oversample_rounds == 4
    assert config.oversample_cushion == pytest.approx(1.2)
    assert config.max_turns == 2
    assert config.max_tokens == 768
    assert config.second_turn_max_tokens == 768
    assert config.max_test_examples == 200
    assert config.max_steps == 20
    assert config.max_wall_clock_seconds == 14_400.0
    assert config.expected_filter_threshold == 0.96
    assert config.load_checkpoint_path is None


def test_training_accepts_bounded_batch_sweep_variants():
    kwargs = {
        "project_endpoint": "https://example.invalid/projects/test",
        "train_filter_manifest_path": "/tmp/manifest.json",
    }
    smaller_groups = CLIConfig(
        **kwargs,
        group_size=4,
        groups_per_batch=32,
        pods_keep=2,
    )
    smaller_batch = CLIConfig(**kwargs, groups_per_batch=8)
    larger_batch = CLIConfig(**kwargs, groups_per_batch=32)
    higher_learning_rates = [
        CLIConfig(**kwargs, learning_rate=learning_rate)
        for learning_rate in (3e-5, 4e-5)
    ]

    assert smaller_groups.group_size * smaller_groups.groups_per_batch == 128
    assert smaller_groups.pods_keep * smaller_groups.groups_per_batch == 64
    assert smaller_batch.groups_per_batch == 8
    assert larger_batch.group_size * larger_batch.groups_per_batch == 256
    assert larger_batch.pods_keep * larger_batch.groups_per_batch == 128
    assert [config.learning_rate for config in higher_learning_rates] == [3e-5, 4e-5]


def test_training_rejects_out_of_bounds_scope_expansion():
    kwargs = {
        "project_endpoint": "https://example.invalid/projects/test",
        "train_filter_manifest_path": "/tmp/manifest.json",
    }
    with pytest.raises(ValueError, match="groups_per_batch"):
        CLIConfig(**kwargs, groups_per_batch=65)
    with pytest.raises(ValueError, match="pods_keep"):
        CLIConfig(**kwargs, group_size=4, pods_keep=5)
    with pytest.raises(ValueError, match="max_steps"):
        CLIConfig(**kwargs, max_steps=301)
    with pytest.raises(ValueError, match="max_wall_clock_seconds"):
        CLIConfig(**kwargs, max_wall_clock_seconds=72_001)
    with pytest.raises(ValueError, match="max_test_examples"):
        CLIConfig(**kwargs, max_test_examples=501)


def test_manifest_validation_requires_threshold_and_replicates(tmp_path):
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(_manifest()))
    assert _read_manifest_summary(valid, 0.95)["eligible_documents"] == 949

    wrong_threshold = tmp_path / "threshold.json"
    wrong_threshold.write_text(json.dumps(_manifest(threshold=0.9)))
    with pytest.raises(ValueError, match="f1_threshold=0.95"):
        _read_manifest_summary(wrong_threshold, 0.95)

    one_source = tmp_path / "source.json"
    one_source.write_text(json.dumps(_manifest(sources=1)))
    with pytest.raises(ValueError, match="at least two"):
        _read_manifest_summary(one_source, 0.95)


def test_training_rejects_alternate_research_workflows():
    kwargs = {
        "project_endpoint": "https://example.invalid/projects/test",
        "train_filter_manifest_path": "/tmp/manifest.json",
    }
    with pytest.raises(ValueError, match="two-turn"):
        CLIConfig(**kwargs, tool_variant="grounded")
    with pytest.raises(ValueError, match="two-turn"):
        CLIConfig(**kwargs, prompt_version="fewshot")


def test_training_checkpoint_path_parsing():
    assert _parse_checkpoint_path("model_abc/final") == (
        "model_abc",
        "final",
    )
    assert _parse_checkpoint_path("abc/210") == ("model_abc", "210")


def test_tab_mixture_config_preserves_openpii_defaults_and_requires_native_protocol():
    kwargs = dict(project_endpoint="https://example.invalid/projects/test", train_filter_manifest_path="/tmp/manifest.json")
    assert CLIConfig(**kwargs).tab_fraction == 0.25
    assert CLIConfig(**kwargs, tab_fraction=0.25).tab_eval_documents == 8
    with pytest.raises(ValueError, match="tab_fraction"):
        CLIConfig(**kwargs, tab_fraction=1)
    with pytest.raises(ValueError, match="two-turn"):
        CLIConfig(**kwargs, tab_fraction=0.25, max_turns=3)


def test_bundled_selection_is_uid_only_and_matches_frozen_provenance():
    from pathlib import Path

    config = CLIConfig(project_endpoint="https://example.invalid")
    path = Path(config.train_filter_manifest_path)
    manifest = json.loads(path.read_text())
    assert len(manifest["eligible"]) == 6541
    assert len({row["uid"] for row in manifest["eligible"]}) == 6541
    assert all(set(row) == {"uid"} for row in manifest["eligible"])
    assert manifest["source_manifest_sha256"] == "9ea99e1905f8445888ef9a0fb3a0683bc835e5a451179edbb0da5d662f4dab6a"
    assert config.train_source_pool_size == 10000
    assert config.training_type is None
    assert _read_manifest_summary(path, 0.96)["eligible_documents"] == 6541


@pytest.fixture
def mocked_training(monkeypatch):
    from interactive_training.recipes.tool_ner_rl import train_azure as recipe

    client = AsyncMock()
    client.create_session.return_value = "session_test"
    cleanup = AsyncMock()
    cleanup.__aenter__.return_value = cleanup
    factory = MagicMock(side_effect=[client, cleanup])
    monkeypatch.setattr("azure.ai.finetuningsessions.aio.FineTuningSessionClient", factory)
    monkeypatch.setenv("AZURE_AI_API_KEY", "test-key")
    main = AsyncMock()
    monkeypatch.setattr(recipe.rl_azure, "main", main)
    return SimpleNamespace(recipe=recipe, client=client, cleanup=cleanup, factory=factory, main=main)


@pytest.mark.asyncio
@pytest.mark.parametrize("off_policy", [None, 1])
@pytest.mark.parametrize("pods_keep", [2, 4])
async def test_training_modes_and_fresh_checkpoint_forwarding(tmp_path, mocked_training, off_policy, pods_keep):
    mocked = mocked_training
    config = CLIConfig(
        project_endpoint="https://example.invalid",
        log_path=str(tmp_path / "run"),
        max_steps_off_policy=off_policy,
        pods_keep=pods_keep,
        load_checkpoint_path="model_prior/15",
    )
    await mocked.recipe.cli_main(config)
    training = mocked.main.await_args.args[0]
    restored = mocked.client.create_session.await_args.kwargs["from_checkpoint"]
    assert restored.source_session_id == "model_prior"
    assert restored.checkpoint_id == "15"
    assert training.remove_constant_reward_groups
    assert not training.refill_on_drop
    assert training.strategy.keep == pods_keep
    if off_policy is None:
        assert training.dynamic_sampling
        assert training.async_config is None
        assert training.max_concurrent_groups == config.max_concurrent_groups
    else:
        assert not training.dynamic_sampling
        assert training.async_config.max_steps_off_policy == off_policy
        assert training.async_config.groups_per_batch == config.groups_per_batch
        assert training.max_concurrent_groups is None
    metadata = json.loads((tmp_path / "run/run_meta.json").read_text())
    assert metadata["batching"]["strategy"] == "pods"
    assert metadata["batching"]["pods_keep"] == pods_keep
    assert metadata["batching"]["dynamic_sampling"] == (off_policy is None)
    assert metadata["batching"]["max_steps_off_policy"] == off_policy
    mocked.client.close.assert_awaited_once()
    assert mocked.factory.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("initialization failed"), asyncio.CancelledError()])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_failure_closes_remote_session_and_preserves_error(tmp_path, mocked_training, failure, cleanup_fails):
    mocked = mocked_training
    mocked.main.side_effect = failure
    if cleanup_fails:
        mocked.cleanup.close_session.side_effect = RuntimeError("cleanup failed")
    config = CLIConfig(project_endpoint="https://example.invalid", log_path=str(tmp_path / "run"))
    with pytest.raises(type(failure)) as caught:
        await mocked.recipe.cli_main(config)
    assert caught.value is failure
    mocked.cleanup.close_session.assert_awaited_once_with("session_test")
    mocked.cleanup.__aexit__.assert_awaited_once()
    mocked.client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_failure_does_not_attempt_unknown_session_cleanup(tmp_path, mocked_training):
    mocked = mocked_training
    mocked.client.create_session.side_effect = RuntimeError("create failed")
    with pytest.raises(RuntimeError, match="create failed"):
        await mocked.recipe.cli_main(CLIConfig(
            project_endpoint="https://example.invalid", log_path=str(tmp_path / "run"),
        ))
    assert mocked.factory.call_count == 1
    mocked.client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_is_rejected_before_overwriting_artifacts(tmp_path, monkeypatch, mocked_training):
    with pytest.raises(ValueError, match="In-place resume"):
        CLIConfig(project_endpoint="https://example.invalid", behavior_if_log_dir_exists="resume")
    ledger = tmp_path / "checkpoints.jsonl"
    original = '{"batch":15,"state_path":"model_prior/15"}\n'
    ledger.write_text(original)
    monkeypatch.setattr("builtins.input", lambda _: "resume")
    with pytest.raises(ValueError, match="fresh log_path"):
        await mocked_training.recipe.cli_main(CLIConfig(
            project_endpoint="https://example.invalid", log_path=str(tmp_path),
        ))
    assert ledger.read_text() == original
    assert not (tmp_path / "run_meta.json").exists()
    mocked_training.factory.assert_not_called()


@pytest.mark.parametrize("off_policy", [0, -1])
def test_async_rejects_invalid_staleness(off_policy):
    with pytest.raises(ValueError, match="max_steps_off_policy"):
        CLIConfig(project_endpoint="https://example.invalid", max_steps_off_policy=off_policy)


@pytest.mark.asyncio
async def test_repeated_tab_evaluation_preserves_both_artifacts(tmp_path, monkeypatch):
    from interactive_training.recipes.tool_ner_rl.train_azure import TabTrainingEvaluator

    evaluator = object.__new__(TabTrainingEvaluator)
    evaluator.output_root = tmp_path
    evaluator.reference_tasks = ()
    evaluator.eval_config = SimpleNamespace(max_tokens=768, second_turn_max_tokens=768, max_concurrent_samples=4)
    monkeypatch.setattr(evaluator, "eval_token_completer", AsyncMock(return_value={"score": 1.0}))
    await evaluator(None, step=2)
    await evaluator(None, step=2)
    outputs = list(tmp_path.glob("step-000002*/metrics.json"))
    assert len(outputs) == 2
    assert all(json.loads(path.read_text()) == {"score": 1.0} for path in outputs)


@pytest.mark.parametrize("pods_keep", [2, 4])
def test_recipe_async_config_runs_shared_loop_with_pods(tmp_path, mocked_training, pods_keep):
    from tests.test_dynamic_sampling import ControlledDataset, FixedRewardGroupBuilder, MixedRewardGroupBuilder, _make_mock_session
    from tests.test_train_loop_characterization import _fingerprint, _run_async

    asyncio.run(mocked_training.recipe.cli_main(CLIConfig(
        project_endpoint="https://example.invalid",
        log_path=str(tmp_path / "run"),
        max_steps_off_policy=1,
        groups_per_batch=2,
        pods_keep=pods_keep,
        eval_every=0,
        save_every=0,
    )))
    training = mocked_training.main.await_args.args[0]
    dataset = ControlledDataset([
        [FixedRewardGroupBuilder(0.0, group_size=8), MixedRewardGroupBuilder([0.0, 1.0] * 4)],
        [MixedRewardGroupBuilder([1.0, 0.0] * 4)],
    ])
    session = _make_mock_session()
    _run_async(training, dataset, session, tmp_path / "loop")
    fingerprint = _fingerprint(session)
    assert fingerprint["n_optim"] == 2
    assert fingerprint["n_datum"] == 2 * 2 * pods_keep