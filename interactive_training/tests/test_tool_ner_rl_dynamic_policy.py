from types import SimpleNamespace

import pytest
from azure.ai.finetuningsessions.models import ModelInput

from interactive_training.dynamic_batching.strategy import PodsStrategy
from interactive_training.recipes.tool_ner_rl.dynamic_policy import NerDynamicBatchingPolicy
from interactive_training.rl.train import Config
from interactive_training.rl.types import Trajectory, TrajectoryGroup


def _group(rewards):
    trajectories = [Trajectory(transitions=[], final_ob=ModelInput(chunks=[])) for _ in rewards]
    return TrajectoryGroup(
        trajectories_G=trajectories,
        final_rewards_G=list(rewards),
        metrics_G=[{} for _ in rewards],
    )


def _variance(values):
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def test_default_policy_builds_pods_with_native_constant_group_refill():
    policy = NerDynamicBatchingPolicy()
    overrides = policy.training_overrides()

    assert policy.group_size == 8
    assert policy.pods_keep == 4
    assert isinstance(overrides["strategy"], PodsStrategy)
    assert overrides["strategy"].keep == 4
    assert overrides["dynamic_sampling"] is True
    assert overrides["remove_constant_reward_groups"] is True
    assert overrides["max_oversample_rounds"] == 3
    assert overrides["oversample_cushion"] == pytest.approx(1.2)
    assert overrides["refill_on_drop"] is False
    assert overrides["max_rolls_per_group"] is None


def test_pods_selects_max_variance_subset_for_continuous_f1_rewards():
    rewards = [0.05, 0.20, 0.35, 0.50, 0.62, 0.78, 0.90, 1.00]
    strategy = NerDynamicBatchingPolicy().build_strategy()

    selection = strategy.select([_group(rewards)])[0]
    selected_rewards = [rewards[index] for index in selection.kept_indices]

    assert len(selection.kept_indices) == 4
    assert 0 in selection.kept_indices
    assert 7 in selection.kept_indices
    assert _variance(selected_rewards) == pytest.approx(
        max(
            _variance([rewards[a], rewards[b], rewards[c], rewards[d]])
            for a in range(8)
            for b in range(a + 1, 8)
            for c in range(b + 1, 8)
            for d in range(c + 1, 8)
        )
    )


def test_combined_pods_and_dynamic_sampling_config_is_accepted():
    policy = NerDynamicBatchingPolicy()
    dataset_builder = SimpleNamespace(group_size=policy.group_size)
    config = Config(
        learning_rate=1e-5,
        dataset_builder=dataset_builder,
        model_name="Qwen/Qwen3.6-35B-A3B",
        max_tokens=768,
        log_path="/tmp/ner-policy-test",
        **policy.training_overrides(),
    )

    assert isinstance(config.strategy, PodsStrategy)
    assert config.dynamic_sampling is True
    assert config.dataset_builder.group_size == 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_size": 1},
        {"group_size": 8, "pods_keep": 1},
        {"group_size": 4, "pods_keep": 5},
        {"max_oversample_rounds": 0},
        {"oversample_cushion": 0.9},
    ],
)
def test_policy_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        NerDynamicBatchingPolicy(**kwargs)