from collections import defaultdict

import pytest
from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

from interactive_training.completers import TokenCompleter, TokensWithLogprobs
from interactive_training.rl.metric_util import RLTestSetEvaluator, ValidationRecord, ValidationSkipped
from interactive_training.rl.rollouts import AuthoritativeGroupGradingError
from interactive_training.rl.types import Env, EnvGroupBuilder, RLDataset, StepResult


class _Env(Env):
    def __init__(self, value: int):
        self.value = value
        self.closed = False
        self.observation = ModelInput(chunks=[ModelInputChunk(tokens=[value])])

    async def initial_observation(self):
        return self.observation, []

    async def step(self, action):
        return StepResult(
            reward=float(self.value),
            episode_done=True,
            next_observation=self.observation,
            next_stop_condition=[],
        )

    async def close(self):
        self.closed = True


class _ZeroRewardEnv(_Env):
    async def step(self, action):
        result = await super().step(action)
        return StepResult(
            reward=0.0,
            episode_done=result.episode_done,
            next_observation=result.next_observation,
            next_stop_condition=result.next_stop_condition,
        )


class _Builder(EnvGroupBuilder):
    def __init__(self, env_groups, tag):
        self.env_groups = list(env_groups)
        self.tag = tag
        self.created_envs = []
        self.reward_calls = 0

    async def make_envs(self):
        envs = self.env_groups.pop(0)
        self.created_envs.extend(envs)
        return envs

    async def compute_group_rewards(self, trajectory_group, env_group):
        self.reward_calls += 1
        return [(float(env.value * 10), {}) for env in env_group]

    def logging_tags(self):
        return [self.tag]


class _AuthoritativeGradingFailureBuilder(_Builder):
    async def compute_group_rewards(self, trajectory_group, env_group):
        raise AuthoritativeGroupGradingError("Foundry validation grading failed")


class _Dataset(RLDataset):
    def __init__(self, builders):
        self.builders = builders

    def __len__(self):
        return len(self.builders)

    def get_batch(self, index):
        return [self.builders[index]]


class _BatchedDataset(RLDataset):
    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def get_batch(self, index):
        return self.batches[index]


class _Policy(TokenCompleter):
    def __init__(self, failures=None):
        self.failures = defaultdict(int, failures or {})
        self.calls = []

    async def __call__(self, model_input, stop, max_tokens=None):
        value = model_input.chunks[0].tokens[0]
        self.calls.append(value)
        if self.failures[value] > 0:
            self.failures[value] -= 1
            raise TimeoutError(f"sample {value} timed out")
        return TokensWithLogprobs(tokens=[value + 100], maybe_logprobs=[-0.1])


def _record_factory(step, trajectory, env, tags):
    assert env.closed is False
    return ValidationRecord(
        step=step,
        input=f"input-{env.value}",
        response=f"response-{trajectory.transitions[-1].ac.tokens[0]}",
        ground_truth=f"truth-{env.value}",
        tags=tags,
    )


class _Observer:
    name = "test"
    max_tokens = None

    def __init__(self, callback):
        self.callback = callback

    def create_record(self, step, trajectory, env, tags):
        return _record_factory(step, trajectory, env, tags)

    def on_validation_complete(self, records):
        return self.callback(records)


@pytest.mark.parametrize("require_full_validation", [False, True])
async def test_validation_records_are_aggregated_once_in_dataset_order(
    require_full_validation,
):
    first = _Builder([[_Env(1), _Env(2)]], "first")
    second = _Builder([[_Env(3)]], "second")
    emitted = []
    policy = _Policy()
    evaluator = RLTestSetEvaluator(
        _BatchedDataset([[first, second]]),
        max_tokens=8,
        observer=_Observer(emitted.append),
        require_full_validation=require_full_validation,
    )

    metrics = await evaluator.eval_token_completer(policy, step=7)

    assert emitted == [
        [
            ValidationRecord(7, "input-1", "response-101", "truth-1", ("first",)),
            ValidationRecord(7, "input-2", "response-102", "truth-2", ("first",)),
            ValidationRecord(7, "input-3", "response-103", "truth-3", ("second",)),
        ]
    ]
    assert metrics["test/rows_emitted"] == 3.0
    assert policy.calls == [1, 2, 3]
    assert first.reward_calls == second.reward_calls == 1
    assert all(env.closed for builder in (first, second) for env in builder.created_envs)


@pytest.mark.parametrize("require_full_validation", [False, True])
async def test_validation_callback_rewards_are_applied_before_metrics(
    require_full_validation,
):
    async def score(records):
        assert [record.input for record in records] == ["input-1", "input-2", "input-3"]
        return [
            (0.25, {"foundry_reward/attempt": 1}),
            (0.75, {"foundry_reward/attempt": 1}),
            (1.0, {"foundry_reward/attempt": 1}),
        ]

    evaluator = RLTestSetEvaluator(
        _Dataset(
            [
                _Builder([[_ZeroRewardEnv(1), _ZeroRewardEnv(2)]], "first"),
                _Builder([[_ZeroRewardEnv(3)]], "second"),
            ]
        ),
        max_tokens=8,
        observer=_Observer(score),
        require_full_validation=require_full_validation,
    )

    metrics = await evaluator.eval_token_completer(_Policy(), step=7)

    assert metrics["test/env/all/reward/total"] == pytest.approx(2.0 / 3)
    assert metrics["test/env/first/reward/total"] == 0.5
    assert metrics["test/env/second/reward/total"] == 1.0
    assert metrics["test/env/all/foundry_reward/attempt"] == 1.0
    assert metrics["test/env/all/total_episodes"] == 3
    assert metrics["test/rows_emitted"] == 3.0


async def test_authoritative_validation_callback_failure_propagates():
    async def fail_callback(records):
        raise AuthoritativeGroupGradingError("combined validation grading failed")

    evaluator = RLTestSetEvaluator(
        _Dataset([_Builder([[_Env(1)]], "authoritative")]),
        max_tokens=8,
        observer=_Observer(fail_callback),
    )

    with pytest.raises(
        AuthoritativeGroupGradingError,
        match="combined validation grading failed",
    ):
        await evaluator.eval_token_completer(_Policy(), step=7)


@pytest.mark.parametrize("require_full_validation", [False, True])
@pytest.mark.parametrize("async_callback", [False, True])
async def test_explicit_skip_never_publishes_placeholder_or_partial_rewards(
    require_full_validation, async_callback, caplog,
):
    def skip(records):
        assert len(records) == 2
        return ValidationSkipped("application grading allowance")

    async def async_skip(records):
        return skip(records)

    builder = _Builder([[_Env(1), _ZeroRewardEnv(2)]], "sample")
    evaluator = RLTestSetEvaluator(
        _Dataset([builder]),
        max_tokens=8,
        observer=_Observer(async_skip if async_callback else skip),
        require_full_validation=require_full_validation,
    )
    metrics = await evaluator.eval_token_completer(_Policy(), step=7)
    assert metrics == {"test/validation_skipped": 1.0, "test/rows_emitted": 2.0}
    assert all(env.closed for env in builder.created_envs)
    assert "Skipping validation metrics at step=7" in caplog.text


@pytest.mark.parametrize("require_full_validation", [False, True])
async def test_validation_records_use_retry_replacement_environment_alignment(
    require_full_validation,
):
    failed_env = _Env(9)
    replacement_env = _Env(4)
    builder = _Builder([[failed_env], [replacement_env]], "retry")
    emitted = []
    evaluator = RLTestSetEvaluator(
        _Dataset([builder]),
        max_tokens=8,
        max_retries_per_trajectory=1,
        max_extra_trajectory_attempts_per_group=1,
        observer=_Observer(emitted.append),
        require_full_validation=require_full_validation,
    )

    metrics = await evaluator.eval_token_completer(_Policy({9: 1}), step=8)

    assert metrics["test/rows_emitted"] == 1.0
    assert emitted[0][0].input == "input-4"
    assert emitted[0][0].response == "response-104"
    assert failed_env.closed is True
    assert replacement_env.closed is True


async def test_validation_publication_excludes_failed_groups_and_continues():
    healthy = _Builder([[_Env(1)]], "healthy")
    failed = _Builder([[_Env(9)]], "failed")
    emitted = []
    evaluator = RLTestSetEvaluator(
        _Dataset([healthy, failed]),
        max_tokens=8,
        observer=_Observer(emitted.append),
    )

    metrics = await evaluator.eval_token_completer(_Policy({9: 1}), step=9)

    assert [record.input for record in emitted[0]] == ["input-1"]
    assert metrics["test/rows_emitted"] == 1.0
    assert metrics["test/eval_groups_dropped"] == 1.0


async def test_authoritative_validation_grading_failure_propagates():
    evaluator = RLTestSetEvaluator(
        _Dataset(
            [
                _AuthoritativeGradingFailureBuilder(
                    [[_Env(1)]],
                    "authoritative",
                )
            ]
        ),
        max_tokens=8,
    )

    with pytest.raises(
        AuthoritativeGroupGradingError,
        match="Foundry validation grading failed",
    ):
        await evaluator.eval_token_completer(_Policy())


async def test_validation_publication_skips_empty_boundary():
    emitted = []
    evaluator = RLTestSetEvaluator(
        _Dataset([_Builder([[_Env(9)]], "failed")]),
        max_tokens=8,
        observer=_Observer(emitted.append),
    )

    metrics = await evaluator.eval_token_completer(_Policy({9: 1}), step=10)

    assert emitted == []
    assert metrics == {
        "test/rows_emitted": 0.0,
        "test/eval_groups_dropped": 1.0,
        "test/empty_boundary": 1.0,
    }


async def test_validation_callback_failure_is_reported_without_raising():
    async def fail_callback(records):
        raise RuntimeError("publisher unavailable")

    evaluator = RLTestSetEvaluator(
        _Dataset([_Builder([[_Env(1)]], "healthy")]),
        max_tokens=8,
        observer=_Observer(fail_callback),
    )

    metrics = await evaluator.eval_token_completer(_Policy(), step=11)

    assert metrics["test/rows_emitted"] == 1.0
    assert metrics["test/callback_failed"] == 1.0


async def test_validation_publication_requires_step_before_sampling():
    policy = _Policy()
    evaluator = RLTestSetEvaluator(
        _Dataset([_Builder([[_Env(1)]], "healthy")]),
        max_tokens=8,
        observer=_Observer(lambda records: None),
    )

    with pytest.raises(ValueError, match="requires a training step"):
        await evaluator.eval_token_completer(policy)

    assert policy.calls == []


@pytest.mark.parametrize("require_full_validation", [False, True])
async def test_metrics_only_validation_samples_each_trajectory_once(require_full_validation):
    policy = _Policy()
    evaluator = RLTestSetEvaluator(
        _Dataset([_Builder([[_Env(1), _Env(2)]], "plain")]),
        max_tokens=8,
        require_full_validation=require_full_validation,
    )

    metrics = await evaluator.eval_token_completer(policy)

    assert policy.calls == [1, 2]
    assert "test/rows_emitted" not in metrics
    assert metrics["test/env/all/reward/total"] == 16.5


@pytest.mark.parametrize("group_size", [1, 3])
@pytest.mark.parametrize("retries", [0, 1])
@pytest.mark.parametrize("with_observer", [False, True])
async def test_full_validation_rejects_dropped_group_before_publication(
    group_size, retries, with_observer
):
    healthy = _Builder([[_Env(1), _Env(2)]], "healthy")
    failed = _Builder(
        [[_Env(9 + i) for i in range(group_size)] for _ in range(retries + 1)],
        "failed",
    )
    emitted = []
    evaluator = RLTestSetEvaluator(
        _BatchedDataset([[healthy, failed]]),
        max_tokens=8,
        max_retries_per_trajectory=retries,
        max_extra_trajectory_attempts_per_group=retries,
        observer=_Observer(emitted.append) if with_observer else None,
        require_full_validation=True,
    )
    policy = _Policy({9: retries + 1})

    with pytest.raises(ValueError, match="1 of 2 validation groups were dropped"):
        await evaluator.eval_token_completer(policy, step=12)

    assert emitted == []
    assert policy.calls.count(9) == retries + 1
    assert healthy.reward_calls == 1
    assert failed.reward_calls == 0
    assert all(env.closed for builder in (healthy, failed) for env in builder.created_envs)


@pytest.mark.parametrize("with_observer", [False, True])
async def test_full_validation_rejects_all_failed_groups(with_observer):
    builders = [_Builder([[_Env(9)]], "first"), _Builder([[_Env(10)]], "second")]
    emitted = []
    evaluator = RLTestSetEvaluator(
        _Dataset(builders),
        max_tokens=8,
        observer=_Observer(emitted.append) if with_observer else None,
        require_full_validation=True,
    )

    with pytest.raises(ValueError, match="2 of 2 validation groups were dropped"):
        await evaluator.eval_token_completer(_Policy({9: 1, 10: 1}), step=13)

    assert emitted == []
    assert all(env.closed for builder in builders for env in builder.created_envs)


@pytest.mark.parametrize("batches", [[], [[], []]])
@pytest.mark.parametrize("with_observer", [False, True])
def test_full_validation_rejects_empty_dataset(batches, with_observer):
    emitted = []

    with pytest.raises(ValueError, match="non-empty validation dataset"):
        RLTestSetEvaluator(
            _BatchedDataset(batches),
            max_tokens=8,
            observer=_Observer(emitted.append) if with_observer else None,
            require_full_validation=True,
        )

    assert emitted == []


@pytest.mark.parametrize("with_observer", [False, True])
async def test_default_validation_tolerates_empty_dataset(with_observer):
    emitted = []
    policy = _Policy()
    evaluator = RLTestSetEvaluator(
        _Dataset([]),
        max_tokens=8,
        observer=_Observer(emitted.append) if with_observer else None,
    )

    metrics = await evaluator.eval_token_completer(policy, step=14)

    assert evaluator.require_full_validation is False
    assert metrics == (
        {
            "test/rows_emitted": 0.0,
            "test/eval_groups_dropped": 0.0,
            "test/empty_boundary": 1.0,
        }
        if with_observer
        else {}
    )
    assert emitted == []
    assert policy.calls == []


@pytest.mark.parametrize("async_callback", [False, True])
async def test_full_validation_propagates_callback_failure(async_callback):
    def fail_callback(records):
        raise RuntimeError("publisher unavailable")

    async def fail_async_callback(records):
        fail_callback(records)

    builder = _Builder([[_Env(1), _Env(2)]], "healthy")
    evaluator = RLTestSetEvaluator(
        _Dataset([builder]),
        max_tokens=8,
        observer=_Observer(fail_async_callback if async_callback else fail_callback),
        require_full_validation=True,
    )

    with pytest.raises(RuntimeError, match="publisher unavailable"):
        await evaluator.eval_token_completer(_Policy(), step=15)

    assert all(env.closed for env in builder.created_envs)


@pytest.mark.parametrize("require_full_validation", [False, True])
async def test_validation_callback_requires_one_reward_per_trajectory(
    require_full_validation,
):
    evaluator = RLTestSetEvaluator(
        _Dataset([_Builder([[_Env(1), _Env(2)]], "healthy")]),
        max_tokens=8,
        observer=_Observer(lambda records: [(1.0, {})]),
        require_full_validation=require_full_validation,
    )

    with pytest.raises(ValueError, match="1 rewards for 2 records"):
        await evaluator.eval_token_completer(_Policy(), step=16)