"""Unit tests for rollout failure tolerance (timeout → retry → circuit breaker).

Covers:
- Retryable slot failure is retried into a fresh env, preserving group size.
- Every env (original + replacement) is closed exactly once.
- The rolling failure tracker aborts on a sustained failure rate.
- The default (no policy) path preserves the original exception.
- A non-retryable error is never masked by the retry policy.
- Exhausting the retry budget raises RolloutGroupExhausted.
"""

import asyncio

import pytest
from azure.ai.finetuningsessions.models import ImageChunk, ModelInput, ModelInputChunk

from interactive_training.completers import (
    ContextLengthExceededError,
    TokenCompleter,
    TokensWithLogprobs,
)
from interactive_training.rl.metric_util import compute_trajectory_metrics
from interactive_training.rl.rollouts import (
    RolloutFailureRateExceeded,
    RolloutFailureTracker,
    RolloutGroupExhausted,
    RolloutRetryPolicy,
    _is_retryable_rollout_error,
    do_group_rollout,
)
from interactive_training.rl.types import Env, EnvGroupBuilder, StepResult


class _Policy(TokenCompleter):
    async def __call__(self, model_input, stop, max_tokens=None):
        return TokensWithLogprobs(tokens=[1], maybe_logprobs=[-0.1])


class _Env(Env):
    def __init__(self, slot: int, *, fail: bool = False, error: Exception | None = None):
        self.slot = slot
        self.fail = fail
        self.error = error
        self.closed = 0
        self.observation = ModelInput(chunks=[ModelInputChunk(tokens=[slot])])

    async def initial_observation(self):
        if self.error is not None:
            raise self.error
        if self.fail:
            raise TimeoutError("sample stalled")
        return self.observation, []

    async def step(self, action):
        return StepResult(
            reward=float(self.slot),
            episode_done=True,
            next_observation=self.observation,
            next_stop_condition=[],
        )

    async def close(self):
        self.closed += 1


class _Builder(EnvGroupBuilder):
    """First group has a failing slot; the replacement group succeeds."""

    def __init__(self):
        self.batches: list[list[_Env]] = []

    async def make_envs(self):
        if not self.batches:
            envs = [_Env(0), _Env(1, fail=True)]
        else:
            envs = [_Env(0), _Env(1)]
        self.batches.append(envs)
        return envs

    async def compute_group_rewards(self, trajectory_group, env_group):
        return [(float(env.slot), {}) for env in env_group]


def test_group_retry_preserves_full_size_and_cleans_up():
    builder = _Builder()
    group = asyncio.run(
        do_group_rollout(
            builder,
            _Policy(),
            retry_policy=RolloutRetryPolicy(
                max_retries_per_trajectory=1,
                max_extra_attempts_per_group=3,
            ),
        )
    )

    # Full group size preserved despite one slot failing on the first pass.
    assert len(group.trajectories_G) == 2
    assert group.final_rewards_G == [0.0, 1.0]
    assert group.metrics_G[0]["rollout/retry_attempts"] == 1
    # Two make_envs calls: original group + one replacement wave.
    assert len(builder.batches) == 2
    # No env leaks: every env created (original + replacement) is closed. A
    # failed slot is closed promptly on failure AND by the group's finally
    # safety net (teardown is idempotent), so the invariant is "closed >= 1".
    assert all(env.closed >= 1 for batch in builder.batches for env in batch)


def test_group_rollout_logs_multimodal_observation_lengths():
    class _MultimodalBuilder(EnvGroupBuilder):
        async def make_envs(self):
            env = _Env(0)
            env.observation = ModelInput(
                chunks=[
                    ModelInputChunk(tokens=[1, 2]),
                    ImageChunk(data=b"\xff\xd8\xffjpeg", format="jpeg", expected_tokens=7),
                ]
            )
            return [env]

        async def compute_group_rewards(self, trajectory_group, env_group):
            return [(1.0, {})]

    group = asyncio.run(do_group_rollout(_MultimodalBuilder(), _Policy()))

    assert len(group.trajectories_G) == 1
    assert group.final_rewards_G == [1.0]
    metrics = compute_trajectory_metrics([group], [["multimodal"]])
    assert metrics["env/all/total_ob_tokens"] == 9


def test_group_rollout_maps_final_results_before_env_cleanup():
    builder = _Builder()

    def map_results(group, envs):
        return [
            (
                env.slot,
                env.closed,
                trajectory.transitions[-1].ac.tokens,
            )
            for trajectory, env in zip(group.trajectories_G, envs, strict=True)
        ]

    group, mapped_results = asyncio.run(
        do_group_rollout(
            builder,
            _Policy(),
            retry_policy=RolloutRetryPolicy(
                max_retries_per_trajectory=1,
                max_extra_attempts_per_group=3,
            ),
            result_mapper=map_results,
        )
    )

    assert len(group.trajectories_G) == 2
    assert mapped_results == [(0, 0, [1]), (1, 0, [1])]
    assert all(env.closed >= 1 for batch in builder.batches for env in batch)


def test_failure_tracker_aborts_on_sustained_failures():
    tracker = RolloutFailureTracker(
        window_size=10,
        min_attempts=10,
        warn_rate=0.10,
        abort_rate=0.20,
    )

    with pytest.raises(RolloutFailureRateExceeded):
        tracker.record(successes=7, failures=3)


def test_failure_tracker_stays_quiet_below_min_attempts():
    tracker = RolloutFailureTracker(window_size=100, min_attempts=25, abort_rate=0.20)
    # 100% failure rate but only 5 attempts observed → below min_attempts, no abort.
    tracker.record(successes=0, failures=5)


def test_breaker_recovery_consumes_budget_then_gives_up():
    # max_resets=2: two trips recover, the third gives up (returns None → abort).
    tracker = RolloutFailureTracker(
        window_size=10, min_attempts=10, abort_rate=0.20,
        max_resets=2, reset_backoff_sec=0.0,
    )
    b1 = tracker.note_breaker_tripped()
    b2 = tracker.note_breaker_tripped()
    b3 = tracker.note_breaker_tripped()
    assert b1 == 0.0 and b2 == 0.0
    assert b3 is None  # budget exhausted
    assert tracker.resets_used == 2


def test_breaker_reset_clears_window():
    tracker = RolloutFailureTracker(
        window_size=10, min_attempts=10, abort_rate=0.20,
        max_resets=1, reset_backoff_sec=0.0,
    )
    # Fill the window to the abort threshold, trip, recover, then a fresh window
    # of successes must not immediately re-abort.
    with pytest.raises(RolloutFailureRateExceeded):
        tracker.record(successes=7, failures=3)
    assert tracker.note_breaker_tripped() == 0.0
    # Window cleared → recording all successes stays well under the threshold.
    tracker.record(successes=10, failures=0)  # no raise


def test_breaker_concurrent_trips_share_one_budget():
    # Two near-simultaneous trips within the backoff window count as one outage.
    tracker = RolloutFailureTracker(
        window_size=10, min_attempts=10, abort_rate=0.20,
        max_resets=1, reset_backoff_sec=1000.0,
    )
    assert tracker.note_breaker_tripped() == 1000.0  # consumes the only budget
    # Second trip within the backoff window is folded in, does NOT abort.
    assert tracker.note_breaker_tripped() == 1000.0
    assert tracker.resets_used == 1


def test_default_group_rollout_preserves_original_exception():
    builder = _Builder()

    # No retry policy → the first failure surfaces unchanged (no masking).
    with pytest.raises(TimeoutError, match="sample stalled"):
        asyncio.run(do_group_rollout(builder, _Policy()))


def test_non_retryable_error_is_not_hidden_by_policy():
    class _BadBuilder(_Builder):
        async def make_envs(self):
            envs = [_Env(0), _Env(1, error=ValueError("bad rollout"))]
            self.batches.append(envs)
            return envs

    builder = _BadBuilder()

    with pytest.raises(ValueError, match="bad rollout"):
        asyncio.run(
            do_group_rollout(
                builder,
                _Policy(),
                retry_policy=RolloutRetryPolicy(
                    max_retries_per_trajectory=1,
                    max_extra_attempts_per_group=3,
                ),
            )
        )


def test_exhausted_retry_budget_raises_group_exhausted():
    class _AlwaysFailBuilder(EnvGroupBuilder):
        def __init__(self):
            self.batches: list[list[_Env]] = []

        async def make_envs(self):
            envs = [_Env(0), _Env(1, fail=True)]
            self.batches.append(envs)
            return envs

        async def compute_group_rewards(self, trajectory_group, env_group):
            return [(float(env.slot), {}) for env in env_group]

    builder = _AlwaysFailBuilder()

    with pytest.raises(RolloutGroupExhausted):
        asyncio.run(
            do_group_rollout(
                builder,
                _Policy(),
                retry_policy=RolloutRetryPolicy(
                    max_retries_per_trajectory=1,
                    max_extra_attempts_per_group=1,
                ),
            )
        )


def test_retry_policy_rejects_negative_budgets():
    with pytest.raises(ValueError):
        RolloutRetryPolicy(max_retries_per_trajectory=-1)
    with pytest.raises(ValueError):
        RolloutRetryPolicy(max_extra_attempts_per_group=-1)


def test_failure_tracker_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        RolloutFailureTracker(window_size=0)
    with pytest.raises(ValueError):
        RolloutFailureTracker(warn_rate=0.3, abort_rate=0.2)
    with pytest.raises(ValueError):
        RolloutFailureTracker(min_attempts=0)


# --- SDK RuntimeError-wrapped sample failures (e.g. transient 403) ----------


def test_sdk_runtimeerror_sample_failure_is_retryable():
    # The SDK surfaces some backend failures as a plain RuntimeError with the
    # HTTP status in the message (not an HttpResponseError). These should be
    # retryable so they flow through the retry -> circuit-breaker path.
    err = RuntimeError(
        "sample request 06a708fc-6e11-796c failed [unknown]: "
        "Client error '403 Forbidden' for url 'https://.../v1/completions'"
    )
    assert _is_retryable_rollout_error(err) is True


def test_sdk_runtimeerror_5xx_sample_failure_is_retryable():
    err = RuntimeError("sample request abc failed [unknown]: '503 Service Unavailable'")
    assert _is_retryable_rollout_error(err) is True


def test_context_length_exceeded_is_not_retryable():
    # ContextLengthExceededError is a RuntimeError subclass with its own
    # graceful trajectory-end handling; it must NOT be treated as retryable.
    assert _is_retryable_rollout_error(ContextLengthExceededError("no room")) is False


def test_unrelated_runtimeerror_is_not_retryable():
    # A generic RuntimeError that isn't a wrapped sample-request failure stays
    # non-retryable (we don't want to mask real bugs).
    assert _is_retryable_rollout_error(RuntimeError("something else broke")) is False


# --- env_needed_after_own_trajectory() eager-close behavior -----------------
#
# Builders with env_needed_after_own_trajectory() == False release a rollout's
# environment as soon as its trajectory finishes. Builders with the default
# True value keep the historical behavior of closing only after the whole group
# finishes, since some compute_group_rewards implementations may need every env
# alive to read final env-side state.


class _SlotDoneEvent(_Env):
    """An _Env whose step() blocks on an asyncio.Event before completing,
    so tests can control exactly when "the rest of the group" is still in
    flight relative to a faster slot's own completion."""

    def __init__(self, slot: int, gate: "asyncio.Event | None" = None):
        super().__init__(slot)
        self._gate = gate

    async def step(self, action):
        if self._gate is not None:
            await self._gate.wait()
        return await super().step(action)


class _GateBuilder(EnvGroupBuilder):
    def __init__(self, envs: list[Env], *, needed: bool):
        self._envs = envs
        self._needed = needed

    async def make_envs(self):
        return self._envs

    def env_needed_after_own_trajectory(self) -> bool:
        return self._needed

    async def compute_group_rewards(self, trajectory_group, env_group):
        return [(float(env.slot), {}) for env in env_group]


def test_eager_close_releases_fast_slot_before_group_finishes():
    gate = asyncio.Event()
    fast_env = _SlotDoneEvent(0)  # no gate: finishes immediately
    slow_env = _SlotDoneEvent(1, gate=gate)  # blocked until we set the gate
    builder = _GateBuilder([fast_env, slow_env], needed=False)

    async def run():
        task = asyncio.create_task(do_group_rollout(builder, _Policy()))
        # Give the fast slot a chance to finish and close while the slow slot
        # is still blocked mid-step -- i.e. the group has not finished yet.
        await asyncio.sleep(0.05)
        assert not task.done(), "group finished before the slow slot was released"
        assert fast_env.closed >= 1, (
            "fast slot's env should already be released back to the pool "
            "even though the group is still in flight"
        )
        assert slow_env.closed == 0
        gate.set()
        group = await task
        # do_group_rollout's group-level finally still closes every env
        # (idempotent for environments that support repeated cleanup), so the
        # fast slot may be closed twice here -- what matters is it was already
        # closed *before* the group finished, asserted above.
        assert slow_env.closed >= 1
        assert group.final_rewards_G == [0.0, 1.0]

    asyncio.run(run())


def test_group_needed_keeps_fast_slot_open_until_group_finishes():
    gate = asyncio.Event()
    fast_env = _SlotDoneEvent(0)
    slow_env = _SlotDoneEvent(1, gate=gate)
    builder = _GateBuilder([fast_env, slow_env], needed=True)

    async def run():
        task = asyncio.create_task(do_group_rollout(builder, _Policy()))
        await asyncio.sleep(0.05)
        assert not task.done()
        assert fast_env.closed == 0, (
            "a builder that still needs every env for compute_group_rewards "
            "must not have its envs closed early"
        )
        gate.set()
        group = await task
        assert fast_env.closed == 1
        assert slow_env.closed == 1
        assert group.final_rewards_G == [0.0, 1.0]

    asyncio.run(run())
