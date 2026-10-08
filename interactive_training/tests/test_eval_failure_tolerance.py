"""Eval-path failure tolerance: a wedged sample must not crash the run.

Regression for the tp_full_5 crash — a single eval sample stuck on sustained
HTTP 500s rode the SDK error budget to exhaustion and killed the whole run,
because the eval rollout had no timeout/retry and no skip-on-failure. These
tests verify the eval path now (a) times out a stuck sample, (b) drops a group
that can't be filled, and (c) never propagates the failure.
"""

import asyncio
from types import SimpleNamespace

import pytest
from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

from interactive_training.rl.metric_util import RLTestSetEvaluator
from interactive_training.rl.types import Env, EnvGroupBuilder, RLDataset, StepResult


class _Env(Env):
    def __init__(self, slot: int, ok: bool):
        self.slot = slot
        self.ok = ok
        self.ob = ModelInput(chunks=[ModelInputChunk(tokens=[slot])])

    async def initial_observation(self):
        return self.ob, []

    async def step(self, action):
        return StepResult(
            reward=float(self.slot), episode_done=True,
            next_observation=self.ob, next_stop_condition=[],
        )

    async def close(self):
        pass


class _Builder(EnvGroupBuilder):
    def __init__(self, ok: bool):
        self._ok = ok

    async def make_envs(self):
        return [_Env(0, self._ok)]

    async def compute_group_rewards(self, trajectory_group, env_group):
        return [(float(e.slot), {}) for e in env_group]

    def logging_tags(self):
        return ["test"]


class _Dataset(RLDataset):
    """One healthy group + one whose sampling wedges forever."""

    def __init__(self):
        self._batches = [[_Builder(ok=True)], [_Builder(ok=False)]]

    def __len__(self):
        return len(self._batches)

    def get_batch(self, i):
        return self._batches[i]


class _StallingClient:
    """sample_async never resolves for the 'bad' env, resolves fast otherwise.

    We key off the prompt's first token (the env slot) — here both envs use
    slot 0, so we instead make ALL samples stall to exercise the timeout path,
    then rely on retry-exhaustion + skip.
    """

    def __init__(self, stall: bool):
        self._stall = stall

    def sample_async(self, **kwargs):
        stall = self._stall

        async def run():
            if stall:
                await asyncio.Event().wait()
            seq = SimpleNamespace(tokens=[7], logprobs=[-0.1])
            return SimpleNamespace(sequences=[seq])

        return run()


async def test_eval_survives_wedged_sample_via_timeout_and_skip():
    """A stuck eval sample times out, the group is dropped, eval returns
    (empty) metrics instead of raising — the run would continue."""
    ev = RLTestSetEvaluator(
        _Dataset(),
        max_tokens=8,
        sample_timeout_sec=0.05,
        max_retries_per_trajectory=1,
        max_extra_trajectory_attempts_per_group=1,
    )
    # All samples stall -> every group times out, exhausts retries, gets dropped.
    metrics = await ev(_StallingClient(stall=True))
    assert metrics == {}, "eval must return empty metrics (not crash) when all groups fail"


async def test_eval_returns_metrics_when_samples_succeed():
    ev = RLTestSetEvaluator(
        _Dataset(),
        max_tokens=8,
        sample_timeout_sec=5.0,
        max_retries_per_trajectory=1,
        max_extra_trajectory_attempts_per_group=1,
    )
    metrics = await ev(_StallingClient(stall=False))
    assert metrics, "eval should produce metrics when sampling succeeds"
    assert any(k.startswith("test/") for k in metrics)


async def test_eval_never_raises_on_sample_error():
    """Even a non-timeout error in sampling is caught and the group dropped."""

    class _ErroringClient:
        def sample_async(self, **kwargs):
            async def run():
                raise RuntimeError("backend 500")

            return run()

    ev = RLTestSetEvaluator(_Dataset(), max_tokens=8)
    metrics = await ev(_ErroringClient())
    assert metrics == {}
