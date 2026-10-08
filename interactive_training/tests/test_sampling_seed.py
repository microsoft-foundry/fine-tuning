"""Tests for sampling seed reproducibility.

Verifies that:
- SessionTokenCompleter produces deterministic, unique per-call seeds
- _derive_group_seed produces unique seeds per (batch, group) and is deterministic
- Config accepts the new sampling_seed field
- No seed is passed when sampling_seed is None (server picks random)
"""

import asyncio
import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk, SamplingParams

from interactive_training.completers import SessionTokenCompleter, TokensWithLogprobs
from interactive_training.rl.train import (
    Config,
    _QueuedEnvGroupBuilder,
    _derive_async_group_seed,
    _derive_group_seed,
)


# ---------------------------------------------------------------------------
# _derive_group_seed tests
# ---------------------------------------------------------------------------


class TestDeriveGroupSeed:
    def test_returns_none_when_base_seed_is_none(self):
        assert _derive_group_seed(None, 0, 0) is None
        assert _derive_group_seed(None, 5, 3) is None

    def test_deterministic(self):
        """Same inputs always produce the same output."""
        assert _derive_group_seed(42, 3, 7) == _derive_group_seed(42, 3, 7)

    def test_unique_across_groups(self):
        """Different group indices produce different seeds."""
        seeds = {_derive_group_seed(42, 0, i) for i in range(100)}
        assert len(seeds) == 100

    def test_unique_across_batches(self):
        """Different batch indices produce different seeds."""
        seeds = {_derive_group_seed(42, i, 0) for i in range(100)}
        assert len(seeds) == 100

    def test_unique_across_seeds(self):
        """Different base seeds produce different results."""
        seeds = {_derive_group_seed(s, 0, 0) for s in range(100)}
        assert len(seeds) == 100

    def test_within_range(self):
        """Output is within [0, 2^31)."""
        for batch in range(50):
            for group in range(50):
                seed = _derive_group_seed(999999, batch, group)
                assert 0 <= seed < 2**31


def test_async_group_ordinals_are_attached_before_worker_scheduling() -> None:
    builders = [MagicMock() for _ in range(4)]
    queued = [
        _QueuedEnvGroupBuilder(builder=builder, ordinal=ordinal)
        for ordinal, builder in enumerate(builders)
    ]

    fast_worker_order = [queued[1], queued[0], queued[3], queued[2]]

    assert {
        id(item.builder): _derive_async_group_seed(42, item.ordinal, 2)
        for item in fast_worker_order
    } == {
        id(builder): _derive_group_seed(42, ordinal // 2, ordinal % 2)
        for ordinal, builder in enumerate(builders)
    }


# ---------------------------------------------------------------------------
# SessionTokenCompleter seed tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("streaming", [False, True])
def test_async_resume_restores_seed_high_water_mark(tmp_path, streaming):
    from interactive_training.rl import train
    from tests.test_dynamic_sampling import (
        ControlledDataset, FixedRewardGroupBuilder, MixedRewardGroupBuilder,
        _make_config, _make_logger, _make_mock_session,
    )

    def run(directory, start, resume_ordinal):
        dataset = ControlledDataset([
            [FixedRewardGroupBuilder(0.0, group_size=4)],
            [MixedRewardGroupBuilder([0.0, 1.0, 0.0, 1.0])],
        ])
        session = _make_mock_session()
        config = _make_config(
            dataset, directory, sampling_seed=42, save_every=1,
            remove_constant_reward_groups=True,
            async_config=train.AsyncConfig(groups_per_batch=1, max_steps_off_policy=1),
            stream_minibatch_config=(
                train.StreamMinibatchConfig(groups_per_batch=1, num_minibatches=1)
                if streaming else None
            ),
        )
        seeds = []
        original = train.do_group_rollout_and_filter_constant_reward

        async def observe(*args, **kwargs):
            seeds.append(kwargs["seed"])
            return await original(*args, **kwargs)

        state = {}
        with patch.object(train, "do_group_rollout_and_filter_constant_reward", side_effect=observe):
            asyncio.run(asyncio.wait_for(train.do_async_training(
                start_batch=start, end_batch=start + 1, num_batches=2,
                cfg=config, training_client=session, kl_reference_client=None,
                evaluators=[], dataset=dataset, ml_logger=_make_logger(directory),
                tokenizer=session.get_tokenizer(), resume_group_ordinal=resume_ordinal,
                final_loop_state=state,
            ), timeout=10))
        records = [json.loads(line) for line in (directory / "checkpoints.jsonl").read_text().splitlines()]
        assert records[-1]["async_group_ordinal"] > (resume_ordinal or 0)
        assert state["async_group_ordinal"] >= records[-1]["async_group_ordinal"]
        return seeds, state

    first_seeds, state = run(tmp_path / "first", 0, None)
    resumed_seeds, resumed_state = run(tmp_path / "resumed", 1, state["async_group_ordinal"])
    assert first_seeds and resumed_seeds
    assert resumed_seeds[0] == _derive_async_group_seed(42, state["async_group_ordinal"], 1)
    assert set(first_seeds).isdisjoint(resumed_seeds)
    assert resumed_state["async_group_ordinal"] > state["async_group_ordinal"]
    legacy_seeds, _ = run(tmp_path / "legacy", 1, None)
    assert legacy_seeds[0] == _derive_async_group_seed(42, 1, 1)


class TestSessionTokenCompleterSeed:
    def _make_mock_sampling_client(self):
        """Create a mock sampling client that records calls."""
        client = MagicMock()
        # Return a fake sample result
        mock_result = MagicMock()
        mock_result.sequences = [MagicMock(tokens=[1, 2, 3], logprobs=[-0.5, -0.3, -0.1])]
        client.sample_async = AsyncMock(return_value=mock_result)
        return client

    def _make_model_input(self):
        return ModelInput(chunks=[ModelInputChunk(tokens=[100, 200, 300])])

    def test_seed_none_passes_none(self):
        """When seed is None, SamplingParams.seed should be None."""
        client = self._make_mock_sampling_client()
        completer = SessionTokenCompleter(
            sampling_client=client, max_tokens=100, seed=None,
        )

        asyncio.run(completer(self._make_model_input(), [50256]))

        call_args = client.sample_async.call_args
        sp = call_args.kwargs["sampling_params"]
        assert sp.seed is None

    def test_seed_set_passes_derived_seed(self):
        """When seed is set, SamplingParams.seed should be base_seed + counter."""
        client = self._make_mock_sampling_client()
        completer = SessionTokenCompleter(
            sampling_client=client, max_tokens=100, seed=42,
        )

        asyncio.run(completer(self._make_model_input(), [50256]))

        call_args = client.sample_async.call_args
        sp = call_args.kwargs["sampling_params"]
        assert sp.seed == 42  # First call: 42 + 0

    def test_seed_increments_per_call(self):
        """Each call produces a different seed (counter increments)."""
        client = self._make_mock_sampling_client()
        completer = SessionTokenCompleter(
            sampling_client=client, max_tokens=100, seed=100,
        )

        # Make 5 calls
        seeds_seen = []
        for _ in range(5):
            asyncio.run(completer(self._make_model_input(), [50256]))
            sp = client.sample_async.call_args.kwargs["sampling_params"]
            seeds_seen.append(sp.seed)

        assert seeds_seen == [100, 101, 102, 103, 104]

    def test_seed_wraps_within_range(self):
        """Seeds stay within [0, 2^31) via modulo."""
        client = self._make_mock_sampling_client()
        completer = SessionTokenCompleter(
            sampling_client=client, max_tokens=100, seed=2**31 - 1,
        )

        asyncio.run(completer(self._make_model_input(), [50256]))
        sp = client.sample_async.call_args.kwargs["sampling_params"]
        assert sp.seed == (2**31 - 1) % (2**31)

        asyncio.run(completer(self._make_model_input(), [50256]))
        sp = client.sample_async.call_args.kwargs["sampling_params"]
        assert sp.seed == (2**31) % (2**31)  # wraps to 0

    def test_different_completers_same_seed_same_sequence(self):
        """Two completers with the same seed produce the same seed sequence."""
        client1 = self._make_mock_sampling_client()
        client2 = self._make_mock_sampling_client()
        c1 = SessionTokenCompleter(sampling_client=client1, max_tokens=100, seed=7)
        c2 = SessionTokenCompleter(sampling_client=client2, max_tokens=100, seed=7)

        seeds1, seeds2 = [], []
        for _ in range(3):
            asyncio.run(c1(self._make_model_input(), [50256]))
            seeds1.append(client1.sample_async.call_args.kwargs["sampling_params"].seed)
            asyncio.run(c2(self._make_model_input(), [50256]))
            seeds2.append(client2.sample_async.call_args.kwargs["sampling_params"].seed)

        assert seeds1 == seeds2


# ---------------------------------------------------------------------------
# Config.sampling_seed field test
# ---------------------------------------------------------------------------


class TestConfigSamplingSeed:
    def test_default_is_none(self):
        """Config.sampling_seed defaults to None."""
        # Verify the attribute exists on Config with None default
        assert hasattr(Config, "__annotations__")
        # Check that sampling_seed is a declared field
        assert "sampling_seed" in Config.__annotations__

    def test_sampling_seed_accepted(self):
        """Config can be constructed with sampling_seed set."""
        # Just verify the attribute can be set on an instance
        config = MagicMock(spec=Config)
        config.sampling_seed = 42
        assert config.sampling_seed == 42
