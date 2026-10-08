"""Smoke test for the on-policy distillation code path.

Proves the distillation wiring end-to-end *without* a live engine or GPU:

1. ``PromptOnlyDataset`` yields zero-reward env groups (the only supervision is
   the KL penalty against the teacher).
2. ``_create_kl_reference_client`` creates an *independent* teacher session and
   snapshots its weights into a sampler client — the capability the Azure SDK
   backend previously rejected with a hard ``ValueError``.
3. ``do_sync_training`` with ``kl_penalty_coef > 0`` runs
   ``incorporate_kl_penalty`` against that reference client inside the real
   training loop.

Student == teacher == same model here; the teacher does not need to be good at
any task — this only proves the code path executes.
"""

import asyncio
import json
from typing import Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import chz
import pytest

from azure.ai.finetuningsessions.models import (
    ForwardBackwardOperationResult,
    ModelInput,
    ModelInputChunk,
    OptimStepOperationResult,
    SampleOperationResult,
    SampledSequence,
)

from interactive_training.distillation.datasets import PromptOnlyDataset, PromptOnlyEnv
from interactive_training.rl.problem_env import ProblemGroupBuilder
from interactive_training.rl.train import Config, KLReferenceConfig, do_sync_training
from interactive_training.rl.train_azure import _create_kl_reference_client
from interactive_training.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult
from interactive_training.utils import ml_log


# ---------------------------------------------------------------------------
# Zero-reward environment (no renderer/tokenizer needed for the loop test)
# ---------------------------------------------------------------------------


class ZeroRewardEnv(Env):
    """One-step env that always returns reward 0.0 — mirrors PromptOnlyEnv."""

    async def initial_observation(self):
        return ModelInput(chunks=[ModelInputChunk(tokens=[1, 2, 3])]), [0]

    async def step(self, action):
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=ModelInput(chunks=[ModelInputChunk(tokens=[1, 2, 3])]),
            next_stop_condition=[0],
        )


class ZeroRewardGroupBuilder(EnvGroupBuilder):
    def __init__(self, group_size: int = 4):
        self._group_size = group_size

    async def make_envs(self) -> Sequence[Env]:
        return [ZeroRewardEnv() for _ in range(self._group_size)]

    def logging_tags(self) -> list[str]:
        return ["distill"]


class OneBatchDataset(RLDataset):
    def __init__(self, builders: list[EnvGroupBuilder]):
        self._builders = builders

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return self._builders

    def __len__(self) -> int:
        return 1


# ---------------------------------------------------------------------------
# Mock teacher (reference) client
# ---------------------------------------------------------------------------


class MockReferenceClient:
    """Returns one log-prob per input token so incorporate_kl_penalty aligns.

    ``incorporate_kl_penalty`` reconstructs a full sequence of length N+1 and
    slices ``[1:]`` to length N (== mask length), so returning one value per
    token in the passed ModelInput is exactly right.
    """

    def __init__(self):
        self.call_count = 0

    async def compute_logprobs_async(self, model_input: ModelInput) -> list[float]:
        self.call_count += 1
        n_tokens = sum(len(c.tokens) for c in model_input.chunks)
        return [-0.4] * n_tokens


# ---------------------------------------------------------------------------
# Mock training session (no GPU)
# ---------------------------------------------------------------------------


def _make_mock_session():
    session = MagicMock()

    sample_result = MagicMock(spec=SampleOperationResult)
    seq = MagicMock(spec=SampledSequence)
    seq.tokens = [10, 11, 12]
    seq.logprobs = [-0.5, -0.3, -0.1]
    sample_result.sequences = [seq]
    session.sample_async = AsyncMock(return_value=sample_result)

    def _make_fwd_bwd_future(data_D, *args, **kwargs):
        fwd_bwd_result = MagicMock(spec=ForwardBackwardOperationResult)
        fwd_bwd_result.loss_fn_outputs = [
            {"logprobs": MagicMock(data=[-0.5, -0.3, -0.1, -0.2, -0.4])} for _ in data_D
        ]
        future = AsyncMock()
        future.result_async = AsyncMock(return_value=fwd_bwd_result)
        return future

    session.forward_backward_async = AsyncMock(side_effect=_make_fwd_bwd_future)

    optim_result = MagicMock(spec=OptimStepOperationResult)
    optim_result.metrics = {"skyrl.ai/grad_norm": 1.0, "skyrl.ai/learning_rate": 1e-5}
    optim_future = AsyncMock()
    optim_future.result_async = AsyncMock(return_value=optim_result)
    session.optim_step_async = AsyncMock(return_value=optim_future)

    sampler_result = MagicMock()
    sampler_result.path = "sampler://mock"
    sampler_future = AsyncMock()
    sampler_future.result_async = AsyncMock(return_value=sampler_result)
    session.save_weights_for_sampler_async = AsyncMock(return_value=sampler_future)

    state_result = MagicMock()
    state_result.path = "session_mock/state"
    state_future = AsyncMock()
    state_future.result_async = AsyncMock(return_value=state_result)
    session.save_state_async = AsyncMock(return_value=state_future)

    session.create_sampling_client = MagicMock(return_value=session)
    session.save_weights_and_get_sampling_client_async = AsyncMock(return_value=session)

    mock_tokenizer = MagicMock()
    mock_tokenizer.decode = lambda tokens: "".join(str(t) for t in tokens)
    mock_tokenizer.encode = lambda text: [ord(c) for c in text]
    session.get_tokenizer = MagicMock(return_value=mock_tokenizer)

    return session


def _make_config(dataset, tmp_path, **overrides) -> Config:
    @chz.chz
    class _DatasetBuilder(RLDatasetBuilder):
        async def __call__(self):
            return dataset, None

    defaults = dict(
        learning_rate=1e-5,
        dataset_builder=_DatasetBuilder(),
        model_name="test-model",
        max_tokens=10,
        log_path=str(tmp_path),
        eval_every=0,
        save_every=0,
        remove_constant_reward_groups=False,
        dynamic_sampling=False,
        max_oversample_rounds=10,
        num_groups_to_log=0,
    )
    defaults.update(overrides)
    return Config(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_prompt_only_env_is_zero_reward():
    """PromptOnlyEnv never rewards — KL is the only signal."""
    env = PromptOnlyEnv("What is 2 + 2?", renderer=MagicMock())
    result = asyncio.run(env.step([1, 2, 3]))
    assert result.reward == 0.0
    assert result.episode_done is True


def test_prompt_only_dataset_builds_groups():
    """PromptOnlyDataset yields one prompt-only group per prompt."""
    dataset = PromptOnlyDataset(
        prompts=["p0", "p1", "p2", "p3"],
        batch_size=2,
        group_size=3,
        renderer=MagicMock(),
    )
    assert len(dataset) == 2
    batch = dataset.get_batch(0)
    assert len(batch) == 2
    assert all(isinstance(b, ProblemGroupBuilder) for b in batch)
    assert all(b.num_envs == 3 for b in batch)


@pytest.mark.parametrize(
    ("training_type", "expected_training_type"),
    [
        (None, None),
        ("GlobalStandard", "GlobalStandard"),
        ("globalstandard", "GlobalStandard"),
    ],
)
def test_create_kl_reference_client_creates_independent_session(
    training_type, expected_training_type
):
    """Phase 0: an independent teacher session is created and snapshotted."""
    client = MagicMock()
    client.create_session = AsyncMock(return_value="model_teacher123")

    cfg = MagicMock()
    cfg.kl_reference_config = KLReferenceConfig(base_model="Qwen/Qwen3-32B")
    cfg.model_name = "Qwen/Qwen3-32B"
    cfg.lora_rank = 16

    tokenizer = MagicMock()

    # The reference weights are snapshotted via
    # AzureSDKTrainingClient.save_weights_and_get_sampling_client_async, whose
    # internals wrap SDK futures; patch the class so the helper gets a ready
    # sampling client back.
    ref_sampling_client = MagicMock()
    recorded_ids = []
    fake_training_client = MagicMock()
    fake_training_client.save_weights_and_get_sampling_client_async = AsyncMock(
        return_value=ref_sampling_client
    )

    with patch(
        "interactive_training.rl.train_azure.AzureSDKTrainingClient",
        return_value=fake_training_client,
    ):
        sampling_client, ref_session_id = asyncio.run(
            _create_kl_reference_client(
                client, cfg, tokenizer, training_type=training_type,
                on_session_created=recorded_ids.append,
            )
        )

    assert ref_session_id == "model_teacher123"
    assert sampling_client is ref_sampling_client
    assert recorded_ids == ["model_teacher123"]
    client.create_session.assert_awaited_once()
    _, kwargs = client.create_session.await_args
    assert kwargs["base_model"] == "Qwen/Qwen3-32B"
    assert kwargs["training_type"] == expected_training_type


@pytest.mark.asyncio
async def test_distillation_records_student_and_teacher_sessions(tmp_path, monkeypatch):
    from interactive_training.recipes.distillation import on_policy_distillation_azure as recipe

    monkeypatch.setenv("AZURE_AI_API_KEY", "test-key")
    client = MagicMock()
    client.create_session = AsyncMock(return_value="session_student")
    client.close = AsyncMock()

    async def fake_training(config, training_client, session_id, **kwargs):
        assert session_id == "session_student"
        kwargs["on_reference_session_created"]("session_teacher")

    with patch(
        "azure.ai.finetuningsessions.aio.FineTuningSessionClient", return_value=client
    ), patch.object(recipe.rl_azure, "main", side_effect=fake_training):
        await recipe.cli_main(recipe.CLIConfig(
            project_endpoint="https://example.test",
            log_path=str(tmp_path / "distillation"),
            max_steps=2,
            behavior_if_log_dir_exists="raise",
        ))

    metadata = json.loads((tmp_path / "distillation" / "run_meta.json").read_text())
    assert metadata["azure"]["session_id"] == "session_student"
    assert metadata["azure"]["reference_session_id"] == "session_teacher"


def test_distillation_kl_loop_runs(tmp_path):
    """End-to-end: sample → KL vs reference → train step, with kl_penalty_coef>0.

    This is the exact path the Azure SDK backend previously rejected. Asserts the
    training step ran AND the reference client was queried for teacher log-probs.
    """
    dataset = OneBatchDataset([ZeroRewardGroupBuilder(group_size=4)] * 2)
    session = _make_mock_session()
    reference = MockReferenceClient()
    cfg = _make_config(dataset, tmp_path, kl_penalty_coef=1.0)
    logger = ml_log.setup_logging(log_dir=str(tmp_path))

    asyncio.run(
        do_sync_training(
            start_batch=0,
            end_batch=1,
            num_batches=1,
            cfg=cfg,
            training_client=session,
            kl_reference_client=reference,
            evaluators=[],
            dataset=dataset,
            ml_logger=logger,
            tokenizer=session.get_tokenizer(),
        )
    )

    # Training happened ...
    assert session.forward_backward_async.call_count == 1
    # ... and the teacher (reference) was queried for KL log-probs.
    assert reference.call_count > 0
