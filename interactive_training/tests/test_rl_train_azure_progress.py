import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from interactive_training.rl import train_azure


def test_rl_train_azure_final_checkpoint_uses_actual_progress(tmp_path):
    save_checkpoint = AsyncMock(return_value={"sampler_path": "sampler"})
    evaluator = AsyncMock(return_value={"test/nll": 1.25})
    cfg = _main_cfg(tmp_path, num_batches=10, evaluators=[lambda: evaluator], eval_every=1)

    async def fake_training_func(**_kwargs):
        return 3, 2

    client, ml_logger = _run_main(cfg, save_checkpoint, fake_training_func)

    save_checkpoint.assert_awaited_once()
    kwargs = save_checkpoint.await_args.kwargs
    assert kwargs["loop_state"] == {"batch": 3}
    assert kwargs["step_number"] == 2
    client.create_sampling_client.assert_called_once_with("sampler")
    evaluator.assert_awaited_once_with("sampling-client", step=3)
    ml_logger.log_metrics.assert_called_once()
    assert ml_logger.log_metrics.call_args.kwargs["step"] == 3


def test_rl_train_azure_final_checkpoint_preserves_prompt_cursor(tmp_path):
    save_checkpoint = AsyncMock(return_value={"sampler_path": "sampler"})
    cfg = _main_cfg(tmp_path, num_batches=10)

    async def fake_training_func(**kwargs):
        kwargs["final_loop_state"]["prompt_cursor"] = 9
        return 3, 2

    _run_main(cfg, save_checkpoint, fake_training_func)

    assert save_checkpoint.await_args.kwargs["loop_state"] == {
        "batch": 3,
        "prompt_cursor": 9,
    }


def test_rl_train_azure_preserves_async_seed_progress(tmp_path):
    from interactive_training.rl.train import AsyncConfig

    cfg = _main_cfg(tmp_path, num_batches=10)
    cfg.async_config = AsyncConfig(groups_per_batch=2, max_steps_off_policy=1)
    save_checkpoint = AsyncMock(return_value={"sampler_path": "sampler"})

    async def fake_training_func(**kwargs):
        assert kwargs["resume_group_ordinal"] == 123
        kwargs["final_loop_state"]["async_group_ordinal"] = 150
        return 3, 2

    with patch.object(train_azure, "do_async_training", new=fake_training_func):
        _run_main(cfg, save_checkpoint, fake_training_func,
                  resume_info={"batch": 2, "async_group_ordinal": 123,
                               "state_path": "model_abc/2"})
    assert save_checkpoint.await_args.kwargs["loop_state"] == {
        "batch": 3, "async_group_ordinal": 150,
    }


def test_rl_train_azure_skips_final_checkpoint_when_no_progress(tmp_path):
    save_checkpoint = AsyncMock(return_value={"sampler_path": "sampler"})
    cfg = _main_cfg(tmp_path, num_batches=4, evaluators=[lambda: AsyncMock()], eval_every=1)

    async def fake_training_func(**_kwargs):
        return 1, None

    client, ml_logger = _run_main(
        cfg,
        save_checkpoint,
        fake_training_func,
            resume_info={"batch": 1, "state_path": "model_source/state"},
    )

    save_checkpoint.assert_not_awaited()
    client.create_sampling_client.assert_not_called()
    ml_logger.log_metrics.assert_not_called()


def test_rl_train_azure_forwards_resume_prompt_cursor(tmp_path):
    save_checkpoint = AsyncMock(return_value={"sampler_path": "sampler"})
    cfg = _main_cfg(tmp_path, num_batches=30)
    received: dict = {}

    async def fake_training_func(**kwargs):
        received.update(kwargs)
        return 21, None

    _run_main(
        cfg,
        save_checkpoint,
        fake_training_func,
        resume_info={
            "batch": 21,
            "prompt_cursor": 117,
            "state_path": "model_abc/21",
        },
    )

    assert received["start_batch"] == 21
    assert received["end_batch"] == 30
    assert received["resume_prompt_cursor"] == 117


@pytest.mark.parametrize(
    "training_type",
    [None, "GlobalStandard"],
)
def test_rl_main_forwards_training_type_to_reference_session(tmp_path, training_type):
    cfg = _main_cfg(tmp_path)
    cfg.kl_penalty_coef = 1
    reference_factory = AsyncMock(return_value=(MagicMock(), "session_reference"))

    async def fake_training_func(**_kwargs):
        return 0, None

    _run_main(
        cfg,
        AsyncMock(),
        fake_training_func,
        training_type=training_type,
        reference_factory=reference_factory,
    )

    reference_factory.assert_awaited_once()
    assert reference_factory.await_args.kwargs["training_type"] == training_type


def _run_main(
    cfg,
    save_checkpoint,
    training_func,
    resume_info=None,
    *,
    training_type=None,
    reference_factory=None,
):
    client = MagicMock()
    client.create_lora_training_client_async = AsyncMock(return_value="training-client")
    client.create_training_client_from_state_with_optimizer_async = AsyncMock(return_value="training-client")
    client.close_session = AsyncMock()
    client.close = AsyncMock()
    training_client = MagicMock()
    training_client.get_tokenizer.return_value = MagicMock()
    training_client.create_sampling_client = MagicMock(return_value="sampling-client")

    async def create_lora_training_client_async(*_args, **_kwargs):
        return training_client

    async def create_training_client_from_state_with_optimizer_async(*_args, **_kwargs):
        return training_client

    client.create_lora_training_client_async = create_lora_training_client_async
    client.create_training_client_from_state_with_optimizer_async = (
        create_training_client_from_state_with_optimizer_async
    )

    with patch.object(train_azure, "ml_log") as ml_mock, \
         patch.object(train_azure, "get_tokenizer", return_value=MagicMock()), \
         patch.object(train_azure, "AzureSDKTrainingClient", return_value=training_client), \
         patch.object(train_azure.checkpoint_utils, "get_last_checkpoint", return_value=resume_info), \
         patch.object(train_azure.checkpoint_utils, "save_checkpoint_async", new=save_checkpoint), \
         patch.object(train_azure, "do_sync_training", new=training_func), \
         patch.object(
             train_azure,
             "_create_kl_reference_client",
             new=reference_factory if reference_factory is not None else AsyncMock(),
         ):
        ml_mock.setup_logging.return_value = MagicMock()
        asyncio.run(
            train_azure.main(cfg, client, "session_id", training_type=training_type)
        )
        return training_client, ml_mock.setup_logging.return_value


def _main_cfg(tmp_path, num_batches=4, evaluators=None, eval_every=0):
    cfg = MagicMock()
    cfg.log_path = str(tmp_path)
    cfg.wandb_project = None
    cfg.wandb_name = None
    cfg.enable_trace = False
    cfg.tokenizer_name = None
    cfg.model_name = "Qwen/Qwen3-0.6B"
    cfg.kl_penalty_coef = 0
    cfg.async_config = None
    cfg.stream_minibatch_config = None
    cfg.eval_strategy = "steps"
    cfg.eval_every = eval_every
    cfg.ttl_seconds = None
    cfg.evaluator_builders = evaluators or []
    cfg.num_groups_to_log = 0
    cfg.max_tokens = 16
    cfg.max_steps = None
    cfg.max_wall_clock_seconds = None
    dataset = MagicMock()
    dataset.__len__ = lambda _self: num_batches

    async def dataset_builder():
        return dataset, None

    cfg.dataset_builder = dataset_builder
    return cfg
