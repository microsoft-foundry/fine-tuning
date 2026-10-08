import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from interactive_training.rl import train as rl_train


def test_periodic_rl_checkpoint_persists_internal_step(tmp_path):
    client = MagicMock()
    client.create_sampling_client = MagicMock(return_value="sampling-client")
    client.save_weights_and_get_sampling_client_async = AsyncMock(
        return_value="sampling-client-no-save"
    )
    client.save_state_async = AsyncMock(return_value=_future("service://model_source/weights/4"))
    client.save_weights_for_sampler_async = AsyncMock(return_value=_future("4"))

    asyncio.run(
        rl_train.save_checkpoint_and_get_sampling_client(
            training_client=client,
            i_batch=4,
            log_path=str(tmp_path),
            save_every=2,
            start_batch=0,
        )
    )

    row = rl_train.checkpoint_utils.get_last_checkpoint(str(tmp_path))

    assert row is not None
    assert row["name"] == "4"
    assert row["batch"] == 4
    assert row["step"] == 3
    assert row["state_path"] == "model_source/4"


def test_final_rl_checkpoint_persists_actual_completed_step(tmp_path):
    save_checkpoint = AsyncMock(return_value={"sampler_path": "sampler"})
    cfg = _main_cfg(tmp_path, num_batches=4)

    async def fake_training_func(**_kwargs):
        return 4, 3

    _run_main(cfg, save_checkpoint, fake_training_func)

    save_checkpoint.assert_awaited_once()
    kwargs = save_checkpoint.await_args.kwargs
    assert kwargs["loop_state"] == {"batch": 4}
    assert kwargs["step_number"] == 3


def test_main_caps_end_batch_at_max_steps(tmp_path):
    """Direct rl.train.main() must honor cfg.max_steps, not run the whole dataset."""
    save_checkpoint = AsyncMock(return_value={"sampler_path": "sampler"})
    cfg = _main_cfg(tmp_path, num_batches=100)
    cfg.max_steps = 7

    captured: dict = {}

    async def fake_training_func(**kwargs):
        captured.update(kwargs)
        return kwargs["end_batch"], kwargs["end_batch"] - 1

    _run_main(cfg, save_checkpoint, fake_training_func)

    # start_batch=0, dataset=100, cap=7 -> end_batch=7 (not 100).
    assert captured["end_batch"] == 7
    assert captured["num_batches"] == 100
    assert captured["start_batch"] == 0


def test_main_without_max_steps_runs_whole_dataset(tmp_path):
    """With max_steps=None, end_batch is the full dataset length (regression guard)."""
    save_checkpoint = AsyncMock(return_value={"sampler_path": "sampler"})
    cfg = _main_cfg(tmp_path, num_batches=42)  # _main_cfg sets max_steps=None

    captured: dict = {}

    async def fake_training_func(**kwargs):
        captured.update(kwargs)
        return kwargs["end_batch"], kwargs["end_batch"] - 1

    _run_main(cfg, save_checkpoint, fake_training_func)

    assert captured["end_batch"] == 42


def test_final_rl_checkpoint_uses_actual_progress_on_early_stop(tmp_path):
    save_checkpoint = AsyncMock(return_value={"sampler_path": "sampler"})
    evaluator = AsyncMock(return_value={"test/nll": 1.25})
    cfg = _main_cfg(tmp_path, num_batches=10, evaluators=[lambda: evaluator], eval_every=1)

    async def fake_training_func(**_kwargs):
        return 3, 2

    training_client, _service_client, ml_logger = _run_main(
        cfg,
        save_checkpoint,
        fake_training_func,
    )

    save_checkpoint.assert_awaited_once()
    kwargs = save_checkpoint.await_args.kwargs
    assert kwargs["loop_state"] == {"batch": 3}
    assert kwargs["step_number"] == 2
    training_client.create_sampling_client.assert_called_once_with("sampler")
    evaluator.assert_awaited_once_with("sampling-client", step=3)
    ml_logger.log_metrics.assert_called_once()
    assert ml_logger.log_metrics.call_args.kwargs["step"] == 3


def test_epoch_rl_final_eval_runs_after_partial_early_stop(tmp_path):
    evaluator = AsyncMock(return_value={"test/nll": 1.25})
    save_checkpoint = AsyncMock(return_value={"sampler_path": "sampler"})
    cfg = _main_cfg(tmp_path, num_batches=10, evaluators=[lambda: evaluator])
    cfg.eval_strategy = "epoch"

    async def fake_training_func(**_kwargs):
        return 3, 2

    training_client, _service_client, ml_logger = _run_main(
        cfg, save_checkpoint, fake_training_func
    )

    save_checkpoint.assert_awaited_once()
    training_client.create_sampling_client.assert_called_once_with("sampler")
    evaluator.assert_awaited_once_with("sampling-client", step=3)
    ml_logger.log_metrics.assert_called_once()
    assert ml_logger.log_metrics.call_args.kwargs["step"] == 3


def test_epoch_rl_final_eval_runs_after_complete_dataset_traversal(tmp_path):
    evaluator = AsyncMock(return_value={"test/nll": 1.25})
    save_checkpoint = AsyncMock(return_value={"sampler_path": "sampler"})
    cfg = _main_cfg(tmp_path, num_batches=4, evaluators=[lambda: evaluator])
    cfg.eval_strategy = "epoch"

    async def fake_training_func(**_kwargs):
        return 4, 3

    training_client, _service_client, ml_logger = _run_main(
        cfg, save_checkpoint, fake_training_func
    )

    training_client.create_sampling_client.assert_called_once_with("sampler")
    evaluator.assert_awaited_once_with("sampling-client", step=4)
    ml_logger.log_metrics.assert_called_once()
    assert ml_logger.log_metrics.call_args.kwargs["step"] == 4


def test_rl_evaluation_strategy_scheduling():
    assert rl_train.should_evaluate_rl_iteration(
        eval_strategy="epoch", eval_every=1, step=0, is_fresh_run=True
    )
    assert not rl_train.should_evaluate_rl_iteration(
        eval_strategy="epoch", eval_every=1, step=1, is_fresh_run=True
    )
    assert not rl_train.should_evaluate_rl_iteration(
        eval_strategy="epoch", eval_every=1, step=0, is_fresh_run=False
    )
    assert rl_train.should_evaluate_rl_iteration(
        eval_strategy="steps", eval_every=2, step=2, is_fresh_run=False
    )
    assert not rl_train.should_evaluate_rl_iteration(
        eval_strategy="steps", eval_every=2, step=3, is_fresh_run=True
    )
    assert rl_train.should_evaluate_rl_final(
        eval_strategy="epoch",
        eval_every=1,
        actual_next_batch=4,
        num_batches=4,
    )
    assert rl_train.should_evaluate_rl_final(
        eval_strategy="epoch",
        eval_every=1,
        actual_next_batch=3,
        num_batches=4,
    )
    assert rl_train.should_evaluate_rl_final(
        eval_strategy="steps",
        eval_every=20,
        actual_next_batch=3,
        num_batches=4,
    )


def test_final_rl_checkpoint_skips_when_no_new_progress(tmp_path):
    save_checkpoint = AsyncMock(return_value={"sampler_path": "sampler"})
    cfg = _main_cfg(tmp_path, num_batches=4)

    async def fake_training_func(**_kwargs):
        return 1, None

    cfg.eval_every = 1
    cfg.evaluator_builders = [lambda: AsyncMock(return_value={"test/nll": 1.25})]
    training_client, _service_client, ml_logger = _run_main(
        cfg,
        save_checkpoint,
        fake_training_func,
        resume_batch=1,
    )

    save_checkpoint.assert_not_awaited()
    training_client.create_sampling_client.assert_not_called()
    ml_logger.log_metrics.assert_not_called()


def test_resumed_batch_zero_is_not_treated_as_a_fresh_epoch_run(tmp_path):
    save_checkpoint = AsyncMock(return_value={"sampler_path": "sampler"})
    cfg = _main_cfg(tmp_path, num_batches=4)
    observed_is_fresh_run = None

    async def fake_training_func(**kwargs):
        nonlocal observed_is_fresh_run
        observed_is_fresh_run = kwargs["is_fresh_run"]
        return 0, None

    _run_main(
        cfg,
        save_checkpoint,
        fake_training_func,
        resume_batch=0,
    )

    assert observed_is_fresh_run is False


def test_stream_minibatch_empty_step_does_not_train_or_complete(tmp_path):
    cfg = MagicMock()
    cfg.stream_minibatch_config.groups_per_batch = 1
    cfg.stream_minibatch_config.num_minibatches = 1
    cfg.num_substeps = 1
    cfg.num_groups_to_log = 0
    cfg.log_path = str(tmp_path)
    cfg.loss_fn = "reinforce"
    cfg.loss_fn_config = None
    training_client = MagicMock()
    training_client.optim_step_async = AsyncMock()
    queue = asyncio.Queue()
    queue.put_nowait(None)

    sampling_client, metrics = asyncio.run(
        rl_train.do_train_step_streaming_and_get_sampling_client(
            cfg=cfg,
            i_batch=3,
            trajectory_groups_queue=queue,
            training_client=training_client,
            kl_reference_client=None,
            tokenizer=MagicMock(),
        )
    )

    assert sampling_client is None
    assert metrics["train/skipped_no_valid_groups"] == 1
    training_client.optim_step_async.assert_not_called()


def _run_main(cfg, save_checkpoint, training_func, resume_batch=None):
    training_client = MagicMock()
    training_client.get_tokenizer.return_value = MagicMock()
    training_client.create_sampling_client = MagicMock(return_value="sampling-client")
    service_client = MagicMock()
    if resume_batch is None:
        get_last_checkpoint = MagicMock(return_value=None)
        service_client.create_lora_training_client_async = AsyncMock(return_value=training_client)
    else:
        get_last_checkpoint = MagicMock(
            return_value={"batch": resume_batch, "state_path": "model_source/state"}
        )
        service_client.create_training_client_from_state_with_optimizer_async = AsyncMock(
            return_value=training_client
        )

    with patch.object(rl_train, "FineTuningSessionClient", return_value=service_client), \
         patch.object(rl_train, "ml_log") as ml_mock, \
         patch.object(rl_train.checkpoint_utils, "get_last_checkpoint", get_last_checkpoint), \
         patch.object(rl_train.checkpoint_utils, "save_checkpoint_async", new=save_checkpoint), \
         patch.object(rl_train, "do_sync_training", new=training_func):
        ml_mock.setup_logging.return_value = MagicMock()
        asyncio.run(rl_train.main(cfg))
        return training_client, service_client, ml_mock.setup_logging.return_value


def _main_cfg(tmp_path, num_batches=4, evaluators=None, eval_every=0):
    cfg = MagicMock()
    cfg.log_path = str(tmp_path)
    cfg.wandb_project = None
    cfg.wandb_name = None
    cfg.enable_trace = False
    cfg.base_url = "https://interactive-post-training.test"
    cfg.load_checkpoint_path = None
    cfg.model_name = "Qwen/Qwen3-0.6B"
    cfg.lora_rank = 4
    cfg.lora_alpha = 8
    cfg.kl_penalty_coef = 0
    cfg.kl_reference_config = None
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


def _future(path):
    result = MagicMock()
    result.path = path
    future = MagicMock()
    future.result_async = AsyncMock(return_value=result)
    return future
