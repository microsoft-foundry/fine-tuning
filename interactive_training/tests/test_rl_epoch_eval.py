from __future__ import annotations

import asyncio

import pytest

from interactive_training.rl.train import (
    do_async_training,
    do_sync_training,
    do_sync_training_with_stream_minibatch,
)

from tests.test_dynamic_sampling import (
    AsyncConfig,
    ControlledDataset,
    MixedRewardGroupBuilder,
    StreamMinibatchConfig,
    _make_config,
    _make_logger,
    _make_mock_session,
)


def test_sync_epoch_strategy_evaluates_only_initial_model(tmp_path):
    dataset = ControlledDataset([
        [MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0])],
        [MixedRewardGroupBuilder([0.0, 1.0, 0.0, 1.0])],
    ])
    session = _make_mock_session()
    cfg = _make_config(
        dataset,
        tmp_path,
        eval_strategy="epoch",
        eval_every=1,
    )
    logger = _make_logger(tmp_path)
    training_steps_seen: list[int] = []

    async def evaluator(_sampling_client, *, step=None):
        training_steps_seen.append(session.forward_backward_async.call_count)
        return {"test/nll": 1.0}

    asyncio.run(
        do_sync_training(
            start_batch=0,
            end_batch=2,
            num_batches=2,
            cfg=cfg,
            training_client=session,
            kl_reference_client=None,
            evaluators=[evaluator],
            dataset=dataset,
            ml_logger=logger,
            tokenizer=session.get_tokenizer(),
        )
    )
    logger.close()

    assert training_steps_seen == [0]
    assert session.forward_backward_async.call_count == 2


def test_sync_epoch_strategy_does_not_repeat_initial_eval_on_resume(tmp_path):
    dataset = ControlledDataset([
        [MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0])],
        [MixedRewardGroupBuilder([0.0, 1.0, 0.0, 1.0])],
    ])
    session = _make_mock_session()
    cfg = _make_config(dataset, tmp_path, eval_strategy="epoch", eval_every=1)
    logger = _make_logger(tmp_path)
    evaluation_steps: list[int] = []

    async def evaluator(_sampling_client, *, step=None):
        evaluation_steps.append(session.forward_backward_async.call_count)
        return {"test/nll": 1.0}

    asyncio.run(
        do_sync_training(
            start_batch=0,
            end_batch=1,
            num_batches=2,
            cfg=cfg,
            training_client=session,
            kl_reference_client=None,
            evaluators=[evaluator],
            dataset=dataset,
            ml_logger=logger,
            tokenizer=session.get_tokenizer(),
            is_fresh_run=False,
        )
    )
    logger.close()

    assert evaluation_steps == []
    assert session.forward_backward_async.call_count == 1


def test_async_epoch_strategy_evaluates_before_training_only(tmp_path):
    dataset = ControlledDataset([
        [
            MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0]),
            MixedRewardGroupBuilder([0.0, 1.0, 0.0, 1.0]),
        ],
    ])
    session = _make_mock_session()
    cfg = _make_config(
        dataset,
        tmp_path,
        eval_strategy="epoch",
        eval_every=1,
        async_config=AsyncConfig(max_steps_off_policy=100, groups_per_batch=2),
    )
    logger = _make_logger(tmp_path)
    training_steps_seen: list[int] = []
    checkpoint_states_seen: list[int] = []

    async def evaluator(_sampling_client, *, step=None):
        training_steps_seen.append(session.forward_backward_async.call_count)
        checkpoint_states_seen.append(session.save_state_async.await_count)
        return {"test/nll": 1.0}

    async def run():
        await asyncio.wait_for(
            do_async_training(
                start_batch=0,
                end_batch=1,
                num_batches=1,
                cfg=cfg,
                training_client=session,
                kl_reference_client=None,
                evaluators=[evaluator],
                dataset=dataset,
                ml_logger=logger,
                tokenizer=session.get_tokenizer(),
            ),
            timeout=30,
        )

    asyncio.run(run())
    logger.close()

    assert training_steps_seen == [0]
    assert checkpoint_states_seen == [0]
    assert session.forward_backward_async.call_count >= 1


def test_async_epoch_strategy_skips_baseline_on_resume(tmp_path):
    # A resumed async run (is_fresh_run=False) must NOT re-run the initial
    # baseline evaluation, and epoch mode runs no periodic evals, so the
    # evaluator is never invoked. The loop must still make progress and
    # terminate cleanly (no deadlock on sampling_client_updated_event).
    dataset = ControlledDataset([
        [
            MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0]),
            MixedRewardGroupBuilder([0.0, 1.0, 0.0, 1.0]),
        ],
    ])
    session = _make_mock_session()
    cfg = _make_config(
        dataset,
        tmp_path,
        eval_strategy="epoch",
        eval_every=1,
        async_config=AsyncConfig(max_steps_off_policy=100, groups_per_batch=2),
    )
    logger = _make_logger(tmp_path)
    eval_calls: list[int] = []

    async def evaluator(_sampling_client, *, step=None):
        eval_calls.append(session.forward_backward_async.call_count)
        return {"test/nll": 1.0}

    async def run():
        await asyncio.wait_for(
            do_async_training(
                start_batch=0,
                end_batch=1,
                num_batches=1,
                cfg=cfg,
                training_client=session,
                kl_reference_client=None,
                evaluators=[evaluator],
                dataset=dataset,
                ml_logger=logger,
                tokenizer=session.get_tokenizer(),
                is_fresh_run=False,
            ),
            timeout=30,
        )

    asyncio.run(run())
    logger.close()

    assert eval_calls == []
    assert session.forward_backward_async.call_count >= 1


def test_async_steps_strategy_evaluates_restored_checkpoint_on_resume(tmp_path):
    dataset = ControlledDataset([
        [
            MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0]),
            MixedRewardGroupBuilder([0.0, 1.0, 0.0, 1.0]),
        ],
        [
            MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0]),
            MixedRewardGroupBuilder([0.0, 1.0, 0.0, 1.0]),
        ],
    ])
    session = _make_mock_session()
    cfg = _make_config(
        dataset,
        tmp_path,
        eval_strategy="steps",
        eval_every=3,
        async_config=AsyncConfig(max_steps_off_policy=100, groups_per_batch=2),
    )
    logger = _make_logger(tmp_path)
    training_steps_seen: list[int] = []

    async def evaluator(_sampling_client):
        training_steps_seen.append(session.forward_backward_async.call_count)
        return {"test/nll": 1.0}

    async def run():
        await asyncio.wait_for(
            do_async_training(
                start_batch=1,
                end_batch=2,
                num_batches=2,
                cfg=cfg,
                training_client=session,
                kl_reference_client=None,
                evaluators=[evaluator],
                dataset=dataset,
                ml_logger=logger,
                tokenizer=session.get_tokenizer(),
                is_fresh_run=False,
            ),
            timeout=30,
        )

    asyncio.run(run())
    logger.close()

    assert training_steps_seen == [0]
    assert session.forward_backward_async.call_count >= 1


def test_stream_minibatch_epoch_strategy_evaluates_only_initial_model(tmp_path):
    # The streaming-minibatch loop is a third training path; verify epoch mode
    # runs the step-0 baseline exactly once and no periodic evals thereafter.
    dataset = ControlledDataset([
        [MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0])],
        [MixedRewardGroupBuilder([0.0, 1.0, 0.0, 1.0])],
    ])
    session = _make_mock_session()
    cfg = _make_config(
        dataset,
        tmp_path,
        eval_strategy="epoch",
        eval_every=1,
        stream_minibatch_config=StreamMinibatchConfig(
            groups_per_batch=1, num_minibatches=1
        ),
    )
    logger = _make_logger(tmp_path)
    training_steps_seen: list[int] = []

    async def evaluator(_sampling_client, *, step=None):
        training_steps_seen.append(session.forward_backward_async.call_count)
        return {"test/nll": 1.0}

    asyncio.run(
        do_sync_training_with_stream_minibatch(
            start_batch=0,
            end_batch=2,
            num_batches=2,
            cfg=cfg,
            training_client=session,
            kl_reference_client=None,
            evaluators=[evaluator],
            dataset=dataset,
            ml_logger=logger,
            tokenizer=session.get_tokenizer(),
        )
    )
    logger.close()

    assert training_steps_seen == [0]
    assert session.forward_backward_async.call_count == 2


def test_stream_minibatch_epoch_strategy_skips_baseline_on_resume(tmp_path):
    # Streaming-minibatch resume at batch 0 must honor is_fresh_run=False and
    # skip the baseline eval.
    dataset = ControlledDataset([
        [MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0])],
        [MixedRewardGroupBuilder([0.0, 1.0, 0.0, 1.0])],
    ])
    session = _make_mock_session()
    cfg = _make_config(
        dataset,
        tmp_path,
        eval_strategy="epoch",
        eval_every=1,
        stream_minibatch_config=StreamMinibatchConfig(
            groups_per_batch=1, num_minibatches=1
        ),
    )
    logger = _make_logger(tmp_path)
    eval_calls: list[int] = []

    async def evaluator(_sampling_client, *, step=None):
        eval_calls.append(session.forward_backward_async.call_count)
        return {"test/nll": 1.0}

    asyncio.run(
        do_sync_training_with_stream_minibatch(
            start_batch=0,
            end_batch=1,
            num_batches=2,
            cfg=cfg,
            training_client=session,
            kl_reference_client=None,
            evaluators=[evaluator],
            dataset=dataset,
            ml_logger=logger,
            tokenizer=session.get_tokenizer(),
            is_fresh_run=False,
        )
    )
    logger.close()

    assert eval_calls == []


@pytest.mark.parametrize("eval_every", [0, 2, 10])
def test_epoch_strategy_rejects_unsupported_eval_every(tmp_path, eval_every):
    # Epoch cadence is explicit: only 1 means evaluate the initial model and
    # after the dataset traversal.
    dataset = ControlledDataset([
        [MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0])],
    ])
    with pytest.raises(Exception, match="requires eval_every=1"):
        _make_config(
            dataset,
            tmp_path,
            eval_strategy="epoch",
            eval_every=eval_every,
        )


def test_epoch_strategy_accepts_eval_every_one(tmp_path):
    dataset = ControlledDataset([
        [MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0])],
    ])
    cfg = _make_config(
        dataset,
        tmp_path,
        eval_strategy="epoch",
        eval_every=1,
    )
    assert cfg.eval_every == 1


def test_steps_strategy_allows_eval_every_zero(tmp_path):
    # The legacy disable pattern (`eval_strategy=steps` + `eval_every=0`) must
    # remain valid so existing step callers are not regressed.
    dataset = ControlledDataset([
        [MixedRewardGroupBuilder([1.0, 0.0, 1.0, 0.0])],
    ])
    cfg = _make_config(
        dataset,
        tmp_path,
        eval_strategy="steps",
        eval_every=0,
    )
    assert cfg.eval_strategy == "steps"
    assert cfg.eval_every == 0
