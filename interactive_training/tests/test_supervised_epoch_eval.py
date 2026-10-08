from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest
from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

from interactive_training.eval.evaluators import TrainingClientEvaluator
from interactive_training.supervised import train


class _Dataset:
    def __init__(self, n_batches: int):
        self.n_batches = n_batches
        self.epochs: list[int] = []

    def __len__(self) -> int:
        return self.n_batches

    def set_epoch(self, seed: int = 0) -> None:
        self.epochs.append(seed)

    def get_batch(self, index: int) -> list[SimpleNamespace]:
        del index
        return [
            SimpleNamespace(
                model_input=ModelInput(chunks=[ModelInputChunk(tokens=[1, 2])]),
                loss_fn_inputs={
                    "weights": SimpleNamespace(data=[1, 1]),
                },
            )
        ]


class _DatasetBuilder:
    def __init__(self, dataset: _Dataset):
        self.dataset = dataset

    def build(self):
        return self.dataset, None


class _Future:
    def __init__(self, result, on_result=None):
        self.result = result
        self.on_result = on_result

    async def result_async(self):
        if self.on_result is not None:
            self.on_result()
        return self.result


class _TrainingClient:
    def __init__(self):
        self.completed_steps = 0

    async def save_weights_and_get_sampling_client_async(self):
        return None

    async def forward_backward_async(self, data, loss_fn):
        del data, loss_fn
        return _Future(
            SimpleNamespace(
                loss_fn_outputs=[{"logprobs": SimpleNamespace()}],
            )
        )

    async def optim_step_async(self, adam_params):
        del adam_params

        def complete_step():
            self.completed_steps += 1

        return _Future(SimpleNamespace(metrics={}), on_result=complete_step)


class _ContextTrainingClient(_TrainingClient):
    def __init__(self):
        super().__init__()
        self.current_label: str | None = None
        self.operation_labels: list[tuple[str, str | None]] = []
        self.result_labels: list[tuple[str, str | None]] = []
        self.reset_labels: list[str] = []

    def set_operation_context(self, label: str):
        previous = self.current_label
        self.current_label = label
        return (previous,)

    def reset_operation_context(self, token) -> None:
        assert self.current_label is not None
        self.reset_labels.append(self.current_label)
        self.current_label = token[0]

    async def forward_backward_async(self, data, loss_fn):
        self.operation_labels.append(("forward_backward", self.current_label))
        future = await super().forward_backward_async(data, loss_fn)
        future.on_result = lambda: self.result_labels.append(
            ("forward_backward", self.current_label)
        )
        return future

    async def optim_step_async(self, adam_params):
        self.operation_labels.append(("optim_step", self.current_label))
        future = await super().optim_step_async(adam_params)
        complete_step = future.on_result

        def record_result():
            self.result_labels.append(("optim_step", self.current_label))
            complete_step()

        future.on_result = record_result
        return future


class _Evaluator(TrainingClientEvaluator):
    def __init__(self):
        self.model_versions: list[int] = []

    async def __call__(self, training_client: _TrainingClient) -> dict[str, float]:
        self.model_versions.append(training_client.completed_steps)
        return {"test/loss": float(training_client.completed_steps)}


class _Logger:
    def __init__(self):
        self.rows: list[tuple[int | None, dict]] = []

    def log_metrics(self, metrics, step=None):
        self.rows.append((step, dict(metrics)))

    def close(self):
        pass


@pytest.mark.parametrize("eval_every", [0, 1])
def test_periodic_checkpoint_records_pre_step_batch(monkeypatch, tmp_path, eval_every):
    dataset = _Dataset(n_batches=3)
    evaluator = _Evaluator()
    logger = _Logger()
    saved_checkpoints: list[dict] = []

    monkeypatch.setattr(train.checkpoint_utils, "get_last_checkpoint", lambda _: None)

    async def fake_save_checkpoint_async(**kwargs):
        kwargs["model_version"] = kwargs["training_client"].completed_steps
        saved_checkpoints.append(kwargs)
        return {}

    monkeypatch.setattr(
        train.checkpoint_utils,
        "save_checkpoint_async",
        fake_save_checkpoint_async,
    )
    monkeypatch.setattr(train.ml_log, "setup_logging", lambda **_: logger)
    monkeypatch.setattr(train, "get_tokenizer", lambda _: object())
    monkeypatch.setattr(train, "compute_mean_nll", lambda *_: 0.0)

    config = train.Config(
        log_path=str(tmp_path),
        model_name="test/model",
        dataset_builder=_DatasetBuilder(dataset),
        evaluator_builders=[lambda: evaluator],
        num_epochs=1,
        eval_every=eval_every,
        save_every=1,
        sanitize_logs=True,
    )

    asyncio.run(train.main(config, _TrainingClient()))

    periodic = next(row for row in saved_checkpoints if row["name"] == "1")
    assert periodic["loop_state"] == {"epoch": 0, "batch": 1}
    assert periodic["model_version"] == 1
    if eval_every:
        assert periodic["model_version"] == evaluator.model_versions[1]


def test_epoch_strategy_evaluates_before_training_and_after_each_epoch(
    monkeypatch,
    tmp_path,
):
    dataset = _Dataset(n_batches=2)
    evaluator = _Evaluator()
    logger = _Logger()
    client = _TrainingClient()

    monkeypatch.setattr(train.checkpoint_utils, "get_last_checkpoint", lambda _: None)

    async def fake_save_checkpoint_async(**kwargs):
        del kwargs
        return {}

    monkeypatch.setattr(
        train.checkpoint_utils,
        "save_checkpoint_async",
        fake_save_checkpoint_async,
    )
    monkeypatch.setattr(train.ml_log, "setup_logging", lambda **_: logger)
    monkeypatch.setattr(train, "get_tokenizer", lambda _: object())
    monkeypatch.setattr(train, "compute_mean_nll", lambda *_: 0.0)

    config = train.Config(
        log_path=str(tmp_path),
        model_name="test/model",
        dataset_builder=_DatasetBuilder(dataset),
        evaluator_builders=[lambda: evaluator],
        num_epochs=2,
        eval_strategy="epoch",
        eval_every=1,
        save_every=0,
        sanitize_logs=True,
    )

    asyncio.run(train.main(config, client))

    assert evaluator.model_versions == [0, 2, 4]
    assert dataset.epochs == [0, 1]
    eval_rows = [
        (step, metrics["test/loss"])
        for step, metrics in logger.rows
        if "test/loss" in metrics
    ]
    assert eval_rows == [(0, 0.0), (1, 2.0), (3, 4.0)]


def test_epoch_strategy_without_evaluators_keeps_training_unchanged(
    monkeypatch,
    tmp_path,
):
    dataset = _Dataset(n_batches=2)
    logger = _Logger()
    client = _TrainingClient()

    monkeypatch.setattr(train.checkpoint_utils, "get_last_checkpoint", lambda _: None)

    async def fake_save_checkpoint_async(**kwargs):
        del kwargs
        return {}

    monkeypatch.setattr(
        train.checkpoint_utils,
        "save_checkpoint_async",
        fake_save_checkpoint_async,
    )
    monkeypatch.setattr(train.ml_log, "setup_logging", lambda **_: logger)
    monkeypatch.setattr(train, "get_tokenizer", lambda _: object())
    monkeypatch.setattr(train, "compute_mean_nll", lambda *_: 0.0)

    config = train.Config(
        log_path=str(tmp_path),
        model_name="test/model",
        dataset_builder=_DatasetBuilder(dataset),
        evaluator_builders=[],
        num_epochs=2,
        eval_strategy="epoch",
        eval_every=1,
        save_every=0,
        sanitize_logs=True,
    )

    asyncio.run(train.main(config, client))

    assert client.completed_steps == 4
    assert all("test/loss" not in metrics for _, metrics in logger.rows)


def test_supervised_steps_label_submit_and_retire_operations(
    monkeypatch, tmp_path, caplog
):
    dataset = _Dataset(n_batches=2)
    client = _ContextTrainingClient()

    with caplog.at_level(logging.INFO, logger=train.__name__):
        _run_epoch_sft(
            monkeypatch,
            tmp_path,
            dataset=dataset,
            client=client,
            evaluator=None,
            num_epochs=1,
        )

    assert client.operation_labels == [
        ("forward_backward", "step=0"),
        ("optim_step", "step=0"),
        ("forward_backward", "step=1"),
        ("optim_step", "step=1"),
    ]
    assert client.result_labels == [
        ("forward_backward", "step=0"),
        ("optim_step", "step=0"),
        ("forward_backward", "step=1"),
        ("optim_step", "step=1"),
    ]
    assert client.reset_labels == ["step=0", "step=1", "step=0", "step=1"]
    assert client.current_label is None
    for expected in (
        "[step_timeline] step=0 submit_start",
        "[step_timeline] step=0 submit_done",
        "[step_timeline] step=0 retire_start",
        "[step_timeline] step=0 retire_done",
        "[step_timeline] step=1 submit_start",
        "[step_timeline] step=1 retire_done",
    ):
        assert expected in caplog.text


def test_supervised_retire_context_resets_when_result_raises(
    monkeypatch, tmp_path
):
    class _FailingContextTrainingClient(_ContextTrainingClient):
        async def forward_backward_async(self, data, loss_fn):
            future = await super().forward_backward_async(data, loss_fn)

            async def fail_result():
                future.on_result()
                raise RuntimeError("result failed")

            future.result_async = fail_result
            return future

    dataset = _Dataset(n_batches=1)
    client = _FailingContextTrainingClient()

    with pytest.raises(RuntimeError, match="result failed"):
        _run_epoch_sft(
            monkeypatch,
            tmp_path,
            dataset=dataset,
            client=client,
            evaluator=None,
            num_epochs=1,
        )

    assert client.result_labels == [("forward_backward", "step=0")]
    assert client.reset_labels == ["step=0", "step=0"]
    assert client.current_label is None


def _run_epoch_sft(monkeypatch, tmp_path, *, dataset, client, evaluator, **config_kwargs):
    """Drive supervised.main in epoch mode with the shared mock harness."""
    logger = _Logger()
    monkeypatch.setattr(train.checkpoint_utils, "get_last_checkpoint", lambda _: None)

    async def fake_save_checkpoint_async(**kwargs):
        del kwargs
        return {}

    monkeypatch.setattr(
        train.checkpoint_utils, "save_checkpoint_async", fake_save_checkpoint_async
    )
    monkeypatch.setattr(train.ml_log, "setup_logging", lambda **_: logger)
    monkeypatch.setattr(train, "get_tokenizer", lambda _: object())
    monkeypatch.setattr(train, "compute_mean_nll", lambda *_: 0.0)

    config = train.Config(
        log_path=str(tmp_path),
        model_name="test/model",
        dataset_builder=_DatasetBuilder(dataset),
        evaluator_builders=[lambda: evaluator] if evaluator is not None else [],
        eval_strategy="epoch",
        eval_every=1,
        save_every=0,
        sanitize_logs=True,
        **config_kwargs,
    )
    asyncio.run(train.main(config, client))
    return logger


def test_epoch_strategy_evaluates_final_model_on_midepoch_early_stop(
    monkeypatch,
    tmp_path,
):
    # num_epochs=2, n_batches=2 -> 4 steps total. max_steps=3 stops in the
    # middle of epoch 1 (after step 2, before step 3). The partial final epoch
    # never reaches its boundary eval, so the trained model must still be
    # evaluated once at the last completed step (regression guard: without the
    # final-eval-on-early-stop path, only the untrained baseline would be
    # reported for a single-epoch capped run).
    dataset = _Dataset(n_batches=2)
    evaluator = _Evaluator()
    client = _TrainingClient()

    logger = _run_epoch_sft(
        monkeypatch,
        tmp_path,
        dataset=dataset,
        client=client,
        evaluator=evaluator,
        num_epochs=2,
        max_steps=3,
    )

    eval_rows = [
        (step, metrics["test/loss"])
        for step, metrics in logger.rows
        if "test/loss" in metrics
    ]
    # step 0 baseline (untrained), epoch-0 boundary at step 1, and the final
    # early-stopped model at step 2 (the last trained step of the partial epoch).
    assert eval_rows == [(0, 0.0), (1, 2.0), (2, 3.0)]
    assert evaluator.model_versions == [0, 2, 3]


def test_epoch_strategy_does_not_double_eval_when_stop_lands_on_epoch_boundary(
    monkeypatch,
    tmp_path,
):
    # num_epochs=3, n_batches=2 -> 6 steps. max_steps=4 stops at the very first
    # batch of epoch 2 (step 4), i.e. exactly on the epoch-1 boundary that was
    # already evaluated. The final-eval path must NOT re-evaluate the same step.
    dataset = _Dataset(n_batches=2)
    evaluator = _Evaluator()
    client = _TrainingClient()

    logger = _run_epoch_sft(
        monkeypatch,
        tmp_path,
        dataset=dataset,
        client=client,
        evaluator=evaluator,
        num_epochs=3,
        max_steps=4,
    )

    eval_steps = [step for step, metrics in logger.rows if "test/loss" in metrics]
    # Baseline (0), epoch-0 boundary (1), epoch-1 boundary (3). No duplicate at 3.
    assert eval_steps == [0, 1, 3]
    assert len(eval_steps) == len(set(eval_steps))


@pytest.mark.parametrize("eval_every", [0, 2, 10])
def test_epoch_strategy_rejects_unsupported_eval_every(tmp_path, eval_every):
    # Epoch cadence is explicit: only 1 means evaluate before training and
    # after every completed epoch.
    dataset = _Dataset(n_batches=2)
    with pytest.raises(Exception, match="requires eval_every=1"):
        train.Config(
            log_path=str(tmp_path),
            model_name="test/model",
            dataset_builder=_DatasetBuilder(dataset),
            eval_strategy="epoch",
            eval_every=eval_every,
        )


def test_epoch_strategy_accepts_eval_every_one(tmp_path):
    dataset = _Dataset(n_batches=2)
    config = train.Config(
        log_path=str(tmp_path),
        model_name="test/model",
        dataset_builder=_DatasetBuilder(dataset),
        eval_strategy="epoch",
        eval_every=1,
    )
    assert config.eval_every == 1


def test_steps_strategy_allows_eval_every_zero(tmp_path):
    # The legacy disable pattern (`eval_strategy=steps` + `eval_every=0`) must
    # remain valid so existing step callers are not regressed.
    dataset = _Dataset(n_batches=2)
    config = train.Config(
        log_path=str(tmp_path),
        model_name="test/model",
        dataset_builder=_DatasetBuilder(dataset),
        eval_strategy="steps",
        eval_every=0,
    )
    assert config.eval_strategy == "steps"
    assert config.eval_every == 0


def _run_steps_sft_capturing_saves(monkeypatch, tmp_path, *, pipeline_depth):
    """Drive supervised.main in `steps` mode and capture eval + save behavior.

    Returns (evaluator.model_versions, client.completed_steps, saved checkpoints).
    `model_versions` records how many optim steps had SETTLED at each eval, so
    it directly reflects the weights eval observed; saved checkpoints include
    each label and the next loop position used by resume.
    """
    dataset = _Dataset(n_batches=6)
    evaluator = _Evaluator()
    client = _TrainingClient()
    logger = _Logger()
    saved_names: list[str] = []
    saved_loop_states: list[dict[str, int]] = []

    monkeypatch.setattr(train.checkpoint_utils, "get_last_checkpoint", lambda _: None)

    async def fake_save_checkpoint_async(**kwargs):
        saved_names.append(kwargs["name"])
        saved_loop_states.append(kwargs["loop_state"])
        return {"state_path": f"model_test/{kwargs['name']}"}

    monkeypatch.setattr(
        train.checkpoint_utils, "save_checkpoint_async", fake_save_checkpoint_async
    )
    monkeypatch.setattr(train.ml_log, "setup_logging", lambda **_: logger)
    monkeypatch.setattr(train, "get_tokenizer", lambda _: object())
    monkeypatch.setattr(train, "compute_mean_nll", lambda *_: 0.0)

    config = train.Config(
        log_path=str(tmp_path),
        model_name="test/model",
        dataset_builder=_DatasetBuilder(dataset),
        evaluator_builders=[lambda: evaluator],
        num_epochs=1,
        eval_strategy="steps",
        eval_every=2,
        save_every=2,
        pipeline_depth=pipeline_depth,
        sanitize_logs=True,
    )
    asyncio.run(train.main(config, client))
    return (
        evaluator.model_versions,
        client.completed_steps,
        saved_names,
        saved_loop_states,
    )


def test_pipeline_depth_is_transparent_to_eval_and_save(monkeypatch, tmp_path):
    # Deep pipelining must not change WHAT eval sees or WHICH step is
    # checkpointed: the loop drains the in-flight window at each eval/save
    # boundary, so depth=3 is observationally identical to depth=1.
    versions_d1, completed_d1, saves_d1, loop_states_d1 = (
        _run_steps_sft_capturing_saves(
            monkeypatch, tmp_path / "d1", pipeline_depth=1
        )
    )
    versions_d3, completed_d3, saves_d3, loop_states_d3 = (
        _run_steps_sft_capturing_saves(
            monkeypatch, tmp_path / "d3", pipeline_depth=3
        )
    )

    # 6 steps (0..5), eval before steps 0/2/4, plus a final-model eval after the
    # loop (step 6). Each eval sees exactly `step` settled optim steps because
    # the pipeline is flushed first.
    assert versions_d1 == [0, 2, 4, 6]
    # Periodic saves capture the pre-step weights at steps 2 and 4, so resume
    # repeats the labeled batch; the final checkpoint advances past the run.
    assert saves_d1 == ["2", "4", "final"]
    assert loop_states_d1 == [
        {"epoch": 0, "batch": 2},
        {"epoch": 0, "batch": 4},
        {"epoch": 1, "batch": 0},
    ]
    assert completed_d1 == 6
    checkpointed_evals_d1 = [
        (record["step"], record["checkpoint_path"])
        for record in map(
            json.loads,
            (tmp_path / "d1" / "checkpoint_evaluations.jsonl").read_text().splitlines(),
        )
    ]
    assert checkpointed_evals_d1 == [
        (2, "model_test/2"),
        (4, "model_test/4"),
        (5, "model_test/final"),
    ]

    # Depth-3 is byte-for-byte identical on all three observables.
    assert versions_d3 == versions_d1
    assert saves_d3 == saves_d1
    assert loop_states_d3 == loop_states_d1
    assert completed_d3 == completed_d1
    checkpointed_evals_d3 = [
        (record["step"], record["checkpoint_path"])
        for record in map(
            json.loads,
            (tmp_path / "d3" / "checkpoint_evaluations.jsonl").read_text().splitlines(),
        )
    ]
    assert checkpointed_evals_d3 == checkpointed_evals_d1
