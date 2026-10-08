import json
import asyncio

from interactive_training import checkpoint_utils


class _Result:
    def __init__(self, path: str):
        self.path = path


class _Future:
    def __init__(self, path: str):
        self._path = path

    async def result_async(self):
        return _Result(self._path)


class _TrainingClient:
    async def save_state_async(self, name, ttl_seconds=None, step_number=None, metrics=None):
        return _Future(f"service://model_source/weights/{name}")

    async def save_weights_for_sampler_async(self, name, ttl_seconds=None):
        return _Future(name)


def test_save_checkpoint_async_persists_source_step(tmp_path):
    asyncio.run(
        checkpoint_utils.save_checkpoint_async(
            training_client=_TrainingClient(),
            name="final",
            log_path=str(tmp_path),
            loop_state={"epoch": 0, "batch": 4},
            kind="both",
            step_number=3,
        )
    )

    row = json.loads((tmp_path / "checkpoints.jsonl").read_text(encoding="utf-8"))

    assert row["name"] == "final"
    assert row["step"] == 3
    assert row["state_path"] == "model_source/final"
    assert row["sampler_path"] == "final"


def test_checkpoint_path_uses_sdk_supported_public_form():
    assert checkpoint_utils.public_checkpoint_path("service://model_source/weights/final") == "model_source/final"
    assert checkpoint_utils.public_checkpoint_path("session_source/final") == "session_source/final"
