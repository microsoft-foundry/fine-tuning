from __future__ import annotations

from types import SimpleNamespace

import pytest

from shared.experiment_manifest import ExperimentManifest, RuntimeManifest
from shared.foundry_operations import (
    create_or_reuse_fine_tuning_job,
    upload_or_reuse_file,
)


class Files:
    def __init__(self) -> None:
        self.data = []
        self.created = 0

    def list(self):
        return SimpleNamespace(data=self.data)

    def create(self, *, file, purpose):
        self.created += 1
        name, stream, _ = file
        assert stream.read()
        item = SimpleNamespace(
            id=f"file-{self.created}",
            filename=name,
            purpose=purpose,
            status="processed",
        )
        self.data.append(item)
        return item

    def retrieve(self, file_id):
        return next(item for item in self.data if item.id == file_id)


class Jobs:
    def __init__(self) -> None:
        self.data = []

    def list(self):
        return SimpleNamespace(data=self.data)

    def create(self, **payload):
        job = SimpleNamespace(id="ftjob-created", status="queued", **payload)
        self.data.append(job)
        return job


def test_upload_reuses_content_addressed_name(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text('{"messages": []}\n', encoding="utf-8")
    client = SimpleNamespace(files=Files())
    first = upload_or_reuse_file(
        client,
        demo_slug="sample-demo",
        purpose_name="training",
        path=path,
    )
    second = upload_or_reuse_file(
        client,
        demo_slug="sample-demo",
        purpose_name="training",
        path=path,
    )
    assert first.reused is False
    assert second.reused is True
    assert first.resource_id == second.resource_id


def test_job_reuse_requires_matching_inputs() -> None:
    jobs = Jobs()
    client = SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs))
    first = create_or_reuse_fine_tuning_job(
        client,
        model="base-model",
        training_file_id="file-train",
        validation_file_id="file-valid",
        suffix="sample-sft",
    )
    second = create_or_reuse_fine_tuning_job(
        client,
        model="base-model",
        training_file_id="file-train",
        validation_file_id="file-valid",
        suffix="sample-sft",
    )
    assert first.reused is False
    assert second.reused is True


def test_manifest_output_boundaries(tmp_path) -> None:
    experiment = ExperimentManifest(
        demo_id="sample-demo",
        hypothesis="Fine-tuning improves exact-match accuracy.",
        base_model="runtime-configured",
        dataset_hashes={"training": "a" * 64},
    )
    assert experiment.write(tmp_path / "experiment.json").is_file()

    runtime = RuntimeManifest(
        demo_id="sample-demo",
        operation_ids={"job": "ftjob-private"},
    )
    with pytest.raises(ValueError, match="outputs"):
        runtime.write(tmp_path / "runtime.json")
    assert runtime.write(tmp_path / "outputs" / "runtime.json").is_file()
