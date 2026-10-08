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


def test_pending_job_is_reused() -> None:
    jobs = Jobs()
    jobs.data.append(
        SimpleNamespace(
            id="ftjob-pending",
            status="pending",
            model="base-model",
            training_file="file-train",
            validation_file=None,
            suffix="sample-sft",
        )
    )
    result = create_or_reuse_fine_tuning_job(
        SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs)),
        model="base-model",
        training_file_id="file-train",
        suffix="sample-sft",
    )
    assert result.reused
    assert result.resource_id == "ftjob-pending"
    assert len(jobs.data) == 1


def test_job_reuse_supports_native_responses_without_suffix_field() -> None:
    jobs = Jobs()
    client = SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs))
    first = create_or_reuse_fine_tuning_job(
        client,
        model="base-model",
        training_file_id="file-train",
        suffix="sample-sft",
    )
    del jobs.data[0].suffix
    second = create_or_reuse_fine_tuning_job(
        client,
        model="base-model",
        training_file_id="file-train",
        suffix="sample-sft",
    )
    assert jobs.data[0].metadata == {"cookbook_suffix": "sample-sft"}
    assert second.reused
    assert second.resource_id == first.resource_id
    assert len(jobs.data) == 1


@pytest.mark.parametrize(
    "recipe",
    [
        {"hyperparameters": {"n_epochs": 2}},
        {"method": {"type": "supervised", "supervised": {"hyperparameters": {"n_epochs": 2}}}},
    ],
)
def test_job_reuse_requires_matching_recipe(recipe) -> None:
    jobs = Jobs()
    client = SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs))
    first = create_or_reuse_fine_tuning_job(
        client, model="base-model", training_file_id="file-train", **recipe
    )
    changed_recipe = {
        "hyperparameters": {"n_epochs": 3}
    } if "hyperparameters" in recipe else {
        "method": {"type": "supervised", "supervised": {"hyperparameters": {"n_epochs": 3}}}
    }
    second = create_or_reuse_fine_tuning_job(
        client, model="base-model", training_file_id="file-train", **changed_recipe
    )
    assert not first.reused
    assert not second.reused
    assert len(jobs.data) == 2


def test_job_creation_is_not_blindly_retried() -> None:
    attempts = 0

    class AmbiguousServiceError(RuntimeError):
        status_code = 503

    def create(**payload):
        nonlocal attempts
        attempts += 1
        raise AmbiguousServiceError("Response lost after accepting request")

    client = SimpleNamespace(
        fine_tuning=SimpleNamespace(jobs=SimpleNamespace(create=create))
    )
    with pytest.raises(AmbiguousServiceError):
        create_or_reuse_fine_tuning_job(
            client, model="base-model", training_file_id="file-train", reuse=False
        )
    assert attempts == 1


def test_job_creation_disables_sdk_automatic_retries() -> None:
    options = []
    jobs = Jobs()
    submission_client = SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs))

    def with_options(**kwargs):
        options.append(kwargs)
        return submission_client

    client = SimpleNamespace(with_options=with_options)
    create_or_reuse_fine_tuning_job(
        client, model="base-model", training_file_id="file-train", reuse=False
    )
    assert options == [{"max_retries": 0}]
    assert len(jobs.data) == 1


def test_file_reuse_searches_all_sdk_pages(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text('{"messages": []}\n', encoding="utf-8")
    files = Files()
    client = SimpleNamespace(files=files)
    first = upload_or_reuse_file(
        client, demo_slug="sample-demo", purpose_name="training", path=path
    )
    matching_file = files.data[0]

    class PaginatedFiles:
        data = []

        def has_next_page(self):
            return True

        def __iter__(self):
            return iter([matching_file])

    files.list = lambda: PaginatedFiles()
    second = upload_or_reuse_file(
        client, demo_slug="sample-demo", purpose_name="training", path=path
    )
    assert second.reused
    assert second.resource_id == first.resource_id
    assert files.created == 1


def test_file_creation_disables_sdk_automatic_retries(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text('{"messages": []}\n', encoding="utf-8")
    options = []
    files = Files()
    submission_client = SimpleNamespace(files=files)

    def with_options(**kwargs):
        options.append(kwargs)
        return submission_client

    client = SimpleNamespace(files=files, with_options=with_options)
    upload_or_reuse_file(
        client, demo_slug="sample-demo", purpose_name="training", path=path, reuse=False
    )
    assert options == [{"max_retries": 0}]
    assert files.created == 1


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
