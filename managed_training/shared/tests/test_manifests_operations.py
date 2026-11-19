from __future__ import annotations

import errno
import json
import os
from types import SimpleNamespace

import pytest

from shared.experiment_manifest import ExperimentManifest, RuntimeManifest
from shared.foundry_operations import (
    create_or_reuse_fine_tuning_job,
    upload_or_reuse_file,
)

SFT_METHOD = {"type": "supervised", "supervised": {"hyperparameters": {"n_epochs": 2}}}


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

    def retrieve(self, file_id, *, timeout):
        assert timeout > 0
        return next(item for item in self.data if item.id == file_id)


class Jobs:
    def __init__(self) -> None:
        self.data = []

    def list(self):
        return SimpleNamespace(data=self.data)

    def create(self, **payload):
        extra_body = payload.pop("extra_body", {})
        job = SimpleNamespace(
            id="ftjob-created", status="queued",
            **{"suffix": None, "validation_file": None, "hyperparameters": None, **payload, **extra_body},
        )
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
        method=SFT_METHOD,
    )
    second = create_or_reuse_fine_tuning_job(
        client,
        model="base-model",
        training_file_id="file-train",
        validation_file_id="file-valid",
        suffix="sample-sft",
        method=SFT_METHOD,
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
            method=SFT_METHOD,
        )
    )
    result = create_or_reuse_fine_tuning_job(
        SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs)),
        model="base-model",
        training_file_id="file-train",
        suffix="sample-sft",
        method=SFT_METHOD,
    )
    assert result.reused
    assert result.resource_id == "ftjob-pending"
    assert len(jobs.data) == 1


def test_training_type_is_sent_and_matching_job_is_reused() -> None:
    jobs = Jobs()
    client = SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs))
    first = create_or_reuse_fine_tuning_job(
        client, model="base-model", training_file_id="file-train",
        training_type="globalStandard", method=SFT_METHOD,
    )
    assert jobs.data[0].trainingType == "globalStandard"
    second = create_or_reuse_fine_tuning_job(
        client, model="base-model", training_file_id="file-train",
        training_type="GlobalStandard", method=SFT_METHOD,
    )
    assert second.reused
    assert first.resource_id == second.resource_id
    assert len(jobs.data) == 1


@pytest.mark.parametrize("actual", [None, "standard", "developer"])
def test_training_type_mismatch_is_not_reused(actual) -> None:
    jobs = Jobs()
    job = SimpleNamespace(
        id="job-recorded", status="pending", model="base-model",
        training_file="file-train", validation_file=None, trainingType=actual,
    )
    jobs.data.append(job)
    jobs.retrieve = lambda _job_id: job
    client = SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs))
    with pytest.raises(RuntimeError, match="trainingType"):
        create_or_reuse_fine_tuning_job(
            client, model="base-model", training_file_id="file-train",
            training_type="globalStandard", existing_job_id=job.id,
        )
    result = create_or_reuse_fine_tuning_job(
        client, model="base-model", training_file_id="file-train",
        training_type="globalStandard",
    )
    assert not result.reused
    assert len(jobs.data) == 2


@pytest.mark.parametrize("status", ["pending", "validating_files", "queued", "running", "succeeded"])
def test_explicit_job_id_recovers_matching_recipe(status) -> None:
    jobs = Jobs()
    method = {
        "type": "reinforcement",
        "reinforcement": {
            "grader": {
                "type": "score_model",
                "input": [{"role": "user", "content": "{{sample.output_text}}", "type": "message"}],
            },
            "hyperparameters": {"n_epochs": 1},
        },
    }
    actual = {
        "type": "reinforcement",
        "reinforcement": {
            "grader": {
                "type": "score_model",
                "input": [{"role": "user", "content": "{{sample.output_text}}", "type": "message"}],
            },
            "hyperparameters": {"n_epochs": 1},
        },
    }
    job = SimpleNamespace(
        id="job-recorded", status=status, model="base-model",
        training_file="file-train", validation_file="file-valid",
        method=SimpleNamespace(model_dump=lambda **kwargs: actual),
        metadata={"cookbook_suffix": "sample-rft"},
    )
    jobs.data.append(job)
    jobs.retrieve = lambda job_id: job if job_id == job.id else None
    client = SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs))

    result = create_or_reuse_fine_tuning_job(
        client, model="base-model", training_file_id="file-train",
        validation_file_id="file-valid", method=method, suffix="sample-rft",
        existing_job_id="job-recorded",
    )

    assert result.reused
    assert result.resource_id == job.id
    assert len(jobs.data) == 1


@pytest.mark.parametrize("changed", [
    {"model": "other-model"},
    {"training_file": "other-training"},
    {"validation_file": "other-validation"},
    {"method": {"type": "supervised"}},
    {"method": {"type": "reinforcement", "reinforcement": {"grader": {"name": "other"}}}},
    {"suffix": "other-suffix"},
    {"status": "failed"},
    {"status": "cancelled"},
])
def test_explicit_job_id_never_replaces_mismatched_or_terminal_job(changed) -> None:
    jobs = Jobs()
    method = {"type": "reinforcement", "reinforcement": {"grader": {"name": "supplied"}}}
    job = SimpleNamespace(
        id="job-recorded", status="pending", model="base-model",
        training_file="file-train", validation_file="file-valid",
        method=method, suffix="sample-rft",
    )
    for name, value in changed.items():
        setattr(job, name, value)
    jobs.data.append(job)
    jobs.retrieve = lambda _job_id: job
    with pytest.raises(RuntimeError, match="does not match|cannot be reused"):
        create_or_reuse_fine_tuning_job(
            SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs)),
            model="base-model", training_file_id="file-train",
            validation_file_id="file-valid", method=method, suffix="sample-rft",
            existing_job_id=job.id,
        )
    assert len(jobs.data) == 1


def test_job_reuse_supports_native_responses_without_suffix_field() -> None:
    jobs = Jobs()
    client = SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs))
    first = create_or_reuse_fine_tuning_job(
        client,
        model="base-model",
        training_file_id="file-train",
        suffix="sample-sft",
        method=SFT_METHOD,
    )
    del jobs.data[0].suffix
    second = create_or_reuse_fine_tuning_job(
        client,
        model="base-model",
        training_file_id="file-train",
        suffix="sample-sft",
        method=SFT_METHOD,
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
    # Upload and the subsequent processing poll each disable SDK retries.
    assert options == [{"max_retries": 0}, {"max_retries": 0}]
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


@pytest.mark.parametrize(
    "method",
    [
        {"type": "reinforcement", "reinforcement": {"hyperparameters": {"n_epochs": 2}}},
        {"type": "dpo", "dpo": {"hyperparameters": {"n_epochs": 2}}},
        SFT_METHOD,
        None,
    ],
)
@pytest.mark.parametrize("requested", [{}, {"hyperparameters": {"n_epochs": 2}}])
def test_omitted_method_never_reuses_other_or_uncertain_default_recipe(method, requested) -> None:
    jobs = Jobs()
    jobs.data.append(SimpleNamespace(
        id="ftjob-other", status="queued", model="base-model",
        training_file="file-train", validation_file=None, suffix=None,
        method=method, hyperparameters={"n_epochs": 2},
    ))
    result = create_or_reuse_fine_tuning_job(
        SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs)),
        model="base-model", training_file_id="file-train", **requested,
    )
    assert not result.reused
    assert len(jobs.data) == 2
    assert not hasattr(jobs.data[-1], "method")  # Preserve the default request shape.


@pytest.mark.parametrize(
    "actual,requested",
    [
        ({"suffix": "custom"}, {}),
        ({"hyperparameters": {"n_epochs": 7}}, {}),
        ({"method": {"type": "reinforcement"}}, {}),
        ({"method": {"type": "dpo"}}, {}),
        ({"method": {"type": "supervised", "supervised": {
            "hyperparameters": {"n_epochs": 2, "batch_size": 8},
        }}}, {}),
        ({"hyperparameters": {"n_epochs": 2, "batch_size": 8}},
         {"hyperparameters": {"n_epochs": 2}}),
    ],
)
def test_job_reuse_does_not_treat_omitted_settings_as_wildcards(actual, requested) -> None:
    jobs = Jobs()
    jobs.data.append(SimpleNamespace(**{
        "id": "ftjob-other", "status": "queued", "model": "base-model",
        "training_file": "file-train", "validation_file": None, "suffix": None,
        "method": SFT_METHOD, "hyperparameters": None, **actual,
    }))
    result = create_or_reuse_fine_tuning_job(
        SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs)),
        model="base-model", training_file_id="file-train", method=SFT_METHOD, **requested,
    )
    assert not result.reused
    assert len(jobs.data) == 2


@pytest.mark.parametrize("hyperparameters", [None, {"n_epochs": 2}])
def test_exact_explicit_recipe_can_reuse_an_observed_default_suffix(hyperparameters) -> None:
    jobs = Jobs()
    jobs.data.append(SimpleNamespace(
        id="ftjob-exact", status="queued", model="base-model",
        training_file="file-train", validation_file=None, suffix=None,
        method=SFT_METHOD, hyperparameters=hyperparameters,
    ))
    result = create_or_reuse_fine_tuning_job(
        SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs)),
        model="base-model", training_file_id="file-train",
        method=SFT_METHOD, hyperparameters=hyperparameters,
    )
    assert result.reused
    assert result.resource_id == "ftjob-exact"
    assert len(jobs.data) == 1


def test_unobservable_default_suffix_conservatively_prevents_reuse() -> None:
    jobs = Jobs()
    jobs.data.append(SimpleNamespace(
        id="ftjob-unknown", status="queued", model="base-model",
        training_file="file-train", validation_file=None, method=SFT_METHOD,
    ))
    result = create_or_reuse_fine_tuning_job(
        SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs)),
        model="base-model", training_file_id="file-train", method=SFT_METHOD,
    )
    assert not result.reused


def test_materialized_default_hyperparameters_are_not_assumed_equivalent() -> None:
    jobs = Jobs()
    jobs.data.append(SimpleNamespace(
        id="ftjob-defaults", status="queued", model="base-model",
        training_file="file-train", validation_file=None, suffix=None,
        method={"type": "supervised"}, hyperparameters=None,
    ))
    result = create_or_reuse_fine_tuning_job(
        SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs)),
        model="base-model", training_file_id="file-train", method={"type": "supervised"},
    )
    assert not result.reused


def test_native_model_dump_recipe_can_be_reused_without_unknown_defaults() -> None:
    class NativeMethod:
        def model_dump(self, *, exclude_none):
            assert exclude_none
            return SFT_METHOD

    jobs = Jobs()
    jobs.data.append(SimpleNamespace(
        id="ftjob-native", status="queued", model="base-model",
        training_file="file-train", validation_file=None, suffix=None,
        method=NativeMethod(), hyperparameters=None,
    ))
    result = create_or_reuse_fine_tuning_job(
        SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs)),
        model="base-model", training_file_id="file-train", method=SFT_METHOD,
    )
    assert result.reused
    assert result.resource_id == "ftjob-native"


@pytest.mark.parametrize("explicit_root", [False, True])
@pytest.mark.parametrize("tail", ["../runtime.json", "../other/outputs/runtime.json"])
def test_runtime_manifest_rejects_traversal_out_of_intended_outputs(tmp_path, explicit_root, tail) -> None:
    root = tmp_path / "outputs"
    options = {"outputs_root": root} if explicit_root else {}
    with pytest.raises(ValueError, match="outputs"):
        RuntimeManifest(demo_id="sample").write(root / tail, **options)
    assert not root.exists()
    assert not (tmp_path / "runtime.json").exists()
    assert not (tmp_path / "other").exists()


def _symlink(link, target, *, directory: bool) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except NotImplementedError:
        pytest.skip("Symlink creation is not implemented on this platform")
    except OSError as error:
        if error.errno in {errno.ENOSYS, errno.EOPNOTSUPP} or (
            os.name == "nt" and getattr(error, "winerror", None) in {1, 50, 1314}
        ):
            pytest.skip("Symlink creation is unsupported or lacks the Windows symlink privilege")
        raise


@pytest.mark.parametrize("explicit_root", [False, True])
@pytest.mark.parametrize("redirect", ["root", "directory", "file"])
def test_runtime_manifest_rejects_symlink_escape(tmp_path, explicit_root, redirect) -> None:
    root = tmp_path / "outputs"
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "runtime.json"
    if redirect == "root":
        _symlink(root, outside, directory=True)
        destination = root / "runtime.json"
    else:
        root.mkdir()
        if redirect == "directory":
            _symlink(root / "link", outside, directory=True)
            destination = root / "link" / "runtime.json"
        else:
            target.write_text("unchanged", encoding="utf-8")
            _symlink(root / "runtime.json", target, directory=False)
            destination = root / "runtime.json"
    before = target.read_bytes() if target.exists() else None
    options = {"outputs_root": root} if explicit_root else {}
    with pytest.raises(ValueError, match="outputs"):
        RuntimeManifest(demo_id="sample").write(destination, **options)
    assert (target.read_bytes() if target.exists() else None) == before


def test_runtime_manifest_explicit_root_and_nested_output_compatibility(tmp_path) -> None:
    root = tmp_path / "outputs"
    runtime = RuntimeManifest(demo_id="sample")
    path = runtime.write(root / "run" / "runtime.json", outputs_root=root)
    assert json.loads(path.read_text(encoding="utf-8"))["demo_id"] == "sample"
    with pytest.raises(ValueError, match="outputs"):
        runtime.write(tmp_path / "another" / "outputs" / "runtime.json", outputs_root=root)


def test_relative_runtime_path_cannot_normalize_away_outputs_before_check(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runtime = RuntimeManifest(demo_id="sample")
    with pytest.raises(ValueError, match="outputs"):
        runtime.write("outputs/../other/outputs/runtime.json")
    assert not (tmp_path / "other").exists()
    assert runtime.write("outputs/run/runtime.json").is_file()


def test_multiple_outputs_ancestors_require_the_explicit_intended_root(tmp_path) -> None:
    root = tmp_path / "outputs" / "demo" / "outputs"
    destination = root / "runtime.json"
    runtime = RuntimeManifest(demo_id="sample")
    with pytest.raises(ValueError, match="unambiguous"):
        runtime.write(destination)
    assert not root.exists()
    assert runtime.write(destination, outputs_root=root).is_file()


def test_runtime_manifest_allows_symlink_target_that_remains_within_root(tmp_path) -> None:
    root = tmp_path / "outputs"
    target = root / "runs"
    target.mkdir(parents=True)
    _symlink(root / "alias", target, directory=True)
    result = RuntimeManifest(demo_id="sample").write(
        root / "alias" / "runtime.json", outputs_root=root,
    )
    assert result == target / "runtime.json"
    assert result.is_file()
