from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from shared.dataset_validation import (
    DatasetValidationError,
    hash_file,
    validate_jsonl,
    validate_split_isolation,
)
from shared.validate_dataset import (
    build_manifest,
    canonical_upload_bytes,
    main as validate_dataset_main,
)


def _write(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(prompt: str, answer: str = "ok") -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    }


def test_validates_and_hashes_jsonl(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    _write(path, [_row("one"), _row("two"), _row("two")])
    report = validate_jsonl(path)
    assert report.valid
    assert report.rows == 3
    assert report.duplicate_rows == 1
    assert report.sha256 == hash_file(path)


def test_detects_schema_and_secret_errors(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    _write(
        path,
        [
            {"messages": [{"role": "assistant", "content": "first"}]},
            _row("api_key=abcdefghijklmnop"),
        ],
    )
    report = validate_jsonl(path)
    assert not report.valid
    with pytest.raises(DatasetValidationError):
        report.require_valid()


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("location", ["extra", "nested-extra", "message-extra", "part-extra"])
@pytest.mark.parametrize("validate_messages", [False, True])
def test_rejects_nonstandard_json_constants_anywhere_in_the_record(
    tmp_path, constant, location, validate_messages,
) -> None:
    row = _row("prompt")
    marker = "NON_FINITE_VALUE"
    if location == "extra":
        row["extra"] = marker
    elif location == "nested-extra":
        row["extra"] = {"nested": [{"score": marker}]}
    elif location == "message-extra":
        row["messages"][1]["weight"] = marker
    else:
        row["messages"][0]["content"] = [{
            "type": "text", "text": "prompt", "extra": {"unused": [marker]},
        }]
    path = tmp_path / "nonstandard.jsonl"
    path.write_text(json.dumps(row).replace(f'"{marker}"', constant) + "\n", encoding="utf-8")
    validated_rows = []

    def row_validator(record, line_number):
        validated_rows.append(record)
        return []

    report = validate_jsonl(path, validate_messages=validate_messages, row_validator=row_validator)
    assert not report.valid
    assert report.rows == 1
    assert report.unique_rows == 0
    assert len(report.issues) == 1
    assert report.issues[0].code == "json.invalid"
    assert report.issues[0].line_number == 1
    assert constant in report.issues[0].message
    assert validated_rows == []
    with pytest.raises(DatasetValidationError, match="not valid JSON"):
        report.require_valid()


def test_json_constant_names_are_allowed_as_strings_with_other_valid_json_scalars(tmp_path) -> None:
    path = tmp_path / "valid-scalars.jsonl"
    row = _row("NaN, Infinity, and -Infinity are strings here")
    row["extra"] = ["NaN", "Infinity", "-Infinity", None, True, False, 0, -1, 1.25]
    _write(path, [row])
    assert validate_jsonl(path).require_valid().valid


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_split_and_manifest_validation_reject_nonfinite_extras_without_writes(tmp_path, constant) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    row = _row("train")
    row["extra"] = "NON_FINITE_VALUE"
    train.write_text(
        json.dumps(row).replace('"NON_FINITE_VALUE"', constant) + "\n", encoding="utf-8",
    )
    _write(validation, [_row("validation")])
    before = {path: path.read_bytes() for path in (train, validation)}
    with pytest.raises(DatasetValidationError, match="not valid JSON"):
        validate_split_isolation({"train": train, "validation": validation})
    prepared = tmp_path / "prepared" / "nested"
    with pytest.raises(DatasetValidationError, match="not valid JSON"):
        build_manifest(train, validation, canonical_output_dir=prepared)
    assert {path: path.read_bytes() for path in before} == before
    assert not prepared.parent.exists()


def test_rejects_split_overlap(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    duplicate = _row("shared")
    _write(train, [duplicate, _row("train-only")])
    _write(validation, [duplicate, _row("validation-only")])
    with pytest.raises(DatasetValidationError, match="contamination"):
        validate_split_isolation({"train": train, "validation": validation})


def test_builds_versioned_adaptation_manifest(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write(train, [_row("train-one"), _row("train-two")])
    _write(validation, [_row("validation-one")])

    manifest = build_manifest(train, validation)

    assert manifest["schema_version"] == "1.0"
    assert manifest["splits"]["train"]["rows"] == 2
    assert manifest["splits"]["validation"]["rows"] == 1
    assert manifest["isolation"]["overlaps"] == []


def test_manifest_upload_evidence_normalizes_crlf(tmp_path) -> None:
    lf_train = tmp_path / "lf-train.jsonl"
    crlf_train = tmp_path / "crlf-train.jsonl"
    validation = tmp_path / "validation.jsonl"
    lf_content = "".join(
        json.dumps(row) + "\n"
        for row in (_row("train-one"), _row("train-two"))
    ).encode("utf-8")
    lf_train.write_bytes(lf_content)
    crlf_train.write_bytes(lf_content.replace(b"\n", b"\r\n"))
    _write(validation, [_row("validation-one")])

    lf_manifest = build_manifest(lf_train, validation)
    prepared = tmp_path / "prepared"
    crlf_manifest = build_manifest(
        crlf_train,
        validation,
        canonical_output_dir=prepared,
    )

    assert (
        lf_manifest["splits"]["train"]["upload_sha256"]
        == crlf_manifest["splits"]["train"]["upload_sha256"]
    )
    assert (
        lf_manifest["splits"]["train"]["upload_bytes"]
        == crlf_manifest["splits"]["train"]["upload_bytes"]
    )
    assert (
        crlf_manifest["splits"]["train"]["source_sha256"]
        != crlf_manifest["splits"]["train"]["upload_sha256"]
    )
    assert (prepared / "train.jsonl").read_bytes() == canonical_upload_bytes(crlf_train)


def test_canonical_output_cannot_overwrite_source(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write(train, [_row("train")])
    _write(validation, [_row("validation")])

    with pytest.raises(ValueError, match="must not overwrite"):
        build_manifest(train, validation, canonical_output_dir=tmp_path)


@pytest.mark.parametrize("input_name", ["train", "validation"])
@pytest.mark.parametrize("output_name", ["train", "validation"])
@pytest.mark.parametrize("parent_alias", [False, True])
def test_canonical_outputs_reject_all_input_collisions_before_writing(
    tmp_path, input_name, output_name, parent_alias,
) -> None:
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "prepared"
    inputs.mkdir()
    outputs.mkdir()
    sources = {name: inputs / f"source-{name}.jsonl" for name in ("train", "validation")}
    collision = outputs / f"{output_name}.jsonl"
    if parent_alias:
        (outputs / "nested").mkdir()
        collision = outputs / "nested" / ".." / collision.name
    sources[input_name] = collision
    for name, path in sources.items():
        path.write_bytes((json.dumps(_row(name)) + "\r\n").encode("utf-8"))
    other_name = "validation" if output_name == "train" else "train"
    (outputs / f"{other_name}.jsonl").write_bytes(b"Retain the existing output.\r\n")
    watched = {*sources.values(), outputs / "train.jsonl", outputs / "validation.jsonl"}
    before = {path: path.read_bytes() for path in watched}
    entries = set(outputs.iterdir())

    with pytest.raises(ValueError, match="must not overwrite"):
        build_manifest(sources["train"], sources["validation"], canonical_output_dir=outputs)

    assert {path: path.read_bytes() for path in watched} == before
    assert set(outputs.iterdir()) == entries


def _dataset_alias(link, target, *, kind) -> None:
    try:
        if kind == "hardlink":
            link.hardlink_to(target)
        else:
            link.symlink_to(target, target_is_directory=kind == "directory")
    except NotImplementedError:
        pytest.skip("Filesystem aliases are not implemented on this platform")
    except OSError as error:
        if error.errno in {errno.ENOSYS, errno.EOPNOTSUPP} or (
            os.name == "nt" and getattr(error, "winerror", None) in {1, 50, 1314}
        ):
            pytest.skip("Filesystem aliases are unsupported or lack the Windows symlink privilege")
        raise


@pytest.mark.parametrize("input_name,output_name", [("validation", "train"), ("train", "validation")])
@pytest.mark.parametrize("kind", ["file", "directory", "hardlink"])
def test_canonical_outputs_reject_filesystem_alias_collisions_before_writing(
    tmp_path, input_name, output_name, kind,
) -> None:
    inputs = tmp_path / "inputs"
    prepared = tmp_path / "prepared"
    inputs.mkdir()
    prepared.mkdir()
    sources = {name: inputs / f"source-{name}.jsonl" for name in ("train", "validation")}
    outputs = prepared
    if kind == "directory":
        sources[input_name] = prepared / f"{output_name}.jsonl"
        outputs = tmp_path / "prepared-alias"
        _dataset_alias(outputs, prepared, kind=kind)
    for name, path in sources.items():
        path.write_bytes((json.dumps(_row(name)) + "\r\n").encode("utf-8"))
    if kind != "directory":
        _dataset_alias(outputs / f"{output_name}.jsonl", sources[input_name], kind=kind)
    other_name = "validation" if output_name == "train" else "train"
    (outputs / f"{other_name}.jsonl").write_bytes(b"Retain the existing output.\r\n")
    watched = {*sources.values(), outputs / "train.jsonl", outputs / "validation.jsonl"}
    before = {path: path.read_bytes() for path in watched}
    entries = set(prepared.iterdir())

    with pytest.raises(ValueError, match="must not overwrite"):
        build_manifest(sources["train"], sources["validation"], canonical_output_dir=outputs)

    assert {path: path.read_bytes() for path in watched} == before
    assert set(prepared.iterdir()) == entries


def _filesystem_snapshot(root):
    entries = set(root.rglob("*"))
    return entries, {path: path.read_bytes() for path in entries if path.is_file()}


@pytest.mark.parametrize("input_name", ["train", "validation"])
@pytest.mark.parametrize("kind", ["same", "relative", "file", "directory", "hardlink"])
@pytest.mark.parametrize("canonicalize", [False, True])
def test_cli_manifest_cannot_alias_either_source_before_any_writes(
    tmp_path, monkeypatch, input_name, kind, canonicalize,
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    sources = {name: inputs / f"source-{name}.jsonl" for name in ("train", "validation")}
    for name, path in sources.items():
        path.write_bytes((json.dumps(_row(name)) + "\r\n").encode("utf-8"))
    target = sources[input_name]
    if kind == "same":
        output = target
    elif kind == "relative":
        monkeypatch.chdir(tmp_path)
        output = Path("inputs") / target.name
    elif kind == "directory":
        alias = tmp_path / "input-alias"
        _dataset_alias(alias, inputs, kind=kind)
        output = alias / target.name
    else:
        output = tmp_path / "manifest.json"
        _dataset_alias(output, target, kind=kind)
    prepared = tmp_path / "new-canonical" / "nested"
    argv = ["--train", str(sources["train"]), "--validation", str(sources["validation"]),
            "--output", str(output)]
    if canonicalize:
        argv.extend(["--canonical-output-dir", str(prepared)])
    before = _filesystem_snapshot(tmp_path)

    with pytest.raises(ValueError, match="must not overwrite"):
        validate_dataset_main(argv)

    assert _filesystem_snapshot(tmp_path) == before
    assert not prepared.parent.exists()


@pytest.mark.parametrize("canonical_name", ["train", "validation", "output-directory"])
@pytest.mark.parametrize("relative", [False, True])
def test_cli_manifest_cannot_alias_planned_canonical_paths_before_creating_directories(
    tmp_path, monkeypatch, canonical_name, relative,
) -> None:
    train = tmp_path / "source-train.jsonl"
    validation = tmp_path / "source-validation.jsonl"
    _write(train, [_row("train")])
    _write(validation, [_row("validation")])
    prepared = tmp_path / "new-canonical" / "nested"
    output = prepared if canonical_name == "output-directory" else prepared / f"{canonical_name}.jsonl"
    if relative:
        monkeypatch.chdir(tmp_path)
        output = output.relative_to(tmp_path)
    before = _filesystem_snapshot(tmp_path)

    with pytest.raises(ValueError, match="must not overwrite"):
        validate_dataset_main([
            "--train", str(train), "--validation", str(validation),
            "--canonical-output-dir", str(prepared), "--output", str(output),
        ])

    assert _filesystem_snapshot(tmp_path) == before
    assert not prepared.parent.exists()


@pytest.mark.parametrize("canonical_name", ["train", "validation"])
def test_cli_dangling_manifest_symlink_cannot_alias_a_planned_canonical_output(
    tmp_path, canonical_name,
) -> None:
    train = tmp_path / "source-train.jsonl"
    validation = tmp_path / "source-validation.jsonl"
    _write(train, [_row("train")])
    _write(validation, [_row("validation")])
    prepared = tmp_path / "new-canonical" / "nested"
    output = tmp_path / "manifest.json"
    _dataset_alias(output, prepared / f"{canonical_name}.jsonl", kind="file")
    before = _filesystem_snapshot(tmp_path)

    with pytest.raises(ValueError, match="must not overwrite"):
        validate_dataset_main([
            "--train", str(train), "--validation", str(validation),
            "--canonical-output-dir", str(prepared), "--output", str(output),
        ])

    assert _filesystem_snapshot(tmp_path) == before
    assert not prepared.parent.exists()


@pytest.mark.parametrize("canonical_name", ["train", "validation"])
@pytest.mark.parametrize("kind", ["same", "relative", "file", "directory", "hardlink"])
def test_cli_manifest_cannot_alias_existing_canonical_outputs_before_rewriting_either(
    tmp_path, monkeypatch, canonical_name, kind,
) -> None:
    train = tmp_path / "source-train.jsonl"
    validation = tmp_path / "source-validation.jsonl"
    _write(train, [_row("train")])
    _write(validation, [_row("validation")])
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    for name in ("train", "validation"):
        (prepared / f"{name}.jsonl").write_bytes(f"Retain existing {name}.\r\n".encode("utf-8"))
    target = prepared / f"{canonical_name}.jsonl"
    if kind == "same":
        output = target
    elif kind == "relative":
        monkeypatch.chdir(tmp_path)
        output = Path("prepared") / target.name
    elif kind == "directory":
        alias = tmp_path / "canonical-alias"
        _dataset_alias(alias, prepared, kind=kind)
        output = alias / target.name
    else:
        output = tmp_path / "manifest.json"
        _dataset_alias(output, target, kind=kind)
    before = _filesystem_snapshot(tmp_path)

    with pytest.raises(ValueError, match="must not overwrite"):
        validate_dataset_main([
            "--train", str(train), "--validation", str(validation),
            "--canonical-output-dir", str(prepared), "--output", str(output),
        ])

    assert _filesystem_snapshot(tmp_path) == before


@pytest.mark.parametrize("kind", ["file", "hardlink"])
@pytest.mark.parametrize("cli", [False, True])
def test_canonical_outputs_must_not_alias_each_other_before_any_writes(tmp_path, kind, cli) -> None:
    train = tmp_path / "source-train.jsonl"
    validation = tmp_path / "source-validation.jsonl"
    _write(train, [_row("train")])
    _write(validation, [_row("validation")])
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "train.jsonl").write_bytes(b"Retain both canonical outputs.\r\n")
    _dataset_alias(prepared / "validation.jsonl", prepared / "train.jsonl", kind=kind)
    manifest = tmp_path / "new-manifests" / "manifest.json"
    before = _filesystem_snapshot(tmp_path)

    with pytest.raises(ValueError, match="must not overwrite"):
        if cli:
            validate_dataset_main([
                "--train", str(train), "--validation", str(validation),
                "--canonical-output-dir", str(prepared), "--output", str(manifest),
            ])
        else:
            build_manifest(train, validation, canonical_output_dir=prepared)

    assert _filesystem_snapshot(tmp_path) == before
    assert not manifest.parent.exists()


@pytest.mark.parametrize("kind", ["same", "relative", "file", "directory", "hardlink"])
def test_cli_source_aliases_are_rejected_before_creating_any_output_directory(
    tmp_path, monkeypatch, kind,
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    train = inputs / "source-train.jsonl"
    _write(train, [_row("train")])
    if kind == "same":
        validation = train
    elif kind == "relative":
        monkeypatch.chdir(tmp_path)
        validation = Path("inputs") / train.name
    elif kind == "directory":
        alias = tmp_path / "input-alias"
        _dataset_alias(alias, inputs, kind=kind)
        validation = alias / train.name
    else:
        validation = inputs / "source-validation.jsonl"
        _dataset_alias(validation, train, kind=kind)
    prepared = tmp_path / "new-canonical" / "nested"
    manifest = tmp_path / "new-manifests" / "manifest.json"
    before = _filesystem_snapshot(tmp_path)

    with pytest.raises(ValueError, match="must not overwrite"):
        validate_dataset_main([
            "--train", str(train), "--validation", str(validation),
            "--canonical-output-dir", str(prepared), "--output", str(manifest),
        ])

    assert _filesystem_snapshot(tmp_path) == before
    assert not prepared.parent.exists()
    assert not manifest.parent.exists()


def test_cli_still_writes_distinct_lf_canonical_outputs_and_nested_manifest(tmp_path, capsys) -> None:
    train = tmp_path / "source-train.jsonl"
    validation = tmp_path / "source-validation.jsonl"
    for path, prompt in ((train, "train"), (validation, "validation")):
        path.write_bytes((json.dumps(_row(prompt)) + "\r\n").encode("utf-8"))
    before = {path: path.read_bytes() for path in (train, validation)}
    prepared = tmp_path / "prepared"
    output = prepared / "manifests" / "manifest.json"

    assert validate_dataset_main([
        "--train", str(train), "--validation", str(validation),
        "--canonical-output-dir", str(prepared), "--output", str(output),
    ]) == 0

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == manifest
    assert manifest["isolation"]["overlaps"] == []
    for name, path in (("train", train), ("validation", validation)):
        canonical_path = prepared / f"{name}.jsonl"
        assert canonical_path.read_bytes() == canonical_upload_bytes(path)
        assert b"\r\n" not in canonical_path.read_bytes()
        assert manifest["splits"][name]["upload_sha256"] == hash_file(canonical_path)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    "content",
    ["", " \n\t", None, False, True, 0, 1.5, {}, {"text": "not a part list"},
     [], ["text"], [42], [{"type": "text", "text": " "}],
     [{"type": "text", "text": 7}], [{"type": "image_url", "image_url": {"url": ""}}],
     [{"type": "unknown", "text": "not supported"}]],
)
@pytest.mark.parametrize("role", ["user", "assistant"])
def test_rejects_empty_scalar_or_unsupported_message_content(tmp_path, content, role) -> None:
    row = _row("prompt")
    row["messages"][0 if role == "user" else 1]["content"] = content
    path = tmp_path / "invalid-content.jsonl"
    _write(path, [row])
    report = validate_jsonl(path)
    assert not report.valid
    assert "message.content" in {issue.code for issue in report.issues}


@pytest.mark.parametrize("content", [None, "", " ", "missing"])
def test_allows_real_assistant_function_call_only_shapes(tmp_path, content) -> None:
    assistant = {
        "role": "assistant",
        "tool_calls": [{
            "id": "call_example", "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }],
    }
    if content != "missing":
        assistant["content"] = content
    row = {"messages": [
        {"role": "user", "content": "Look up the item"}, assistant,
        {"role": "tool", "tool_call_id": "call_example", "content": "found"},
        {"role": "assistant", "content": "Done"},
    ]}
    path = tmp_path / "tool-only.jsonl"
    _write(path, [row])
    assert validate_jsonl(path).require_valid().roles["assistant"] == 2


def _call_message(*identifiers: str) -> dict:
    return {"role": "assistant", "tool_calls": [
        {"id": identifier, "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        for identifier in identifiers
    ]}


def _tool_message(identifier: object = "call_one") -> dict:
    return {"role": "tool", "tool_call_id": identifier, "content": "found"}


@pytest.mark.parametrize("tail", [
    [_call_message("call_one"), _tool_message("unknown")],
    [_call_message("call_one"), _tool_message(None)],
    [_call_message("call_one"), _tool_message([])],
    [{"role": "assistant", "content": "no calls"}, _tool_message()],
    [_tool_message(), {"role": "assistant", "content": "done"}],
    [_call_message("call_one"), _tool_message(), _tool_message()],
    [_call_message("call_one", "call_one"), _tool_message()],
    [_call_message("call_one", "call_two"), _tool_message()],
    [_call_message("call_one"), {"role": "assistant", "content": "skipped response"}],
    [_call_message("call_one"), {"role": "user", "content": "interrupt"}, _tool_message()],
    [_call_message("call_one"), _call_message("call_two"), _tool_message("call_two")],
])
def test_rejects_unmatched_duplicate_or_interrupted_tool_results(tmp_path, tail) -> None:
    path = tmp_path / "invalid-tool-sequence.jsonl"
    _write(path, [{"messages": [{"role": "user", "content": "lookup"}, *tail]}])
    report = validate_jsonl(path)
    assert not report.valid
    assert any(issue.code.startswith("message.tool-") for issue in report.issues)


@pytest.mark.parametrize("terminal_calls", [False, True])
def test_tracks_parallel_tool_results_and_allows_terminal_call_supervision(tmp_path, terminal_calls) -> None:
    messages = [
        {"role": "user", "content": "lookup"},
        _call_message("call_one", "call_two"),
        _tool_message("call_two"), _tool_message("call_one"),
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "another lookup"},
        _call_message("call_three"),
    ]
    if not terminal_calls:
        messages.extend([_tool_message("call_three"), {"role": "assistant", "content": "done again"}])
    path = tmp_path / "valid-tool-sequence.jsonl"
    _write(path, [{"messages": messages}])
    assert validate_jsonl(path).require_valid().valid


@pytest.mark.parametrize("calls", [[], {}, ["not a call"], [{"type": "function"}], [
    {"id": "call_example", "type": "function", "function": {"name": "lookup", "arguments": {}}},
]])
def test_tool_call_escape_hatch_requires_real_function_call_shape(tmp_path, calls) -> None:
    row = _row("prompt")
    row["messages"][1] = {"role": "assistant", "content": None, "tool_calls": calls}
    path = tmp_path / "bad-tool-call.jsonl"
    _write(path, [row])
    report = validate_jsonl(path)
    assert not report.valid
    assert "message.tool-calls" in {issue.code for issue in report.issues}


def test_accepts_nonempty_typed_text_and_image_parts(tmp_path) -> None:
    row = _row("prompt")
    row["messages"][0]["content"] = [
        {"type": "text", "text": "Describe the image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
    ]
    row["messages"][1]["content"] = [{"type": "text", "text": "A shape"}]
    path = tmp_path / "multimodal.jsonl"
    _write(path, [row])
    assert validate_jsonl(path).valid


def test_accepts_image_only_parts_and_null_optional_tool_calls(tmp_path) -> None:
    row = _row("prompt")
    row["messages"][0]["content"] = [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA==", "detail": "low"}},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}},
    ]
    row["messages"][1]["tool_calls"] = None
    path = tmp_path / "image-only.jsonl"
    _write(path, [row])
    assert validate_jsonl(path).valid


def test_malformed_role_is_reported_not_raised_as_unhashable(tmp_path) -> None:
    row = _row("prompt")
    row["messages"][0]["role"] = []
    path = tmp_path / "bad-role.jsonl"
    _write(path, [row])
    assert not validate_jsonl(path).valid


@pytest.mark.parametrize(
    "relative",
    [
        "00-foundations/02-first-sft-bug-detection/data/bug-detection-train.jsonl",
        "00-foundations/02-first-sft-bug-detection/data/bug-detection-validation.jsonl",
        "01-core-sft/01-news-summarization/data/news-summarization-train.jsonl",
        "01-core-sft/01-news-summarization/data/news-summarization-validation.jsonl",
        "01-core-sft/02-scientific-summarization/data/scientific-summarization-train.jsonl",
        "01-core-sft/02-scientific-summarization/data/scientific-summarization-validation.jsonl",
        "04-agentic-rft/02-retail-agent-capstone/data/sft-train.jsonl",
        "04-agentic-rft/02-retail-agent-capstone/data/sft-validation.jsonl",
    ],
)
def test_accepts_actual_bundled_sft_datasets(relative) -> None:
    path = Path(__file__).parents[2] / relative
    assert validate_jsonl(path).require_valid().rows > 0


def test_accepts_an_actual_bundled_chart_message_shape(tmp_path) -> None:
    path = (Path(__file__).parents[2] / "02-multimodal-sft" / "01-chart-reasoning"
            / "data" / "preserved" / "chart-qa-run-20261002-v4-val.jsonl")
    with path.open(encoding="utf-8") as stream:
        row = json.loads(stream.readline())
    selected = tmp_path / "actual-chart-row.jsonl"
    _write(selected, [row])
    assert validate_jsonl(selected).require_valid().rows == 1
