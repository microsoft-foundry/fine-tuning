from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rft_common import (
    canonical_jsonl_bytes,
    content_addressed_name,
    upload_file_once,
    validate_hash_manifest,
)


class TransientPostError(Exception):
    status_code = 500


def upload_client(*, existing=(), downloaded=b"", error=None):
    created = SimpleNamespace(
        id="uploaded-file", purpose="fine-tune", bytes=len(downloaded),
    )
    create = Mock(return_value=created, side_effect=error)
    files = SimpleNamespace(
        list=Mock(return_value=SimpleNamespace(data=list(existing))),
        retrieve=Mock(return_value=created),
        create=Mock(side_effect=AssertionError("SDK retries were not disabled")),
        wait_for_processing=Mock(),
        content=Mock(return_value=SimpleNamespace(content=downloaded)),
    )
    single_attempt = SimpleNamespace(files=SimpleNamespace(create=create))
    return SimpleNamespace(
        files=files,
        with_options=Mock(return_value=single_attempt),
        create=create,
        created=created,
    )


def test_windows_crlf_manifest_name_and_upload_use_identical_canonical_bytes(tmp_path):
    path = tmp_path / "input.jsonl"
    canonical = b'{"messages":[{"content":"keep\\r\\ninside"}],"answer":"1"}\n'
    path.write_bytes(canonical.replace(b"\n", b"\r\n"))
    digest = hashlib.sha256(canonical).hexdigest()
    manifest = tmp_path / "hashes.json"
    manifest.write_text(json.dumps({
        "normalization": "CRLF-to-LF",
        "artifacts": [{"path": path.name, "bytes": len(canonical), "sha256": digest}],
    }), encoding="utf-8")
    validate_hash_manifest(manifest, tmp_path)
    name = content_addressed_name("math", path)
    assert name == f"math-{digest[:12]}.jsonl"
    assert canonical_jsonl_bytes(path.read_bytes()) == canonical
    client = upload_client(downloaded=canonical)
    order = []
    client.create.side_effect = lambda **_kwargs: (
        order.append("post") or client.created
    )
    client.files.wait_for_processing.side_effect = lambda _id: order.append("wait")
    result = upload_file_once(
        client, path, prefix="math", purpose="fine-tune",
        before_create=lambda: order.append("intent"),
        on_file=lambda _file: order.append("id"),
    )
    assert result.id == "uploaded-file"
    assert order == ["intent", "post", "id", "wait"]
    client.with_options.assert_called_once_with(max_retries=0)
    client.create.assert_called_once_with(
        file=(name, canonical, "application/jsonl"), purpose="fine-tune",
    )
    path.write_bytes(canonical)
    assert content_addressed_name("math", path) == name
    assert b"\\r\\n" in canonical  # JSON string escapes were not normalized.


def test_reused_crlf_download_compares_canonical_content_but_raw_byte_count(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_bytes(b'{"answer":"1"}\n')
    downloaded = path.read_bytes().replace(b"\n", b"\r\n")
    remote = SimpleNamespace(
        id="existing-file", filename=content_addressed_name("math", path),
        purpose="fine-tune", bytes=len(downloaded),
    )
    client = upload_client(existing=[remote], downloaded=downloaded)
    assert upload_file_once(client, path, prefix="math", purpose="fine-tune") is remote
    client.create.assert_not_called()
    client.with_options.assert_not_called()


@pytest.mark.parametrize("defect", ["content", "size", "purpose", "duplicates"])
def test_matching_name_does_not_allow_unverified_file_reuse(tmp_path, defect):
    path = tmp_path / "input.jsonl"
    content = b'{"answer":"1"}\n'
    path.write_bytes(content)
    remote = SimpleNamespace(
        id="existing-file", filename=content_addressed_name("math", path),
        purpose="fine-tune", bytes=len(content),
    )
    if defect == "size":
        remote.bytes += 1
    if defect == "purpose":
        remote.purpose = "assistants"
    client = upload_client(
        existing=[remote, remote] if defect == "duplicates" else [remote],
        downloaded=content.replace(b'"1"', b'"2"') if defect == "content" else content,
    )
    with pytest.raises(RuntimeError):
        upload_file_once(client, path, prefix="math", purpose="fine-tune")
    client.create.assert_not_called()


def test_upload_post_has_no_helper_or_sdk_retry_even_for_transient_error(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_bytes(b'{"answer":"1"}\n')
    client = upload_client(error=TransientPostError("response lost"))
    with pytest.raises(TransientPostError):
        upload_file_once(client, path, prefix="math", purpose="fine-tune")
    client.with_options.assert_called_once_with(max_retries=0)
    assert client.create.call_count == 1
    client.files.wait_for_processing.assert_not_called()


@pytest.mark.parametrize("callback", ["intent", "id"])
def test_checkpoint_failure_stops_upload_workflow(tmp_path, callback):
    path = tmp_path / "input.jsonl"
    path.write_bytes(b'{"answer":"1"}\n')
    client = upload_client(downloaded=path.read_bytes())

    def fail(*_args):
        raise OSError("checkpoint disk failure")

    with pytest.raises(OSError, match="checkpoint disk failure"):
        upload_file_once(
            client, path, prefix="math", purpose="fine-tune",
            before_create=fail if callback == "intent" else None,
            on_file=fail if callback == "id" else None,
        )
    assert client.create.call_count == (0 if callback == "intent" else 1)
    client.files.wait_for_processing.assert_not_called()
    client.files.content.assert_not_called()


def test_ambiguous_upload_does_not_retry_an_empty_listing(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_bytes(b'{"answer":"1"}\n')
    client = upload_client()
    with pytest.raises(RuntimeError, match="ambiguous"):
        upload_file_once(
            client, path, prefix="math", purpose="fine-tune", reconcile_only=True,
        )
    client.create.assert_not_called()
    client.with_options.assert_not_called()


def test_saved_file_id_does_not_fall_back_to_upload_when_retrieve_fails(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_bytes(b'{"answer":"1"}\n')
    client = upload_client()
    client.files.retrieve.side_effect = ValueError("file cannot be retrieved")
    with pytest.raises(RuntimeError, match="retrieve saved RFT file"):
        upload_file_once(
            client, path, prefix="math", purpose="fine-tune",
            existing_file_id="saved-file",
        )
    client.files.list.assert_not_called()
    client.create.assert_not_called()