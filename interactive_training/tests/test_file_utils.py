"""File permission handling works with and without descriptor-based chmod."""

import os
import stat
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from interactive_training.utils import file_utils


def test_private_permissions_prefer_descriptor(monkeypatch):
    api = SimpleNamespace(fchmod=Mock(), chmod=Mock())
    monkeypatch.setattr(file_utils, "os", api)

    file_utils.set_private_file_permissions(42, "run_meta.json")

    api.fchmod.assert_called_once_with(42, 0o600)
    api.chmod.assert_not_called()


def test_private_permissions_fall_back_when_fchmod_is_unavailable(monkeypatch):
    api = SimpleNamespace(chmod=Mock())
    monkeypatch.setattr(file_utils, "os", api)

    file_utils.set_private_file_permissions(42, "run_meta.json")

    api.chmod.assert_called_once_with("run_meta.json", 0o600)


@pytest.mark.parametrize("descriptor_api", [False, True])
def test_private_permissions_do_not_suppress_permission_errors(monkeypatch, descriptor_api):
    failing_chmod = Mock(side_effect=PermissionError("permission denied"))
    api = SimpleNamespace(chmod=Mock() if descriptor_api else failing_chmod)
    if descriptor_api:
        api.fchmod = failing_chmod
    monkeypatch.setattr(file_utils, "os", api)

    with pytest.raises(PermissionError, match="permission denied"):
        file_utils.set_private_file_permissions(42, "run_meta.json")
    if descriptor_api:
        api.chmod.assert_not_called()


@pytest.mark.parametrize("without_fchmod", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_private_permissions_on_new_and_existing_files(monkeypatch, tmp_path, without_fchmod, existing):
    path = tmp_path / "output.json"
    if existing:
        path.write_text("stale", encoding="utf-8")
        path.chmod(0o644)
    if without_fchmod:
        monkeypatch.delattr(os, "fchmod", raising=False)

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        file_utils.set_private_file_permissions(descriptor, path)
        output.write('{"complete": true}\n')

    assert path.read_text(encoding="utf-8") == '{"complete": true}\n'
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600