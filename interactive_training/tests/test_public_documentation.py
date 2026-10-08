"""Shared docs and notebooks work with the source SDK and the public PyPI install."""

import ast
import asyncio
import inspect
import json
import re
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import unquote, urlsplit

import pytest

from interactive_training.rl.train_azure import AzureSDKTrainingClient


ROOT = Path(__file__).resolve().parents[1]
SDK_DISTRIBUTION = "azure-ai-finetuningsessions"


def _source_sdk_root(root):
    # A standalone public checkout can sit next to an SDK checkout, so sibling
    # existence alone must not relax its public-install/link requirements.
    repo = root.parent
    sdk = repo / SDK_DISTRIBUTION
    if (
        not (root / ".git").exists()
        and (repo / ".git").exists()
        and (repo / "scripts" / "sync_finetuning_cookbook.py").is_file()
        and (sdk / "pyproject.toml").is_file()
        and (sdk / "azure" / "ai" / "finetuningsessions" / "__init__.py").is_file()
    ):
        return sdk.resolve()
    return None


SOURCE_SDK_ROOT = _source_sdk_root(ROOT)


def _is_public_path(path, root=ROOT):
    path.relative_to(root)
    return not any(
        (parent / ".internal_only").is_file()
        for parent in path.parents
        if parent.is_relative_to(root)
    )


MARKDOWN = sorted(
    path
    for path in (
        [ROOT / "README.md"]
        + list((ROOT / "docs").rglob("*.md"))
        + list((ROOT / "envs").rglob("*.md"))
        + list((ROOT / "interactive_training" / "recipes").rglob("*.md"))
    )
    if _is_public_path(path)
)
NOTEBOOKS = sorted(path for path in (ROOT / "notebooks").rglob("*.ipynb") if _is_public_path(path))


def _markdown_sources(path):
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"]
    return [path.read_text(encoding="utf-8")]


def _assert_relative_links_resolve(path, root, source_sdk_root):
    for source in _markdown_sources(path):
        source = re.sub(r"(?ms)^```.*?^```\s*$", "", source)
        for match in re.finditer(r"\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", source):
            target = urlsplit(match.group(1))
            if target.scheme or target.netloc:
                continue
            linked = (path.parent / unquote(target.path)).resolve() if target.path else path
            assert linked.is_relative_to(root) or (
                source_sdk_root is not None and linked.is_relative_to(source_sdk_root)
            ), (path, match.group(1))
            assert linked.exists(), (path, match.group(1))


@pytest.mark.parametrize("path", MARKDOWN + NOTEBOOKS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_public_relative_links_stay_in_repo_and_resolve(path):
    # Only a detected source checkout may link outside the cookbook, and only
    # into its SDK. Missing SDK files and other monorepo paths still fail.
    _assert_relative_links_resolve(path, ROOT, SOURCE_SDK_ROOT)


@pytest.mark.parametrize("worktree", [False, True])
def test_source_sdk_detection_requires_monorepo_layout(tmp_path, worktree):
    root = tmp_path / "cookbook"
    root.mkdir()
    sdk = tmp_path / SDK_DISTRIBUTION
    package = sdk / "azure" / "ai" / "finetuningsessions"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (sdk / "pyproject.toml").touch()
    assert _source_sdk_root(root) is None

    if worktree:
        (tmp_path / ".git").write_text("gitdir: /example/worktree\n", encoding="utf-8")
    else:
        (tmp_path / ".git").mkdir()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "sync_finetuning_cookbook.py").touch()
    assert _source_sdk_root(root) == sdk.resolve()

    (root / ".git").mkdir()
    assert _source_sdk_root(root) is None


def test_source_sdk_link_exception_is_narrow_and_checks_existence(tmp_path):
    root = tmp_path / "cookbook"
    root.mkdir()
    sdk = tmp_path / SDK_DISTRIBUTION
    sdk.mkdir()
    (sdk / "README.md").write_text("# SDK\n", encoding="utf-8")
    path = root / "README.md"
    path.write_text(f"[SDK](../{SDK_DISTRIBUTION}/README.md)\n", encoding="utf-8")
    _assert_relative_links_resolve(path, root, sdk)
    with pytest.raises(AssertionError):
        _assert_relative_links_resolve(path, root, None)

    for target in (f"../{SDK_DISTRIBUTION}/missing.md", "../outside.md"):
        (tmp_path / "outside.md").write_text("Outside\n", encoding="utf-8")
        path.write_text(f"[invalid]({target})\n", encoding="utf-8")
        with pytest.raises(AssertionError):
            _assert_relative_links_resolve(path, root, sdk)


def test_public_document_inventory_respects_nested_internal_markers(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    (private / ".internal_only").touch()
    assert not _is_public_path(private / "nested" / "README.md", tmp_path)
    assert _is_public_path(tmp_path / "private_sibling" / "README.md", tmp_path)
    assert _is_public_path(tmp_path / "README.md", tmp_path)


def test_public_setup_does_not_reference_unavailable_helpers():
    assert len(MARKDOWN) >= 30, "Public setup inventory must not pass vacuously"
    for path in MARKDOWN:
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"experiments/[^\s`]*setup[^\s`]*", source), path
        assert not re.search(r"\binternal\s+(?:training\s+)?monorepo\b", source, re.IGNORECASE), path


def test_cookbook_and_sdk_migrations_are_separate():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for anchor in ("updating-an-older-cookbook-checkout", "replacing-an-older-sdk-preview"):
        assert f"./docs/troubleshooting.md#{anchor}" in readme
    source = (ROOT / "docs/troubleshooting.md").read_text(encoding="utf-8")
    migration = source.split("## Migrating an older preview environment\n", 1)[1].split("\n## ", 1)[0]
    cookbook = migration.split("### Updating an older cookbook checkout\n", 1)[1].split("\n### ", 1)[0]
    assert "python -m pip uninstall -y interactive-post-training" in cookbook
    assert not re.search(r"(?m)^python -m pip uninstall[^\n]*azure-ai-", cookbook)
    sdk = migration.split("### Replacing an older SDK preview\n", 1)[1]
    assert "Skip this section if the environment already uses the pinned public SDK." in sdk


@pytest.mark.parametrize("relative", [
    "interactive_training/recipes/code_rl/README.md",
    "interactive_training/sandbox/sandboxfusion.py",
    "interactive_training/recipes/code_rl/code_grading.py",
])
def test_public_sandbox_startup_examples_bind_loopback(relative):
    source = (ROOT / relative).read_text(encoding="utf-8")
    ports = re.findall(r"docker\s+run[^\n]*-p\s+([^\s\"`]+)", source)
    assert ports, f"Missing documented sandbox startup command: {relative}"
    assert all(port == "127.0.0.1:8080:8080" for port in ports), relative


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_public_notebooks_compile_and_have_no_saved_outputs(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"], 1):
        if cell["cell_type"] != "code":
            continue
        assert not cell.get("outputs"), f"{path.name} cell {index} retains output"
        assert cell.get("execution_count") is None
        compile("".join(cell["source"]), f"{path.name}:cell{index}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


def test_sdk_dependency_matches_source_or_public_install():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = [name for name in metadata["project"]["dependencies"] if name.startswith(SDK_DISTRIBUTION)]
    assert len(dependencies) == 1
    sources = metadata["tool"]["uv"]["sources"]
    if SOURCE_SDK_ROOT is not None:
        assert dependencies == [SDK_DISTRIBUTION]
        assert sources[SDK_DISTRIBUTION] == {"path": f"../{SDK_DISTRIBUTION}"}
        assert (ROOT / sources[SDK_DISTRIBUTION]["path"]).resolve() == SOURCE_SDK_ROOT
    else:
        assert re.fullmatch(r"azure-ai-finetuningsessions==[0-9][0-9A-Za-z.!+-]*", dependencies[0])
        assert SDK_DISTRIBUTION not in sources


def test_async_and_notebook_dependencies_are_explicit():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert "aiohttp" in metadata["dependencies"]
    assert any(name.startswith("mcp>=") for name in metadata["optional-dependencies"]["test"])
    assert "psutil>=5.9; sys_platform == 'win32'" in metadata["optional-dependencies"]["test"]
    assert any(name.startswith("ipykernel>=") for name in metadata["optional-dependencies"]["notebook"])


def test_image_extras_include_fast_processor_dependency():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for extra in ("image", "mcp"):
        assert "torchvision" in metadata["project"]["optional-dependencies"][extra]
    sources = metadata["tool"]["uv"]["sources"]
    assert sources["torchvision"] == sources["torch"]


def test_public_model_tables_distinguish_available_and_legacy_models():
    text = (ROOT / "docs/supported_models.md").read_text(encoding="utf-8")
    for heading, next_heading, expected, availability in (
        (
            "Available today", "Legacy models",
            ["Qwen/Qwen3.8-27B", "Qwen/Qwen3.6-35B-A3B",
             "openai/gpt-oss-120b", "meta-models/Muse-Glimmer-30B"],
            "Available",
        ),
        (
            "Legacy models", "Vision fine-tuning",
            ["Qwen/Qwen3-32B", "openai/gpt-oss-20b",
             "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"],
            "⚠️ Legacy",
        ),
    ):
        section = text.split(f"### {heading}\n", 1)[1].split(f"### {next_heading}\n", 1)[0]
        rows = [
            [cell.strip() for cell in line.strip("|").split("|")]
            for line in section.splitlines()
            if line.startswith("| ") and "`" in line.split("|")[2]
            and not line.startswith("| Model ")
        ]
        assert [row[1].strip("`") for row in rows] == expected
        assert all(len(row) == 3 and row[2] == availability for row in rows)


def _notebook_cell(path, marker):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    matches = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(matches) == 1, marker
    return matches[0]


def test_foundational_notebook_default_keeps_non_thinking_preference():
    from interactive_training import model_info

    path = ROOT / "notebooks/foundational/foundational_rl.ipynb"
    source = _notebook_cell(path, "class Config:")
    namespace = {}
    exec(compile(source, "notebook-config", "exec"), namespace)
    config = namespace["CONFIG"]
    assert config.model_name == "Qwen/Qwen3.8-27B"
    assert config.renderer_name is None
    setup = ast.parse(_notebook_cell(path, "renderer_name = CONFIG.renderer_name or"))
    assignments = [
        node for node in ast.walk(setup)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "renderer_name" for target in node.targets)
    ]
    assert len(assignments) == 1
    namespace["model_info"] = model_info
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "notebook-renderer", "exec"), namespace)
    assert namespace["renderer_name"] == "qwen3_8_disable_thinking"


@pytest.mark.parametrize("working_dir", [ROOT, ROOT / "notebooks", ROOT / "notebooks" / "foundational"])
def test_foundational_import_bootstrap_supports_documented_working_dirs(monkeypatch, working_dir):
    monkeypatch.chdir(working_dir)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    source = _notebook_cell(ROOT / "notebooks/foundational/foundational_rl.ipynb", "NOTEBOOK_DIR =")
    namespace = {}
    exec(compile(source, "notebook-bootstrap", "exec"), namespace)
    assert namespace["NOTEBOOK_DIR"] == ROOT / "notebooks" / "foundational"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
async def test_notebooks_report_missing_endpoint_before_client_creation(monkeypatch, path):
    for name in ("AZURE_AI_PROJECT_ENDPOINT", "FOUNDRY_PROJECT_ENDPOINT", "INTERACTIVE_POST_TRAINING_PROJECT_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    source = _notebook_cell(path, "PROJECT_ENDPOINT =")
    code = compile(source, path.name, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    with pytest.raises(ValueError, match="Set AZURE_AI_PROJECT_ENDPOINT"):
        result = eval(code, {})
        if inspect.isawaitable(result):
            await result


def _foundational_cleanup_state(*, use_api_key, session_id="session_test"):
    path = ROOT / "notebooks/foundational/foundational_rl.ipynb"
    source = ast.parse(_notebook_cell(path, "async def close_training_session"))
    state_names = {"session_id", "resources_closed", "session_closed", "client_closed", "credential_closed"}
    statements = [
        node for node in source.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in state_names for target in node.targets)
        ) or (isinstance(node, ast.AsyncFunctionDef) and node.name == "close_training_session")
    ]
    client = SimpleNamespace(close_session=AsyncMock(), close=AsyncMock())
    # API-key credentials have no close method; identity credentials own one.
    credential = SimpleNamespace() if use_api_key else SimpleNamespace(close=AsyncMock())
    namespace = dict(client=client, credential=credential, api_key="test-key" if use_api_key else None)
    exec(compile(ast.Module(body=statements, type_ignores=[]), "notebook-cleanup", "exec"), namespace)
    namespace["session_id"] = session_id
    return namespace, client, credential


@pytest.mark.asyncio
@pytest.mark.parametrize("use_api_key", [False, True])
@pytest.mark.parametrize("error_type", [ConnectionError, asyncio.CancelledError])
async def test_foundational_cleanup_retries_remote_failure(use_api_key, error_type, capsys):
    namespace, client, credential = _foundational_cleanup_state(use_api_key=use_api_key)
    client.close_session.side_effect = [error_type("transient close failure"), None]
    path = ROOT / "notebooks/foundational/foundational_rl.ipynb"
    cleanup_cell = compile(
        _notebook_cell(path, "Training session and clients closed."),
        "notebook-cleanup-cell", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
    )

    with pytest.raises(error_type):
        await eval(cleanup_cell, namespace)
    assert "Training session and clients closed." not in capsys.readouterr().out
    assert namespace["resources_closed"] is False
    assert namespace["session_closed"] is False
    client.close.assert_not_awaited()
    if not use_api_key:
        credential.close.assert_not_awaited()

    await eval(cleanup_cell, namespace)
    assert "Training session and clients closed." in capsys.readouterr().out
    assert client.close_session.await_count == 2
    client.close_session.assert_awaited_with("session_test")
    client.close.assert_awaited_once()
    if not use_api_key:
        credential.close.assert_awaited_once()
    assert all(namespace[name] is True for name in (
        "session_closed", "client_closed", "credential_closed", "resources_closed"
    ))

    await namespace["close_training_session"]()
    assert client.close_session.await_count == 2
    client.close.assert_awaited_once()
    if not use_api_key:
        credential.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(("use_api_key", "failure"), [(True, "client"), (False, "client"), (False, "credential")])
async def test_foundational_cleanup_retries_only_unfinished_local_close(use_api_key, failure):
    namespace, client, credential = _foundational_cleanup_state(use_api_key=use_api_key)
    closer = client.close if failure == "client" else credential.close
    closer.side_effect = [ConnectionError("local close failure"), None]

    with pytest.raises(ConnectionError, match="local close failure"):
        await namespace["close_training_session"]()
    assert namespace["session_closed"] is True
    assert namespace["resources_closed"] is False
    assert namespace["client_closed"] is (failure == "credential")
    if failure == "client" and not use_api_key:
        credential.close.assert_not_awaited()

    await namespace["close_training_session"]()
    await namespace["close_training_session"]()
    client.close_session.assert_awaited_once_with("session_test")
    assert client.close.await_count == (2 if failure == "client" else 1)
    if not use_api_key:
        assert credential.close.await_count == (2 if failure == "credential" else 1)
    assert namespace["resources_closed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("use_api_key", [False, True])
@pytest.mark.parametrize("session_id", [None, "session_test"])
async def test_foundational_cleanup_success_is_idempotent(use_api_key, session_id):
    namespace, client, credential = _foundational_cleanup_state(use_api_key=use_api_key, session_id=session_id)
    await namespace["close_training_session"]()
    await namespace["close_training_session"]()
    if session_id is None:
        client.close_session.assert_not_awaited()
    else:
        client.close_session.assert_awaited_once_with(session_id)
    client.close.assert_awaited_once()
    if not use_api_key:
        credential.close.assert_awaited_once()
    assert namespace["resources_closed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_training", [False, True])
async def test_sdk_guide_helper_matches_wrapper_signature_and_closes_session(monkeypatch, fail_training):
    text = (ROOT / "docs/sdk-reference.md").read_text(encoding="utf-8").split("## Putting it together", 1)[1]
    source = re.search(r"```python\n(.*?)```", text, re.DOTALL).group(1)
    namespace = {}
    exec(compile(source, "sdk-guide-example", "exec"), namespace)
    monkeypatch.setenv("AZURE_AI_PROJECT_ENDPOINT", "https://example.invalid/api/projects/test")
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.create_session.return_value = "session_test"
    credential = AsyncMock()
    credential.__aenter__.return_value = credential
    forward = SimpleNamespace(result_async=AsyncMock(side_effect=ValueError("bad batch") if fail_training else None))
    optimizer = SimpleNamespace(result_async=AsyncMock())
    checkpoint = SimpleNamespace(result_async=AsyncMock(return_value=SimpleNamespace(path="checkpoint_test")))
    trainer = SimpleNamespace(
        forward_backward_async=AsyncMock(return_value=forward),
        optim_step_async=AsyncMock(return_value=optimizer),
        save_state_async=AsyncMock(return_value=checkpoint),
    )

    def training_factory(*args, **kwargs):
        inspect.signature(AzureSDKTrainingClient).bind(*args, **kwargs)
        assert args[0] is client
        assert args[1] == "session_test"
        return trainer

    namespace.update(
        DefaultAzureCredential=lambda: credential,
        FineTuningSessionClient=lambda **kwargs: client,
        AzureSDKTrainingClient=training_factory,
        get_tokenizer=lambda model: object(),
    )
    if fail_training:
        with pytest.raises(ValueError, match="bad batch"):
            await namespace["train_one_batch"]([object()])
        trainer.optim_step_async.assert_not_called()
    else:
        assert await namespace["train_one_batch"]([object()]) == "checkpoint_test"
        optimizer.result_async.assert_awaited_once()
        checkpoint.result_async.assert_awaited_once()
    client.close_session.assert_awaited_once_with("session_test")


def test_management_notebook_chooses_latest_checkpoint_by_time():
    path = ROOT / "notebooks/manage_sessions.ipynb"
    source = _notebook_cell(path, "latest_training =")
    client = Mock()
    namespace = dict(
        ckpts=SimpleNamespace(checkpoints=[
            SimpleNamespace(checkpoint_type="training", checkpoint_id="newest", time=20),
            SimpleNamespace(checkpoint_type="training", checkpoint_id="older", time=10),
        ]),
        client=client, SESSION_ID="session_test", OPTS={}, _enum=lambda value: value,
    )
    exec(compile(source, "latest-checkpoint-example", "exec"), namespace)
    assert client.checkpoints.get.call_args.kwargs["checkpoint_id"] == "newest"