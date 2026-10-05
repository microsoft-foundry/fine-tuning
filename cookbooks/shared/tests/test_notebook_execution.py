from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

RETAIL = (
    Path(__file__).parents[2]
    / "04-agentic-rft"
    / "02-retail-agent-capstone"
    / "notebooks"
    / "demo.ipynb"
)


def retail_source(marker: str) -> str:
    cells = json.loads(RETAIL.read_text(encoding="utf-8"))["cells"]
    return next(
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code" and marker in "".join(cell["source"])
    )


def test_retail_saved_state_loads_without_uploads_or_training(tmp_path: Path) -> None:
    source = retail_source("def save_state()")
    namespace = {
        "OUTPUTS": tmp_path / "outputs",
        "STATE_PATH": tmp_path / "outputs" / "notebook-state.json",
        "SFT_ENDPOINT": "sft-project", "RFT_ENDPOINT": "rft-project",
        "SFT_MODEL": "sft-model", "RFT_MODEL": "rft-model",
        "TRAINING_TYPE": "globalStandard", "validated_inputs": [],
        "sft_method": {"type": "supervised"},
        "rft_method": {"type": "reinforcement"},
        "json": json, "uuid": uuid,
    }
    code = compile(source, "retail-state", "exec")
    exec(code, namespace)
    namespace["state"]["jobs"] = {
        "sft": {"id": "job-sft", "status": "succeeded", "fine_tuned_model": "sft-artifact"},
        "rft": {"id": "job-rft", "status": "succeeded", "fine_tuned_model": "rft-artifact"},
    }
    namespace["save_state"]()
    exec(code, namespace)
    assert namespace["state"]["jobs"]["sft"]["fine_tuned_model"] == "sft-artifact"
    assert namespace["state"]["files"] == {}
    assert "sft_client" not in namespace
    assert "rft_client" not in namespace


def test_retail_remote_tools_still_require_sandbox() -> None:
    source = retail_source("def execute_remote_tool(")
    function = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "execute_remote_tool"
    )
    namespace = {
        "READ_ONLY_TOOLS": {"get_order_details"},
        "env_flag": lambda _name: False,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "retail-tools", "exec"),
        namespace,
    )
    call = SimpleNamespace(function=SimpleNamespace(name="get_order_details"))
    with pytest.raises(RuntimeError, match="acknowledged isolated sandbox"):
        namespace["execute_remote_tool"](call)
