from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from catalog_server import execute_search_catalog
from cookbook_utils import (
    grade_sku_answer,
    load_jsonl,
    responses_input,
    responses_tools,
    sha256_file,
    validate_tool_calling_rows,
)


def test_preserved_files_and_rows_match_manifest() -> None:
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        rows = load_jsonl(path)
        assert sha256_file(path) == entry["sha256"]
        assert len(rows) == entry["records"]
        validate_tool_calling_rows(rows, entry["split"])


def test_reference_tool_arguments_include_expected_sku() -> None:
    rows = load_jsonl(ROOT / "data" / "tool-calling-validation.jsonl")
    for row in rows:
        arguments = row["reference_tool_args"][0]
        result = execute_search_catalog(**arguments)
        assert row["reference_answer"] in {item["sku"] for item in result["items"]}


def test_catalog_category_and_specific_queries() -> None:
    assert len(execute_search_catalog("jacket", top_k=3)["items"]) == 3
    assert len(execute_search_catalog("Alpine jacket under $185", top_k=5)["items"]) == 3
    assert execute_search_catalog("Urban Puffer", top_k=1) == {
        "items": [{"sku": "JKT-URB-009", "name": "Urban Puffer", "price": 149.50}]
    }


def test_grader_is_strict_about_output_format() -> None:
    assert grade_sku_answer("JKT-URB-009", "JKT-URB-009")["score"] == 1.0
    assert grade_sku_answer("JKT-ALP-001", "JKT-URB-009")["score"] < 0.1
    assert grade_sku_answer("The SKU is JKT-URB-009", "JKT-URB-009")["score"] == 0.0
    assert grade_sku_answer("", "JKT-URB-009")["score"] == 0.0


def test_chat_tools_convert_to_responses_shape() -> None:
    row = load_jsonl(ROOT / "data" / "tool-calling-validation.jsonl")[0]
    converted = responses_tools(row["tools"])
    assert converted[0]["name"] == "search_catalog"
    assert "function" not in converted[0]
    assert converted[0]["parameters"]["required"] == ["query"]


def test_chat_messages_convert_to_responses_shape() -> None:
    row = load_jsonl(ROOT / "data" / "tool-calling-validation.jsonl")[0]
    converted = responses_input(row["messages"])
    assert converted[0]["type"] == "message"
    assert converted[0]["role"] == "developer"
    assert converted[1]["content"][0]["type"] == "input_text"
