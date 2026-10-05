from __future__ import annotations

import hmac
import json
import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict

app = FastAPI(title="Agentic RFT catalog tool")

CATALOG = [
    {"sku": "JKT-URB-009", "name": "Urban Puffer", "price": 149.50},
    {"sku": "JKT-ALP-001", "name": "Alpine Light Jacket", "price": 179.99},
    {"sku": "JKT-ALP-002", "name": "Alpine Insulated", "price": 199.00},
]
CATEGORY_QUERIES = {"all", "catalog", "jacket", "jackets"}


class ToolCallRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    call_id: str | None = None
    arguments: str | None = None
    query: str | None = None
    top_k: int | None = 3


def _authorize(authorization: str | None, function_key: str | None) -> None:
    expected = os.getenv("FOUNDRY_TOOL_AUTH_TOKEN", "").strip()
    if not expected:
        return
    supplied = (authorization or function_key or "").strip()
    if supplied.lower().startswith("bearer "):
        supplied = supplied[7:].strip()
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


def execute_search_catalog(query: str, top_k: int = 3) -> dict[str, Any]:
    normalized = query.casefold().strip()
    matches = [
        item
        for item in CATALOG
        if not normalized
        or normalized in CATEGORY_QUERIES
        or normalized in item["name"].casefold()
        or normalized in item["sku"].casefold()
    ]
    if not matches:
        matches = CATALOG
    return {"items": matches[: max(1, min(top_k, len(CATALOG)))]}


@app.post("/tool/search_catalog")
def search_catalog(
    request: ToolCallRequest,
    authorization: str | None = Header(default=None),
    function_key: str | None = Header(default=None, alias="X-Functions-Key"),
) -> dict[str, str]:
    _authorize(authorization, function_key)
    arguments: dict[str, Any] = {}
    if request.arguments:
        try:
            parsed = json.loads(request.arguments)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="arguments must be JSON") from exc
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="arguments must decode to an object")
        arguments = parsed

    query = request.query or str(arguments.get("query", ""))
    top_k = request.top_k if request.top_k is not None else arguments.get("top_k", 3)
    if not isinstance(top_k, int):
        raise HTTPException(status_code=400, detail="top_k must be an integer")

    call_id = request.call_id or "call_search_1"
    function_call_id = request.id or f"fc_{call_id.removeprefix('call_')}"
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "id": function_call_id,
        "output": json.dumps(execute_search_catalog(query, top_k)),
    }
