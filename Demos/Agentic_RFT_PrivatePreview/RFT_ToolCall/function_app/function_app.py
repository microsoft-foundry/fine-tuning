
import json
import re
import time
import os
import hmac
from typing import Any, Dict, Optional

import azure.functions as func
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict

# ----------------------------
# ENV switches
# ----------------------------
# If true, /score accepts anonymous requests (no headers).
SCORE_AUTH_DISABLED = os.getenv("SCORE_AUTH_DISABLED", "false").lower() == "true"

# If true, /tool/* accepts anonymous requests (no headers).
TOOL_AUTH_DISABLED = os.getenv("TOOL_AUTH_DISABLED", "false").lower() == "true"

# Optional: app-level shared secret for tool protection (recommended if TOOL_AUTH_DISABLED is false)
# If empty, we only check "a header exists" (same as your original check_auth behavior).
TOOL_SHARED_SECRET = os.getenv("TOOL_SHARED_SECRET", "").strip()

fastapi_app = FastAPI(title="Agentic RFT Tools & Endpoint Grader")

# ----------------------------
# Auth helper (app-level)
# ----------------------------
def require_auth(x_functions_key: Optional[str], authorization: Optional[str], *, shared_secret: str = ""):
    # Accept either Azure Functions key-style header or Authorization header (same as your intent)
    provided = x_functions_key or authorization
    if not provided:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # If you provide a shared secret, enforce it (recommended).
    if shared_secret:
        token = provided.strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        if not hmac.compare_digest(token, shared_secret):
            raise HTTPException(status_code=401, detail="Unauthorized")

# ---------------------------------------------------------
# In-memory trace store: trace_id -> last_seen + tool_called
# ---------------------------------------------------------
_TRACE_TTL_SEC = 30 * 60  # 30 minutes
_trace_state: Dict[str, Dict[str, Any]] = {}

def mark_tool_called(trace_id: str):
    if not trace_id:
        return
    _trace_state[trace_id] = {"tool_called": True, "ts": time.time()}

def was_tool_called(trace_id: str) -> bool:
    if not trace_id:
        return False
    rec = _trace_state.get(trace_id)
    if not rec:
        return False
    if time.time() - rec.get("ts", 0) > _TRACE_TTL_SEC:
        _trace_state.pop(trace_id, None)
        return False
    return bool(rec.get("tool_called", False))

# ----------------------------
# Tool endpoint: search_catalog
# ----------------------------
class ToolCallRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    # wrapper fields that Agentic RFT sends
    type: Optional[str] = None
    id: Optional[str] = None          # fc_...
    call_id: Optional[str] = None     # call_...
    name: Optional[str] = None        # search_catalog
    arguments: Optional[str] = None   # JSON string
    item: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None

    # allow direct args too (handy for local curl tests)
    query: Optional[str] = None
    top_k: Optional[int] = 3

@fastapi_app.post("/tool/search_catalog")
def search_catalog(
    req: ToolCallRequest,
    x_functions_key: Optional[str] = Header(default=None, alias="X-Functions-Key"),
    authorization: Optional[str] = Header(default=None),
):
    # IMPORTANT: This is app-level auth (NOT Functions host auth).
    # If you want tool to require a header, leave TOOL_AUTH_DISABLED=false.
    if not TOOL_AUTH_DISABLED:
        require_auth(x_functions_key, authorization, shared_secret=TOOL_SHARED_SECRET)

    if req.trace_id:
        mark_tool_called(req.trace_id)

    # Parse arguments if provided (Agentic RFT path)
    args = {}
    if req.arguments:
        try:
            args = json.loads(req.arguments)
        except Exception:
            args = {}

    query = req.query or args.get("query") or ""
    top_k = req.top_k if req.top_k is not None else args.get("top_k", 3)
    top_k = int(top_k) if str(top_k).isdigit() else 3

    # Dummy catalog
    items = [
        {"sku": "JKT-URB-009", "name": "Urban Puffer", "price": 149.50},
        {"sku": "JKT-ALP-001", "name": "Alpine Light Jacket", "price": 179.99},
        {"sku": "JKT-ALP-002", "name": "Alpine Insulated", "price": 199.00},
    ]
    out = {"items": items[:top_k]}

    call_id = req.call_id or "call_search_1"
    fc_id = req.id or f"fc_{call_id.replace('call_', '')}"

    # ✅ SAME OUTPUT PATTERN AS YOUR ORIGINAL
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "id": fc_id,
        "output": json.dumps(out),
    }

# ----------------------------
# Endpoint grader: /score
# ----------------------------
SKU_RE = re.compile(r"^[A-Z]{3}-[A-Z]{3}-\d{3}$")  # e.g., JKT-URB-009

class GraderRequest(BaseModel):
    sample: Dict[str, Any]
    item: Dict[str, Any]
    trace_id: Optional[str] = None

def extract_output_text(sample: Dict[str, Any]) -> str:
    if isinstance(sample.get("output_text"), str):
        return sample["output_text"]
    for k in ("text", "content", "output"):
        v = sample.get(k)
        if isinstance(v, str):
            return v
    try:
        return sample["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(sample)

@fastapi_app.post("/score")
def score(
    req: GraderRequest,
    x_functions_key: Optional[str] = Header(default=None, alias="X-Functions-Key"),
    authorization: Optional[str] = Header(default=None),
):
    # If you want /score to be anonymously callable (for Endpoint Grader no-auth test):
    # set SCORE_AUTH_DISABLED=true in Function App settings.
    if not SCORE_AUTH_DISABLED:
        require_auth(x_functions_key, authorization, shared_secret="")  # keep as "header must exist" unless you add a SCORE secret

    model_text = extract_output_text(req.sample).strip()
    ref = str(req.item.get("reference_answer", "")).strip()

    score_val = 0.0
    if model_text == ref and ref:
        score_val = 1.0
    elif ref and ref.lower() in model_text.lower():
        score_val = 0.7
    elif SKU_RE.match(model_text):
        score_val = 0.3

    if req.trace_id and was_tool_called(req.trace_id) and SKU_RE.match(model_text):
        score_val = min(1.0, score_val + 0.1)

    # ✅ SAME OUTPUT PATTERN AS YOUR ORIGINAL
    return {"score": float(score_val)}

# ----------------------------
# Azure Functions wrapper (host auth)
# ----------------------------
# ✅ This makes Functions host NOT require function keys (platform-level).
# The function_call_output format stays identical (it’s your returned JSON).
main = func.AsgiFunctionApp(app=fastapi_app, http_auth_level=func.AuthLevel.ANONYMOUS)
