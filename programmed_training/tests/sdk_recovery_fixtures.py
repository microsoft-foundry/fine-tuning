"""Offline SDK transport and backend state for recovery integration tests."""

import asyncio
import json
from dataclasses import dataclass
from urllib.parse import urlsplit

from azure.ai.finetuningsessions.aio import FineTuningSessionClient
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.transport import AsyncHttpResponse, AsyncHttpTransport
from azure.core.utils import case_insensitive_dict


class Response(AsyncHttpResponse):
    def __init__(self, request, payload, status=200):
        super().__init__(request, None)
        self.status_code = status
        self.reason = "offline recovery fixture"
        self.headers = case_insensitive_dict({"content-type": "application/json"})
        self.content_type = "application/json"
        self._body = json.dumps(payload).encode()

    def body(self):
        return self._body

    def text(self, encoding=None):
        return self._body.decode(encoding or "utf-8")

    def json(self):
        return json.loads(self._body)

    async def read(self):
        return self._body

    async def close(self):
        pass


@dataclass
class ModelState:
    weights: int = 0
    optimizer: int = 100


class Service:
    def __init__(
        self, *, failures=1, failure_stage="poll", creation_failures=0, failure_weights=None,
        failed_close_response=None,
    ):
        self.states = {}
        self.checkpoints = {}
        self.requests = {}
        self.creations = []
        self.closed = []
        self.failed_states = []
        self.failures = failures
        self.failure_stage = failure_stage
        self.trace = []
        self.unavailable = set()
        self.list_error = None
        self.creation_failures = creation_failures
        self.failure_weights = None if failure_weights is None else list(failure_weights)
        self.failed_close_response = failed_close_response

    def client(self):
        return FineTuningSessionClient(
            "https://unit.invalid/api/projects/recovery",
            AzureKeyCredential("offline-key"),
            transport=Transport(self), retry_total=0,
        )

    def reply(self, request):
        path = urlsplit(request.url).path.split("/fine_tuning/sessions", 1)[-1]
        parts = path.strip("/").split("/") if path.strip("/") else []
        body = json.loads(request.content) if request.content else {}
        method = request.method
        self.trace.append((method, tuple(parts)))
        if not parts and method == "POST":
            session = f"session_{len(self.creations)}"
            checkpoint = body.get("from_checkpoint")
            state = ModelState()
            if checkpoint:
                source = checkpoint["source_session_id"].replace("model_", "session_")
                state = ModelState(**vars(self.checkpoints[(source, checkpoint["checkpoint_id"])]))
            self.states[session] = state
            self.creations.append((session, body, ModelState(**vars(state))))
            request_id = self.operation("create", {})
            return Response(request, {"session_id": session, "request_id": request_id})
        session = parts[0].replace("model_", "session_")
        operation = parts[1]
        if method == "GET" and operation == "checkpoints":
            if self.list_error:
                return Response(request, {"message": "Discovery failed"}, self.list_error)
            rows = [
                {"checkpoint_id": name, "checkpoint_type": "training",
                 "time": "2026-10-20T00:00:00Z"}
                for source, name in self.checkpoints
                if source == session and (source, name) not in self.unavailable
            ]
            return Response(request, {"checkpoints": rows})
        if method == "GET" and operation == "request":
            kind, result = self.requests[parts[2]]
            if kind == "create" and session != "session_0" and self.creation_failures:
                self.creation_failures -= 1
                kind = "fail"
            if kind == "fail":
                return Response(request, {
                    "status": "failed", "error_code": "engine_dead",
                    "error": "synthetic engine loss", "should_retry": False,
                })
            return Response(request, {"status": "completed", "result": result})
        if method == "POST" and operation in ("complete", "heartbeat"):
            if operation == "complete":
                self.closed.append(session)
                if self.failed_close_response and any(source == session for source, _ in self.failed_states):
                    status, payload = self.failed_close_response
                    return Response(request, payload, status)
                return Response(request, {"request_id": self.operation("complete", {})})
            return Response(request, {})
        assert method == "POST", (method, parts)
        state = self.states[session]
        result = {}
        if operation in ("forward_backward", "forward"):
            data = body[f"{operation}_input"]["data"]
            result["loss_fn_outputs"] = [
                {"logprobs": {"data": [-0.5] * len(datum["loss_fn_inputs"]["target_tokens"]["data"])}}
                for datum in data
            ]
            result["metrics"] = {"loss": 0.5}
        elif operation == "optim_step":
            state.weights += 1
            state.optimizer += 7
            fail_step = (
                state.weights == 4 if self.failure_weights is None else
                bool(self.failure_weights) and state.weights == self.failure_weights[0]
                and not any(failed_session == session for failed_session, _ in self.failed_states)
            )
            if fail_step and self.failures > 0:
                self.failures -= 1
                if self.failure_weights is not None:
                    self.failure_weights.pop(0)
                self.failed_states.append((session, ModelState(**vars(state))))
                if self.failure_stage == "submit":
                    return Response(request, {
                        "error_code": "engine_dead", "reason": "engine_dead",
                        "message": "synthetic engine loss",
                    }, 409)
                operation = "fail"
            result["metrics"] = {"step_count": state.weights, "skyrl.ai/grad_norm": 1.0}
        elif operation == "checkpoint":
            name = body["path"]
            self.checkpoints[(session, name)] = ModelState(**vars(state))
            result = {"checkpoint_id": name, "path": f"loom://model_{session.removeprefix('session_')}/weights/{name}"}
        elif operation == "checkpoint_sample":
            result = {"checkpoint_id": body.get("path") or "ephemeral"}
        elif operation == "sample":
            result = {"sequences": [{"tokens": [10, 11, 12], "logprobs": [-0.5] * 3,
                                     "stop_reason": "stop"}]}
            if body.get("prompt_logprobs"):
                tokens = [token for chunk in body["prompt"]["chunks"] for token in chunk["tokens"]]
                result["prompt_logprobs"] = [None] + [-0.5] * (len(tokens) - 1)
        else:
            raise AssertionError(f"Unexpected fixture operation: {operation}")
        return Response(request, {"request_id": self.operation(operation, result)})

    def operation(self, kind, result):
        request_id = f"request_{len(self.requests)}"
        self.requests[request_id] = (kind, result)
        return request_id


class Transport(AsyncHttpTransport):
    def __init__(self, service):
        self.service = service

    async def send(self, request, **kwargs):
        await asyncio.sleep(0)
        return self.service.reply(request)

    async def open(self):
        pass

    async def close(self):
        pass

    async def __aexit__(self, *args):
        await self.close()
