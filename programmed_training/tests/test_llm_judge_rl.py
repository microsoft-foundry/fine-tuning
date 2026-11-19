"""Focused offline coverage for the public BillSum LLM-as-judge recipe."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

from programmed_training import model_info
from programmed_training.recipes.llm_judge_rl.data import (
    BillSumExample,
    deterministic_train_validation_split,
    load_synthetic_jsonl,
    prepare_billsum_data,
)
from programmed_training.recipes.llm_judge_rl.env import (
    LLMJudgeGroupBuilder,
    SummaryEnv,
    build_policy_messages,
)
from programmed_training.recipes.llm_judge_rl.grader import (
    DIMENSION_NAMES,
    AzureOpenAIJudge,
    JudgeGradingError,
    JudgeHTTPResponse,
    JudgeResult,
    build_judge_messages,
    compute_weighted_reward,
    is_transient_status,
    judge_response_format,
    judge_responses_text_format,
    parse_judge_response,
)
from programmed_training.recipes.llm_judge_rl.train_azure import (
    CLIConfig,
    JudgeDropRecorder,
    _resolve_renderer_name,
    build_run_metadata,
)
from programmed_training.rl.rollouts import GroupDropEvent, IncompleteGroupGradingError


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "programmed_training"
    / "recipes"
    / "llm_judge_rl"
    / "data"
    / "synthetic_billsum.jsonl"
)


def _valid_dimensions(score: int = 4) -> list[dict[str, object]]:
    return [
        {"name": name, "score": score, "reason": f"Reason for {name}"}
        for name in DIMENSION_NAMES
    ]


def _judgment(score: int = 4):
    return parse_judge_response(json.dumps({"dimensions": _valid_dimensions(score)}))


def test_strict_schema_parsing_rejects_bool_missing_extra_and_duplicates():
    parsed = _judgment(0)
    assert [dimension.score for dimension in parsed.dimensions] == [0] * 5

    invalid_payloads = []
    bool_scores = _valid_dimensions()
    bool_scores[0]["score"] = True
    invalid_payloads.append({"dimensions": bool_scores})
    missing_reason = _valid_dimensions()
    missing_reason[0].pop("reason")
    invalid_payloads.append({"dimensions": missing_reason})
    extra_field = _valid_dimensions()
    extra_field[0]["overall"] = 1
    invalid_payloads.append({"dimensions": extra_field})
    duplicate = _valid_dimensions()
    duplicate[-1]["name"] = duplicate[0]["name"]
    invalid_payloads.append({"dimensions": duplicate})
    invalid_payloads.append({"dimensions": _valid_dimensions(), "overall": 1})
    invalid_payloads.append({"dimensions": _valid_dimensions()[:-1]})

    for payload in invalid_payloads:
        with pytest.raises(ValueError):
            parse_judge_response(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate JSON object key: score"):
        parse_judge_response(
            '{"dimensions":['
            '{"name":"factual_consistency","score":0,"score":4,"reason":"x"},'
            '{"name":"coverage","score":4,"reason":"x"},'
            '{"name":"relevance","score":4,"reason":"x"},'
            '{"name":"clarity","score":4,"reason":"x"},'
            '{"name":"concision","score":4,"reason":"x"}]}'
        )
    with pytest.raises(ValueError):
        parse_judge_response("not-json")


def test_weighted_reward_and_factuality_cap_are_local():
    reward, dimensions, capped = compute_weighted_reward(_judgment(4))
    assert reward == pytest.approx(1.0)
    assert dimensions == {name: 1.0 for name in DIMENSION_NAMES}
    assert capped is False

    scores = _valid_dimensions(4)
    scores[0]["score"] = 0
    reward, dimensions, capped = compute_weighted_reward(
        parse_judge_response(json.dumps({"dimensions": scores}))
    )
    assert dimensions["factual_consistency"] == 0.0
    assert reward == pytest.approx(0.5)
    assert capped is True


class _FakeRenderer:
    def __init__(
        self,
        candidate: str = "A concise summary.",
        *,
        parse_success: bool = True,
    ) -> None:
        self.candidate = candidate
        self.parse_success = parse_success
        self.last_messages = None

    def build_generation_prompt(self, messages):
        self.last_messages = messages
        return ModelInput(chunks=[ModelInputChunk(tokens=[1, 2, 3])])

    def get_stop_sequences(self):
        return []

    def parse_response(self, action):
        return (
            {"role": "assistant", "content": self.candidate},
            self.parse_success,
        )


class _ZeroJudge:
    async def grade(self, example, candidate):
        return JudgeResult(
            reward=0.0,
            normalized_dimensions={name: 0.0 for name in DIMENSION_NAMES},
            factuality_cap_applied=False,
            transport_retries=0,
            structured_retries=0,
            latency_ms=1.0,
            prompt_tokens=10,
        )


class _FailingJudge:
    async def grade(self, example, candidate):
        raise JudgeGradingError("transport_exhausted", transport_retries=2)


class _HighJudge:
    async def grade(self, example, candidate):
        return JudgeResult(
            reward=1.0,
            normalized_dimensions={name: 1.0 for name in DIMENSION_NAMES},
            factuality_cap_applied=False,
            transport_retries=0,
            structured_retries=0,
            latency_ms=1.0,
            prompt_tokens=10,
        )


@pytest.mark.asyncio
async def test_legitimate_zero_is_scored_but_judge_failure_is_not_zero():
    example = load_synthetic_jsonl(FIXTURE)[0]
    env = SummaryEnv(
        renderer=_FakeRenderer("Summary."),
        example=example,
        word_limit=200,
        policy_max_tokens=32,
        dataset_metrics={},
    )
    await env.step([1, 2])
    zero_builder = LLMJudgeGroupBuilder(
        renderer=_FakeRenderer(),
        example=example,
        judge=_ZeroJudge(),
        num_envs=1,
        word_limit=200,
        policy_max_tokens=32,
        dataset_metrics={},
    )
    scored = await zero_builder.compute_group_rewards([], [env])
    assert scored[0][0] == 0.0
    assert scored[0][1]["judge/failures"] == 0.0
    assert scored[0][1]["policy/eligibility_adjusted_reward"] == 0.0

    failing_builder = LLMJudgeGroupBuilder(
        renderer=_FakeRenderer(),
        example=example,
        judge=_FailingJudge(),
        num_envs=1,
        word_limit=200,
        policy_max_tokens=32,
        dataset_metrics={},
    )
    with pytest.raises(IncompleteGroupGradingError) as exc_info:
        await failing_builder.compute_group_rewards([], [env])
    assert isinstance(exc_info.value.__cause__, JudgeGradingError)


@pytest.mark.asyncio
async def test_transport_retries_are_bounded_and_classified(monkeypatch):
    calls = 0

    async def no_sleep(_delay):
        return None

    judge = AzureOpenAIJudge(
        endpoint="https://judge.example",
        deployment="judge",
        api_version="2025-04-01-preview",
        max_transport_retries=2,
        sleep=no_sleep,
    )

    async def rate_limited(_payload):
        nonlocal calls
        calls += 1
        return JudgeHTTPResponse(
            status=429, headers={"Retry-After": "0"}, payload={}
        )

    monkeypatch.setattr(judge, "_post_once", rate_limited)
    with pytest.raises(JudgeGradingError) as exc_info:
        await judge._request_with_retries({})
    assert calls == 3
    assert exc_info.value.category == "transport_exhausted"
    assert exc_info.value.transport_retries == 2
    assert is_transient_status(408)
    assert is_transient_status(500)
    assert is_transient_status(429)
    assert not is_transient_status(400)


@pytest.mark.asyncio
async def test_structured_output_retry_is_separate_and_bounded(monkeypatch):
    example = load_synthetic_jsonl(FIXTURE)[0]
    calls = 0
    judge = AzureOpenAIJudge(
        endpoint="https://judge.example",
        deployment="judge",
        api_version="2025-04-01-preview",
        max_transport_retries=0,
        max_structured_retries=1,
    )

    async def malformed(_payload):
        nonlocal calls
        calls += 1
        return JudgeHTTPResponse(
            status=200,
            headers={},
            payload={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "not-json", "refusal": None},
                    }
                ]
            },
        )

    monkeypatch.setattr(judge, "_post_once", malformed)
    with pytest.raises(JudgeGradingError) as exc_info:
        await judge.grade(example, "candidate")
    assert calls == 2
    assert exc_info.value.category == "malformed_structured_output"
    assert exc_info.value.transport_retries == 0
    assert exc_info.value.structured_retries == 1


@pytest.mark.asyncio
async def test_transport_retry_budget_is_cumulative_across_structured_attempts(
    monkeypatch,
):
    example = load_synthetic_jsonl(FIXTURE)[0]
    calls = 0

    async def no_sleep(_delay):
        return None

    malformed = JudgeHTTPResponse(
        status=200,
        headers={},
        payload={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "not-json", "refusal": None},
                }
            ]
        },
    )
    responses = [
        JudgeHTTPResponse(status=429, headers={"Retry-After": "0"}, payload={}),
        JudgeHTTPResponse(status=429, headers={"Retry-After": "0"}, payload={}),
        malformed,
        JudgeHTTPResponse(status=429, headers={"Retry-After": "0"}, payload={}),
    ]
    judge = AzureOpenAIJudge(
        endpoint="https://judge.example",
        deployment="judge",
        api_version="2025-04-01-preview",
        max_transport_retries=2,
        max_structured_retries=1,
        sleep=no_sleep,
    )

    async def sequence(_payload):
        nonlocal calls
        response = responses[calls]
        calls += 1
        return response

    monkeypatch.setattr(judge, "_post_once", sequence)
    with pytest.raises(JudgeGradingError) as exc_info:
        await judge.grade(example, "candidate")
    assert calls == 4
    assert exc_info.value.category == "transport_exhausted"
    assert exc_info.value.transport_retries == 2
    assert exc_info.value.structured_retries == 1


@pytest.mark.asyncio
async def test_grade_concurrency_is_bounded_across_examples(monkeypatch):
    examples = load_synthetic_jsonl(FIXTURE)[:6]
    active = 0
    peak_active = 0
    judge = AzureOpenAIJudge(
        endpoint="https://judge.example",
        deployment="judge",
        api_version="2025-04-01-preview",
        max_transport_retries=0,
        max_concurrency=2,
    )
    valid_content = json.dumps({"dimensions": _valid_dimensions(4)})

    async def delayed_response(_payload, *, max_retries=None):
        nonlocal active, peak_active
        assert max_retries == 0
        active += 1
        peak_active = max(peak_active, active)
        try:
            await asyncio.sleep(0.01)
        finally:
            active -= 1
        return (
            JudgeHTTPResponse(
                status=200,
                headers={},
                payload={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": valid_content,
                                "refusal": None,
                            },
                        }
                    ]
                },
            ),
            0,
        )

    monkeypatch.setattr(judge, "_request_with_retries", delayed_response)
    results = await asyncio.gather(
        *[judge.grade(example, "candidate") for example in examples]
    )
    assert peak_active == 2
    assert all(result.reward == pytest.approx(1.0) for result in results)


def test_judge_max_concurrency_must_be_positive():
    with pytest.raises(ValueError, match="max_concurrency must be at least 1"):
        AzureOpenAIJudge(
            endpoint="https://judge.example",
            deployment="judge",
            api_version="2025-04-01-preview",
            max_concurrency=0,
        )


def test_protocol_urls_and_payloads_use_the_native_api_shapes():
    messages = [{"role": "user", "content": "grade"}]
    responses_judge = AzureOpenAIJudge(
        endpoint="https://judge.example/openai/v1/",
        deployment="gpt-5.6-sol",
        api_version="unused-for-responses",
        max_response_tokens=321,
    )
    assert responses_judge.protocol == "responses"
    assert responses_judge._request_url() == (
        "https://judge.example/openai/v1/responses"
    )
    assert responses_judge._request_payload(messages) == {
        "model": "gpt-5.6-sol",
        "input": messages,
        "max_output_tokens": 321,
        "text": {"format": judge_responses_text_format()},
    }
    assert judge_responses_text_format() == {
        "type": "json_schema",
        "name": "billsum_summary_judgment",
        "strict": True,
        "schema": judge_response_format()["json_schema"]["schema"],
    }
    serialized_schema = json.dumps(judge_response_format())
    for unsupported in ("minItems", "maxItems", "minimum", "maximum", "minLength"):
        assert unsupported not in serialized_schema

    chat_judge = AzureOpenAIJudge(
        endpoint="https://judge.example",
        deployment="reasoning/deployment",
        api_version="2025-04-01-preview",
        max_response_tokens=654,
    )
    assert chat_judge.protocol == "chat_completions"
    assert chat_judge._request_url() == (
        "https://judge.example/openai/deployments/reasoning%2Fdeployment/"
        "chat/completions?api-version=2025-04-01-preview"
    )
    assert chat_judge._request_payload(messages) == {
        "messages": messages,
        "temperature": 0,
        "max_completion_tokens": 654,
        "response_format": judge_response_format(),
    }


@pytest.mark.asyncio
async def test_responses_request_uses_exact_url_payload_and_api_key(monkeypatch):
    api_key = "mock-responses-api-key"
    monkeypatch.setenv("AZURE_OPENAI_JUDGE_API_KEY", api_key)
    messages = [{"role": "user", "content": "grade"}]
    captured = []

    class Response:
        status = 200
        headers = {}

        async def text(self):
            return "{}"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

    class Session:
        def post(self, url, *, headers, json):
            captured.append((url, dict(headers), json))
            return Response()

    judge = AzureOpenAIJudge(
        endpoint="https://judge.example/openai/v1",
        deployment="gpt-5.6-sol",
        api_version="not-sent",
        max_response_tokens=222,
    )
    judge._session = Session()
    payload = judge._request_payload(messages)
    await judge._post_once(payload)

    assert captured == [
        (
            "https://judge.example/openai/v1/responses",
            {"api-key": api_key, "Content-Type": "application/json"},
            {
                "model": "gpt-5.6-sol",
                "input": messages,
                "max_output_tokens": 222,
                "text": {"format": judge_responses_text_format()},
            },
        )
    ]
    assert "api-version" not in captured[0][0]


def test_responses_content_extraction_accepts_completed_structured_text():
    judge = AzureOpenAIJudge(
        endpoint="https://judge.example/openai/v1",
        deployment="gpt-5.6-sol",
        api_version="unused",
    )
    content = json.dumps({"dimensions": _valid_dimensions(3)})
    response = JudgeHTTPResponse(
        status=200,
        headers={},
        payload={
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": []},
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": content,
                            "annotations": [],
                        }
                    ],
                },
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 200,
                "total_tokens": 300,
            },
        },
    )
    assert judge._content_from_response(response) == content
    assert parse_judge_response(judge._content_from_response(response)).dimensions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "category", "expected_calls", "structured_retries"),
    [
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [],
            },
            "response_truncated",
            1,
            0,
        ),
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
                "output": [],
            },
            "response_incomplete",
            1,
            0,
        ),
        (
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "status": "completed",
                        "content": [
                            {"type": "refusal", "refusal": "Cannot grade"}
                        ],
                    }
                ],
            },
            "response_refusal",
            1,
            0,
        ),
        (
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "status": "completed",
                        "content": [],
                    }
                ],
            },
            "missing_response_text",
            2,
            1,
        ),
    ],
)
async def test_responses_failures_are_explicit_and_never_become_zero(
    monkeypatch, payload, category, expected_calls, structured_retries
):
    example = load_synthetic_jsonl(FIXTURE)[0]
    calls = 0
    judge = AzureOpenAIJudge(
        endpoint="https://judge.example/openai/v1",
        deployment="gpt-5.6-sol",
        api_version="unused",
        max_structured_retries=1,
    )

    async def response_once(_payload):
        nonlocal calls
        calls += 1
        return JudgeHTTPResponse(status=200, headers={}, payload=payload)

    monkeypatch.setattr(judge, "_post_once", response_once)
    with pytest.raises(JudgeGradingError) as exc_info:
        await judge.grade(example, "candidate")
    assert calls == expected_calls
    assert exc_info.value.category == category
    assert exc_info.value.structured_retries == structured_retries


@pytest.mark.asyncio
async def test_responses_http_400_is_sanitized_and_not_retried(monkeypatch):
    judge = AzureOpenAIJudge(
        endpoint="https://judge.example/openai/v1",
        deployment="gpt-5.6-sol",
        api_version="unused",
        max_transport_retries=2,
    )
    calls = 0

    async def bad_request(_payload):
        nonlocal calls
        calls += 1
        return JudgeHTTPResponse(
            status=400,
            headers={},
            payload={"error": {"message": "sensitive provider detail"}},
        )

    monkeypatch.setattr(judge, "_post_once", bad_request)
    with pytest.raises(JudgeGradingError) as exc_info:
        await judge._request_with_retries({})
    assert calls == 1
    assert exc_info.value.category == "http_non_retryable"
    assert exc_info.value.status == 400
    assert "sensitive provider detail" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_chat_truncation_is_explicit_and_not_structurally_retried(monkeypatch):
    example = load_synthetic_jsonl(FIXTURE)[0]
    calls = 0
    judge = AzureOpenAIJudge(
        endpoint="https://judge.example",
        deployment="judge",
        api_version="2025-04-01-preview",
        max_structured_retries=1,
    )

    async def truncated(_payload):
        nonlocal calls
        calls += 1
        return JudgeHTTPResponse(
            status=200,
            headers={},
            payload={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "", "refusal": None},
                    }
                ]
            },
        )

    monkeypatch.setattr(judge, "_post_once", truncated)
    with pytest.raises(JudgeGradingError) as exc_info:
        await judge.grade(example, "candidate")
    assert calls == 1
    assert exc_info.value.category == "response_truncated"


@pytest.mark.asyncio
async def test_entra_auth_uses_bearer_token_without_training_credentials(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_JUDGE_API_KEY", raising=False)

    class Credential:
        async def get_token(self, scope):
            assert scope == "https://cognitiveservices.azure.com/.default"
            return type("Token", (), {"token": "judge-token"})()

    judge = AzureOpenAIJudge(
        endpoint="https://judge.example",
        deployment="judge",
        api_version="2025-04-01-preview",
    )
    judge._credential = Credential()
    headers = await judge._request_headers()
    assert headers["Authorization"] == "Bearer judge-token"
    assert "api-key" not in headers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "expected_scope"),
    [
        (
            "https://judge.example",
            "https://cognitiveservices.azure.com/.default",
        ),
        (
            "https://judge.example/openai/v1",
            "https://ai.azure.com/.default",
        ),
    ],
)
async def test_outbound_entra_headers_use_protocol_scope_and_each_token(
    monkeypatch, endpoint, expected_scope
):
    monkeypatch.delenv("AZURE_OPENAI_JUDGE_API_KEY", raising=False)

    class Credential:
        def __init__(self):
            self.tokens = iter(("judge-token-one", "judge-token-two"))
            self.scopes = []

        async def get_token(self, scope):
            self.scopes.append(scope)
            return SimpleNamespace(token=next(self.tokens))

    class Response:
        status = 200
        headers = {}

        async def text(self):
            return "{}"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

    class Session:
        def __init__(self):
            self.request_headers = []

        def post(self, _url, *, headers, json):
            assert json == {"request": 1}
            self.request_headers.append(dict(headers))
            return Response()

    judge = AzureOpenAIJudge(
        endpoint=endpoint,
        deployment="judge",
        api_version="2025-04-01-preview",
    )
    credential = Credential()
    session = Session()
    judge._credential = credential
    judge._session = session

    await judge._post_once({"request": 1})
    await judge._post_once({"request": 1})

    assert credential.scopes == [
        expected_scope,
        expected_scope,
    ]
    assert session.request_headers == [
        {
            "Authorization": "Bearer judge-token-one",
            "Content-Type": "application/json",
        },
        {
            "Authorization": "Bearer judge-token-two",
            "Content-Type": "application/json",
        },
    ]
    assert all(
        "redacted" not in headers["Authorization"].lower()
        and "*" not in headers["Authorization"]
        for headers in session.request_headers
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    ["https://judge.example", "https://judge.example/openai/v1"],
)
async def test_outbound_api_key_headers_keep_existing_behavior(
    monkeypatch, endpoint
):
    api_key = "mock-judge-api-key"
    monkeypatch.setenv("AZURE_OPENAI_JUDGE_API_KEY", api_key)

    class Response:
        status = 200
        headers = {}

        async def text(self):
            return "{}"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

    captured = []

    class Session:
        def post(self, _url, *, headers, json):
            captured.append((dict(headers), json))
            return Response()

    judge = AzureOpenAIJudge(
        endpoint=endpoint,
        deployment="judge",
        api_version="2025-04-01-preview",
    )
    judge._session = Session()
    await judge._post_once({"request": 1})

    assert captured == [
        (
            {"api-key": api_key, "Content-Type": "application/json"},
            {"request": 1},
        )
    ]


def test_reference_is_isolated_from_policy_messages():
    example = load_synthetic_jsonl(FIXTURE)[0]
    policy_text = json.dumps(build_policy_messages(example, word_limit=200))
    judge_text = json.dumps(build_judge_messages(example, "candidate"))
    assert example.reference_summary not in policy_text
    assert example.reference_summary in judge_text
    assert example.text in policy_text
    assert example.text in judge_text


def test_deterministic_split_uses_only_train_without_leakage():
    examples = load_synthetic_jsonl(FIXTURE)
    first = deterministic_train_validation_split(
        examples, validation_size=2, seed=17
    )
    second = deterministic_train_validation_split(
        examples, validation_size=2, seed=17
    )
    assert [[item.example_id for item in split] for split in first] == [
        [item.example_id for item in split] for split in second
    ]
    train_ids = {item.example_id for item in first[0]}
    validation_ids = {item.example_id for item in first[1]}
    assert train_ids.isdisjoint(validation_ids)
    assert {item.source_split for item in first[0] + first[1]} == {"train"}


@pytest.mark.parametrize(
    ("row_update", "match"),
    [
        ({"source_split": None}, "non-train source_split"),
        ({"source_split": "test"}, "non-train source_split"),
        ({"source_split": "ca_test"}, "non-train source_split"),
    ],
)
def test_local_jsonl_rejects_missing_or_non_train_split(
    tmp_path, row_update, match
):
    row = {
        "source_split": "train",
        "source_index": 0,
        "title": "Synthetic title",
        "text": "Synthetic bill text.",
        "summary": "Synthetic summary.",
    }
    row.update(row_update)
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_synthetic_jsonl(path)


def test_local_jsonl_requires_explicit_source_split(tmp_path):
    path = tmp_path / "missing-split.jsonl"
    path.write_text(
        json.dumps(
            {
                "source_index": 0,
                "title": "Synthetic title",
                "text": "Synthetic bill text.",
                "summary": "Synthetic summary.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must declare source_split='train'"):
        load_synthetic_jsonl(path)


def test_overlength_rows_are_filtered_without_source_truncation():
    examples = load_synthetic_jsonl(FIXTURE)
    original_text = {example.example_id: example.text for example in examples}

    prepared = prepare_billsum_data(
        examples=examples,
        validation_size=2,
        seed=3,
        policy_token_count=lambda example: (
            1000 if example.source_index == 0 else 10
        ),
        judge_base_token_count=lambda example: (
            1000 if example.source_index == 1 else 10
        ),
        policy_prompt_token_budget=100,
        judge_prompt_token_budget=100,
        judge_candidate_token_reserve=10,
    )
    retained = prepared.train + prepared.validation
    assert {example.source_index for example in retained}.isdisjoint({0, 1})
    assert prepared.stats.rows_policy_overlength == 1
    assert prepared.stats.rows_judge_overlength == 1
    assert all(example.text == original_text[example.example_id] for example in retained)


@pytest.mark.asyncio
async def test_word_limit_is_mechanical_and_exposes_metrics():
    example = load_synthetic_jsonl(FIXTURE)[0]
    env = SummaryEnv(
        renderer=_FakeRenderer("one two three four"),
        example=example,
        word_limit=3,
        policy_max_tokens=2,
        dataset_metrics={},
    )
    result = await env.step([1, 2])
    assert result.metrics["summary/word_count"] == 4.0
    assert result.metrics["summary/word_limit_compliant"] == 0.0
    assert result.metrics["summary/word_limit_excess"] == 1.0
    assert result.metrics["policy/output_at_token_limit"] == 1.0

    builder = LLMJudgeGroupBuilder(
        renderer=_FakeRenderer(),
        example=example,
        judge=_ZeroJudge(),
        num_envs=1,
        word_limit=3,
        policy_max_tokens=2,
        dataset_metrics={},
    )
    scored = await builder.compute_group_rewards([], [env])
    assert scored[0][0] == 0.0
    assert scored[0][1]["summary/mechanical_limit_zeroed_reward"] == 1.0


@pytest.mark.asyncio
async def test_parse_failed_policy_output_cannot_receive_positive_reward():
    example = load_synthetic_jsonl(FIXTURE)[0]
    env = SummaryEnv(
        renderer=_FakeRenderer(
            "A plausible but truncated summary.",
            parse_success=False,
        ),
        example=example,
        word_limit=200,
        policy_max_tokens=32,
        dataset_metrics={},
    )
    step = await env.step([1, 2, 3])
    assert step.metrics["policy/parse_success"] == 0.0
    assert step.metrics["summary/word_limit_compliant"] == 1.0

    builder = LLMJudgeGroupBuilder(
        renderer=_FakeRenderer(),
        example=example,
        judge=_HighJudge(),
        num_envs=1,
        word_limit=200,
        policy_max_tokens=32,
        dataset_metrics={},
    )
    scored = await builder.compute_group_rewards([], [env])
    reward, metrics = scored[0]
    assert reward == 0.0
    assert metrics["judge/aggregate_reward"] == 1.0
    assert metrics["policy/eligibility_adjusted_reward"] == reward
    assert metrics["policy/parse_failure_zeroed_reward"] == 1.0
    assert metrics["policy/reward_eligible"] == 0.0
    assert metrics["summary/mechanical_limit_zeroed_reward"] == 0.0


@pytest.mark.asyncio
async def test_thinking_only_output_cannot_become_the_expected_summary():
    class ThinkingOnlyRenderer(_FakeRenderer):
        def parse_response(self, action):
            return (
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Hidden reasoning only."}
                    ],
                },
                True,
            )

    example = load_synthetic_jsonl(FIXTURE)[0]
    env = SummaryEnv(
        renderer=ThinkingOnlyRenderer(),
        example=example,
        word_limit=200,
        policy_max_tokens=512,
        dataset_metrics={},
    )
    step = await env.step([1, 2, 3])
    assert env.candidate == ""
    assert step.metrics["policy/parse_success"] == 0.0

    builder = LLMJudgeGroupBuilder(
        renderer=_FakeRenderer(),
        example=example,
        judge=_HighJudge(),
        num_envs=1,
        word_limit=200,
        policy_max_tokens=512,
        dataset_metrics={},
    )
    reward, metrics = (await builder.compute_group_rewards([], [env]))[0]
    assert reward == 0.0
    assert metrics["policy/reward_eligible"] == 0.0
    assert metrics["policy/eligibility_adjusted_reward"] == reward


def test_judge_credentials_are_absent_from_durable_metadata(monkeypatch, tmp_path):
    api_key = "not-for-metadata-api-key"
    access_token = "not-for-metadata-access-token"
    monkeypatch.setenv("AZURE_OPENAI_JUDGE_API_KEY", api_key)
    judge = AzureOpenAIJudge(
        endpoint="https://judge.example",
        deployment="judge-deployment",
        api_version="2025-04-01-preview",
    )
    config = CLIConfig(
        project_endpoint="https://training.example/api/projects/test",
        judge_endpoint="https://judge.example",
        judge_deployment="judge-deployment",
    )
    metadata = build_run_metadata(
        cli_config=config,
        run_name="test",
        log_path=str(tmp_path),
        tokenizer_name=config.model_name,
        renderer_name="qwen3_disable_thinking",
        judge=judge,
    )
    error = JudgeGradingError(
        "transport_exhausted",
        transport_retries=2,
        structured_retries=1,
        status=429,
    )
    durable = json.dumps(metadata) + repr(judge) + str(error) + repr(error)
    assert api_key not in durable
    assert access_token not in durable
    assert metadata["judge"]["api_key_env_var"] == "AZURE_OPENAI_JUDGE_API_KEY"
    assert metadata["judge"]["deployment"] == "judge-deployment"
    assert metadata["judge"]["max_concurrency"] == 8


def test_default_model_resolves_to_non_thinking_renderer():
    config = CLIConfig(project_endpoint="https://example.invalid/api/projects/test")
    assert config.model_name == "Qwen/Qwen3-32B"
    assert _resolve_renderer_name(config) == "qwen3_disable_thinking"


def test_renderer_override_and_other_model_resolution_are_preserved(monkeypatch):
    explicit = CLIConfig(
        project_endpoint="https://example.invalid/api/projects/test",
        renderer_name="custom_renderer",
    )
    assert _resolve_renderer_name(explicit) == "custom_renderer"

    monkeypatch.setattr(
        model_info, "get_recommended_renderer_name", lambda model: f"renderer:{model}"
    )
    other = CLIConfig(
        project_endpoint="https://example.invalid/api/projects/test",
        model_name="openai/gpt-oss-20b",
    )
    assert _resolve_renderer_name(other) == "renderer:openai/gpt-oss-20b"


@pytest.mark.asyncio
async def test_transport_error_is_sanitized_without_credential_cause(monkeypatch):
    api_key = "mock-key-that-must-not-leak"
    monkeypatch.setenv("AZURE_OPENAI_JUDGE_API_KEY", api_key)
    judge = AzureOpenAIJudge(
        endpoint="https://judge.example",
        deployment="judge",
        api_version="2025-04-01-preview",
        max_transport_retries=0,
    )

    async def fail(_payload):
        raise aiohttp.ClientConnectionError(f"transport failed with {api_key}")

    monkeypatch.setattr(judge, "_post_once", fail)
    with pytest.raises(JudgeGradingError) as exc_info:
        await judge._request_with_retries({})
    assert exc_info.value.__cause__ is None
    sanitized = str(exc_info.value) + repr(exc_info.value)
    assert api_key not in sanitized


@pytest.mark.asyncio
async def test_judge_drop_recorder_has_a_bounded_failure_budget(tmp_path):
    recorder = JudgeDropRecorder(
        tmp_path / "judge_failures.jsonl", max_failed_groups=0
    )
    error = IncompleteGroupGradingError("failed", trajectory_count=2)
    error.__cause__ = JudgeGradingError("transport_exhausted")
    with pytest.raises(RuntimeError, match="failure budget exceeded"):
        await recorder(GroupDropEvent(error=error, training_step=3))
    record = json.loads(
        (tmp_path / "judge_failures.jsonl").read_text(encoding="utf-8")
    )
    assert record == {
        "category": "transport_exhausted",
        "timestamp": record["timestamp"],
        "total_dropped_groups": 1,
        "training_step": 3,
        "trajectory_count": 2,
    }


@pytest.mark.asyncio
async def test_synthetic_fixture_loads_and_integrates_with_environment():
    examples = load_synthetic_jsonl(FIXTURE)
    assert len(examples) == 8
    assert len({example.example_id for example in examples}) == 8
    renderer = _FakeRenderer("A short synthetic summary.")
    env = SummaryEnv(
        renderer=renderer,
        example=examples[0],
        word_limit=200,
        policy_max_tokens=32,
        dataset_metrics={"data/train_retained": 6.0},
    )
    observation, _ = await env.initial_observation()
    assert observation.chunks[0].tokens == [1, 2, 3]
    policy_messages = renderer.last_messages
    assert examples[0].text in json.dumps(policy_messages)
    assert examples[0].reference_summary not in json.dumps(policy_messages)
    step = await env.step([7, 8])
    assert step.episode_done is True
    assert step.metrics["data/train_retained"] == 6.0
