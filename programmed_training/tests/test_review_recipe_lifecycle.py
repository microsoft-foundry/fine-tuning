"""Offline regressions for recipe resource ownership and vision configuration."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import chz
import datasets
import pytest
import torch

import azure.ai.finetuningsessions as sdk
import azure.ai.finetuningsessions.aio as async_sdk
from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

from programmed_training import image_processing_utils, tokenizer_utils
from programmed_training.recipes.tulu3_sft import train_azure as tulu
from programmed_training.recipes.visual_spatial import (
    train_rft_azure,
    train_sft_azure,
    training as vision,
)
from programmed_training.rl import train_azure as azure_adapter
from programmed_training.supervised import types as supervised_types


@pytest.fixture(autouse=True)
def offline_boundaries(monkeypatch):
    """Unexpected SDK construction or pretrained/data loading fails locally."""
    def forbidden(*args, **kwargs):
        raise AssertionError("offline recipe test crossed an unstubbed external boundary")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.delenv("LOOM_API_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_AI_API_KEY", raising=False)
    monkeypatch.setattr(sdk, "FineTuningSessionClient", forbidden)
    monkeypatch.setattr(sdk.FineTuningSession, "create", forbidden)
    monkeypatch.setattr(async_sdk, "FineTuningSessionClient", forbidden)
    monkeypatch.setattr(datasets, "load_dataset", forbidden)
    monkeypatch.setattr(tokenizer_utils, "get_tokenizer", forbidden)
    monkeypatch.setattr(supervised_types, "get_tokenizer", forbidden)
    monkeypatch.setattr(image_processing_utils, "get_image_processor", forbidden)
    monkeypatch.setattr(vision, "get_tokenizer", forbidden)
    monkeypatch.setattr(vision, "get_image_processor", forbidden)


_CLOSE_ORDER = [
    "poller.close",
    "session.close",
    "async_client.close",
    "sync_client.close",
    "async_credential.close",
    "sync_credential.close",
]


@dataclass
class _TuluProbe:
    config: tulu.CLIConfig
    failures: dict[str, BaseException] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)
    client_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    credential_options: list[tuple[bool, str | None]] = field(default_factory=list)
    create_options: dict[str, Any] = field(default_factory=dict)
    async_hooks: dict[str, Callable[[], Awaitable[None]]] = field(default_factory=dict)
    create_hook: Callable[[], None] | None = None
    session_close_hook: Callable[[], None] | None = None
    uses_token_credential: bool = True
    prepared_datasets: tuple = field(default_factory=lambda: ([], None))
    received_datasets: tuple | None = None

    def hit(self, event: str) -> None:
        self.events.append(event)
        error = self.failures.get(event)
        if error is not None:
            raise error

    async def async_hit(self, event: str) -> None:
        self.hit(event)
        hook = self.async_hooks.get(event)
        if hook is not None:
            await hook()

    @property
    def closes(self) -> list[str]:
        return [event for event in self.events if event.endswith(".close")]


@pytest.fixture
def tulu_runtime(monkeypatch, tmp_path):
    probe = _TuluProbe(
        config=tulu.CLIConfig(
            project_endpoint="https://example.invalid/api/projects/review",
            model_name="Qwen/Qwen3-32B",
            tokenizer_name="review/tokenizer-only",
            renderer_name="qwen3",
            log_path=str(tmp_path / "tulu"),
            behavior_if_log_dir_exists="raise",
            training_type="GlobalStandard",
            user_metadata={"experimentName": "offline-review"},
        )
    )

    class SyncCredential:
        __slots__ = ()

        def close(self):
            probe.hit("sync_credential.close")

    class AsyncCredential:
        __slots__ = ()

        async def close(self):
            await probe.async_hit("async_credential.close")

    class KeyCredential:
        __slots__ = ()

    sync_credential = SyncCredential()
    async_credential = AsyncCredential()
    key_credential = KeyCredential()

    def build_credentials(*, direct_path, azure_api_key):
        probe.hit("credentials")
        probe.credential_options.append((direct_path, azure_api_key))
        if probe.uses_token_credential:
            return sync_credential, async_credential, True
        return key_credential, key_credential, False

    class SyncClient:
        __slots__ = ()

        def __init__(
            self, *, endpoint, credential, credential_scopes=None,
            allow_insecure_http=False,
        ):
            probe.hit("sync_client.construct")
            expected = sync_credential if probe.uses_token_credential else key_credential
            assert credential is expected
            probe.client_options["sync"] = {
                "endpoint": endpoint,
                "credential_scopes": credential_scopes,
                "allow_insecure_http": allow_insecure_http,
            }

        def close(self):
            probe.hit("sync_client.close")

    class Session:
        __slots__ = ()
        session_id = "session_0123abcd"

        @classmethod
        def create(
            cls, client, *, base_model, lora_config, type,
            from_checkpoint, timeout_sec, user_metadata, training_type,
        ):
            assert isinstance(client, SyncClient)
            probe.hit("session.create")
            probe.create_options.update(
                base_model=base_model,
                rank=lora_config.rank,
                type=type,
                from_checkpoint=from_checkpoint,
                timeout_sec=timeout_sec,
                user_metadata=user_metadata,
                training_type=training_type,
            )
            if probe.create_hook is not None:
                probe.create_hook()
            return cls()

        def close(self):
            probe.hit("session.close")
            if probe.session_close_hook is not None:
                probe.session_close_hook()

    class AsyncClient:
        __slots__ = ()

        def __init__(
            self, *, endpoint, credential, credential_scopes=None,
            allow_insecure_http=False,
        ):
            probe.hit("async_client.construct")
            expected = async_credential if probe.uses_token_credential else key_credential
            assert credential is expected
            probe.client_options["async"] = {
                "endpoint": endpoint,
                "credential_scopes": credential_scopes,
                "allow_insecure_http": allow_insecure_http,
            }

        async def close(self):
            await probe.async_hit("async_client.close")

    tokenizer = object()

    def get_tokenizer(name):
        assert name == probe.config.tokenizer_name
        probe.hit("tokenizer")
        return tokenizer

    class TrainingClient:
        __slots__ = ()

        def __init__(self, client, session_id, supplied_tokenizer):
            assert isinstance(client, AsyncClient)
            assert session_id == Session.session_id
            assert supplied_tokenizer is tokenizer
            probe.hit("training_client.construct")

        async def close_poller(self):
            await probe.async_hit("poller.close")

    async def train_main(config, training_client, *, prepared_datasets=None):
        assert isinstance(training_client, TrainingClient)
        assert config.model_name == probe.config.model_name
        probe.received_datasets = prepared_datasets
        await probe.async_hit("train")

    def prepare_datasets(config):
        assert config.preflight_dataset
        probe.hit("preflight")
        return probe.prepared_datasets

    monkeypatch.setattr(tulu, "_build_session_credentials", build_credentials)
    monkeypatch.setattr(sdk, "FineTuningSessionClient", SyncClient)
    monkeypatch.setattr(sdk, "FineTuningSession", Session)
    monkeypatch.setattr(async_sdk, "FineTuningSessionClient", AsyncClient)
    monkeypatch.setattr(tokenizer_utils, "get_tokenizer", get_tokenizer)
    monkeypatch.setattr(azure_adapter, "AzureSDKTrainingClient", TrainingClient)
    monkeypatch.setattr(tulu.train, "main", train_main)
    monkeypatch.setattr(tulu.train, "prepare_datasets", prepare_datasets)
    return probe


@pytest.mark.asyncio
@pytest.mark.parametrize("training_type", [None, "GlobalStandard"])
@pytest.mark.parametrize("preflight", [False, True])
async def test_tulu_success_closes_all_owned_resources(tulu_runtime, training_type, preflight):
    probe = tulu_runtime
    probe.config = chz.replace(
        probe.config, training_type=training_type, preflight_dataset=preflight,
    )

    await tulu.cli_main(probe.config)

    assert probe.closes == _CLOSE_ORDER
    assert probe.create_options["training_type"] == training_type
    assert probe.create_options["user_metadata"] == probe.config.user_metadata
    assert probe.create_options["base_model"] == probe.config.model_name
    assert probe.create_options["rank"] == probe.config.lora_rank
    assert probe.received_datasets is (probe.prepared_datasets if preflight else None)


@pytest.mark.asyncio
@pytest.mark.parametrize("joins_needed", [2, 3])
@pytest.mark.parametrize("primary_type", [None, ValueError, asyncio.CancelledError])
async def test_tulu_drains_heartbeat_and_completes_before_disposing_clients(
    tulu_runtime, joins_needed, primary_type,
):
    # Use the actual source SDK close/stop implementation, without a real thread,
    # sleep, transport, session creation, or cloud call. The SDK's first join
    # expires before an in-flight heartbeat finishes; the next join can drain it.
    from azure.ai.finetuningsessions._patch import FineTuningSession

    probe = tulu_runtime
    owner = object.__new__(FineTuningSession)
    owner._resource_session_id = "session_0123abcd"
    owner._heartbeat_stop = threading.Event()
    joins = []
    completions = []

    class InFlightHeartbeat:
        def join(self, timeout):
            assert owner._heartbeat_stop.is_set()
            assert "sync_client.close" not in probe.events
            joins.append(timeout)

        def is_alive(self):
            return len(joins) < joins_needed

    def send_request(request):
        assert "sync_client.close" not in probe.events
        assert owner._heartbeat_thread is None
        assert request.method == "POST" and request.url.split("?", 1)[0].endswith("/complete")
        completions.append(request)
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None)

    owner._heartbeat_thread = InFlightHeartbeat()
    owner._client = SimpleNamespace(send_request=send_request)
    probe.session_close_hook = owner.close
    if primary_type is None:
        await tulu.cli_main(probe.config)
    else:
        primary = primary_type("training failed")
        probe.failures["train"] = primary
        with pytest.raises(primary_type) as caught:
            await tulu.cli_main(probe.config)
        assert caught.value is primary

    assert joins == [5.0] * joins_needed
    assert len(completions) == 1
    assert probe.closes == ["poller.close", *(["session.close"] * joins_needed), *_CLOSE_ORDER[2:]]


def test_tulu_heartbeat_drain_retries_are_bounded():
    from unittest.mock import Mock

    error = RuntimeError(tulu._HEARTBEAT_STILL_RUNNING)
    session = SimpleNamespace(close=Mock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        tulu._close_owned_session(session)
    assert caught.value is error
    assert session.close.call_count == tulu._SESSION_CLOSE_HEARTBEAT_ATTEMPTS


@pytest.mark.parametrize("error", [
    RuntimeError("completion response lost"), OSError("completion transport failed"),
    asyncio.CancelledError("operation cancelled"),
])
def test_tulu_never_replays_unknown_or_ambiguous_completion_failures(error):
    from unittest.mock import Mock

    session = SimpleNamespace(close=Mock(side_effect=error))
    with pytest.raises(type(error)) as caught:
        tulu._close_owned_session(session)
    assert caught.value is error
    session.close.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "expected_closes"),
    [
        ("sync_client.construct", _CLOSE_ORDER[4:]),
        ("session.create", _CLOSE_ORDER[3:]),
        ("async_client.construct", [_CLOSE_ORDER[1], *_CLOSE_ORDER[3:]]),
        ("tokenizer", _CLOSE_ORDER[1:]),
        ("training_client.construct", _CLOSE_ORDER[1:]),
        ("train", _CLOSE_ORDER),
    ],
)
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_tulu_initialization_and_workload_failures_release_ownership(
    tulu_runtime, stage, expected_closes, error_type,
):
    probe = tulu_runtime
    primary = error_type(stage)
    probe.failures[stage] = primary

    with pytest.raises(error_type) as caught:
        await tulu.cli_main(probe.config)

    assert caught.value is primary
    assert probe.closes == expected_closes


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_stage", _CLOSE_ORDER)
@pytest.mark.parametrize("primary_type", [None, ValueError, asyncio.CancelledError])
@pytest.mark.parametrize("cleanup_type", [RuntimeError, asyncio.CancelledError])
async def test_tulu_cleanup_attempts_every_close_and_preserves_primary(
    tulu_runtime, cleanup_stage, primary_type, cleanup_type,
):
    probe = tulu_runtime
    cleanup_error = cleanup_type(cleanup_stage)
    probe.failures[cleanup_stage] = cleanup_error
    primary = primary_type("training failed") if primary_type is not None else cleanup_error
    if primary_type is not None:
        probe.failures["train"] = primary

    with pytest.raises(type(primary)) as caught:
        await tulu.cli_main(probe.config)

    assert caught.value is primary
    assert probe.closes == _CLOSE_ORDER


@pytest.mark.asyncio
async def test_tulu_metadata_failure_after_creation_still_closes_session(
    tulu_runtime, monkeypatch,
):
    probe = tulu_runtime
    primary = OSError("metadata unavailable")
    original_dump = tulu.json.dump

    def dump(metadata, output, *, indent, default):
        if metadata["azure"]["session_id"] is not None:
            raise primary
        return original_dump(metadata, output, indent=indent, default=default)

    monkeypatch.setattr(tulu.json, "dump", dump)
    with pytest.raises(OSError) as caught:
        await tulu.cli_main(probe.config)

    assert caught.value is primary
    assert "async_client.construct" not in probe.events
    assert probe.closes == ["session.close", *_CLOSE_ORDER[3:]]


@pytest.mark.asyncio
async def test_tulu_cancellation_during_executor_creation_retains_session(tulu_runtime):
    probe = tulu_runtime
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()

    def pause_creation():
        loop.call_soon_threadsafe(started.set)
        if not release.wait(5):
            raise TimeoutError("test did not release session creation")

    probe.create_hook = pause_creation
    task = asyncio.create_task(tulu.cli_main(probe.config))
    try:
        await asyncio.wait_for(started.wait(), 5)
        task.cancel("stop during acquisition")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert "async_client.construct" not in probe.events
    assert probe.closes == ["session.close", *_CLOSE_ORDER[3:]]


@pytest.mark.asyncio
async def test_tulu_second_cancellation_does_not_abandon_cleanup(tulu_runtime):
    probe = tulu_runtime
    training_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release = asyncio.Event()

    async def training():
        training_started.set()
        await asyncio.Event().wait()

    async def cleanup():
        cleanup_started.set()
        await release.wait()

    probe.async_hooks.update({"train": training, "poller.close": cleanup})
    task = asyncio.create_task(tulu.cli_main(probe.config))
    try:
        await asyncio.wait_for(training_started.wait(), 5)
        task.cancel("original cancellation")
        await asyncio.wait_for(cleanup_started.wait(), 5)
        task.cancel("second cancellation")
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(task, 5)
        assert caught.value.args == ("original cancellation",)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert probe.closes == _CLOSE_ORDER


@pytest.mark.asyncio
async def test_tulu_key_credential_has_no_owned_close(tulu_runtime, monkeypatch):
    probe = tulu_runtime
    probe.uses_token_credential = False
    monkeypatch.setenv("AZURE_AI_API_KEY", "offline-test-key")

    await tulu.cli_main(probe.config)

    assert probe.closes == _CLOSE_ORDER[:4]
    assert probe.client_options["sync"]["credential_scopes"] is None


@pytest.mark.asyncio
async def test_tulu_direct_route_and_checkpoint_contracts_are_preserved(tulu_runtime, monkeypatch):
    probe = tulu_runtime
    endpoint = "http://127.0.0.1:8000"
    route_calls = []

    class DirectApi:
        DIRECT_TOKEN_SCOPE = "api://review/.default"

        @staticmethod
        def resolve_endpoint(project_endpoint):
            assert project_endpoint == probe.config.project_endpoint
            route_calls.append("resolve")
            return endpoint

        @staticmethod
        def preflight(resolved_endpoint):
            assert resolved_endpoint == endpoint
            route_calls.append("preflight")

        @staticmethod
        def allow_insecure_http(resolved_endpoint):
            assert resolved_endpoint == endpoint
            return True

    def import_direct_api(name):
        assert name == "utils.direct_api"
        return DirectApi

    monkeypatch.setenv("LOOM_API_ENDPOINT", endpoint)
    monkeypatch.setenv("AZURE_AI_API_KEY", "ignored-offline-key")
    monkeypatch.setattr(tulu, "import_module", import_direct_api)
    probe.config = chz.replace(probe.config, load_checkpoint_path="model_source/final")

    await tulu.cli_main(probe.config)

    assert route_calls == ["resolve", "preflight"]
    assert probe.credential_options == [(True, None)]
    assert probe.client_options["sync"] == probe.client_options["async"] == {
        "endpoint": endpoint,
        "credential_scopes": [DirectApi.DIRECT_TOKEN_SCOPE],
        "allow_insecure_http": True,
    }
    source = probe.create_options["from_checkpoint"]
    assert (source.source_session_id, source.checkpoint_id) == ("session_source", "final")
    assert probe.closes == _CLOSE_ORDER


@pytest.mark.asyncio
async def test_tulu_preflight_only_does_not_construct_owned_resources(tulu_runtime):
    probe = tulu_runtime
    await tulu.cli_main(chz.replace(probe.config, preflight_only=True))
    assert probe.events == ["preflight"]


@pytest.mark.parametrize("close_fails", [False, True])
def test_tulu_async_credential_construction_failure_closes_sync_credential(
    monkeypatch, close_fails,
):
    import azure.identity
    import azure.identity.aio

    primary = RuntimeError("async credential construction failed")
    closes = []

    class SyncCredential:
        __slots__ = ()

        def close(self):
            closes.append("sync")
            if close_fails:
                raise OSError("sync credential close failed")

    def async_credential():
        raise primary

    monkeypatch.setattr(azure.identity, "DefaultAzureCredential", SyncCredential)
    monkeypatch.setattr(azure.identity.aio, "DefaultAzureCredential", async_credential)
    with pytest.raises(RuntimeError) as caught:
        tulu._build_session_credentials(direct_path=True, azure_api_key=None)

    assert caught.value is primary
    assert closes == ["sync"]


_VISION_MODEL = "review/vision-model"
_VISION_TOKENIZER = "review/tokenizer-only"


@dataclass
class _VisionProbe:
    tokenizer_names: list[str] = field(default_factory=list)
    processor_names: list[str] = field(default_factory=list)
    renderers: list[Any] = field(default_factory=list)


@pytest.fixture
def vision_runtime(monkeypatch):
    probe = _VisionProbe()
    source = datasets.Dataset.from_dict({
        "question": ["first?", "second?"],
        "answer": ["1", "1"],
        "image": ["first-image", "second-image"],
    })

    class Tokenizer:
        __slots__ = ()

    class Processor:
        __slots__ = ()

    class Renderer:
        __slots__ = ("image_cache_dir", "name", "tokenizer", "processor")

        def __init__(self, name, tokenizer, processor):
            self.image_cache_dir = None
            self.name = name
            self.tokenizer = tokenizer
            self.processor = processor

        def build_supervised_example(self, messages, *, train_on_what):
            assert messages[-1] == {"role": "assistant", "content": "1"}
            assert train_on_what is not None
            return self.build_generation_prompt(messages), torch.ones(4)

        def build_generation_prompt(self, messages):
            assert messages[0]["role"] == "user"
            return ModelInput(chunks=[ModelInputChunk(tokens=[10, 11, 12, 13])])

        def get_stop_sequences(self):
            return ["stop"]

        def parse_response(self, tokens):
            assert tokens == [1]
            return {"role": "assistant", "content": "1"}, True

    def load_dataset(seed, data_path, image_cache_dir=None):
        assert seed == 0
        assert data_path is None
        return source, source

    def get_tokenizer(name):
        assert name in {_VISION_MODEL, _VISION_TOKENIZER}
        probe.tokenizer_names.append(name)
        return Tokenizer()

    def get_processor(name):
        assert name == _VISION_MODEL, "a tokenizer-only repository is not an image processor"
        probe.processor_names.append(name)
        return Processor()

    def get_renderer(name, tokenizer, *, image_processor):
        assert isinstance(tokenizer, Tokenizer)
        assert isinstance(image_processor, Processor)
        renderer = Renderer(name, tokenizer, image_processor)
        probe.renderers.append(renderer)
        return renderer

    def require_vision_language_model(model_name, *, workload):
        assert model_name == _VISION_MODEL
        assert workload.startswith("visual-spatial")

    monkeypatch.setattr(vision, "_load_dataset", load_dataset)
    monkeypatch.setattr(vision, "get_tokenizer", get_tokenizer)
    monkeypatch.setattr(supervised_types, "get_tokenizer", get_tokenizer)
    monkeypatch.setattr(vision, "get_image_processor", get_processor)
    monkeypatch.setattr(vision, "get_renderer", get_renderer)
    monkeypatch.setattr(
        train_sft_azure.model_info, "require_vision_language_model",
        require_vision_language_model,
    )
    return probe


def test_vision_sft_renderer_is_builder_local_and_uses_base_model(vision_runtime, tmp_path):
    probe = vision_runtime
    common = supervised_types.ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=_VISION_TOKENIZER,
        renderer_name="first-renderer",
        max_length=64,
        batch_size=1,
    )
    first = vision.VisualSpatialSftBuilder(
        model_name=_VISION_MODEL,
        common_config=common,
        image_cache_dir=str(tmp_path / "first-cache"),
    )
    dataset, _ = first.build()
    for _ in range(2):
        for index in range(len(dataset)):
            dataset.get_batch(index)

    assert probe.processor_names == [_VISION_MODEL]
    assert probe.tokenizer_names == [_VISION_TOKENIZER]
    assert len(probe.renderers) == 1
    assert first.renderer is probe.renderers[0]

    second = vision.VisualSpatialSftBuilder(
        model_name=_VISION_MODEL,
        common_config=chz.replace(common, renderer_name="second-renderer"),
        image_cache_dir=str(tmp_path / "second-cache"),
    )
    second.build()[0].get_batch(0)

    assert second.renderer is not first.renderer
    assert first.renderer.image_cache_dir == str(tmp_path / "first-cache")
    assert second.renderer.image_cache_dir == str(tmp_path / "second-cache")
    assert [renderer.name for renderer in probe.renderers] == [
        "first-renderer", "second-renderer",
    ]
    assert probe.processor_names == [_VISION_MODEL, _VISION_MODEL]
    assert probe.tokenizer_names == [_VISION_TOKENIZER, _VISION_TOKENIZER]


@pytest.mark.parametrize(
    "model_options", [{}, {"model_name": None}], ids=["omitted", "explicit-none"],
)
def test_vision_sft_builder_retains_legacy_model_default(
    vision_runtime, tmp_path, model_options,
):
    probe = vision_runtime
    common = supervised_types.ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=_VISION_MODEL,
        renderer_name="legacy-renderer",
        max_length=64,
        batch_size=1,
    )
    builder = vision.VisualSpatialSftBuilder(
        common_config=common,
        image_cache_dir=str(tmp_path / "legacy-cache"),
        **model_options,
    )

    assert builder.model_name is None
    for _ in range(2):
        dataset, _ = builder.build()
        for index in range(len(dataset)):
            dataset.get_batch(index)

    assert probe.processor_names == [_VISION_MODEL]
    assert probe.tokenizer_names == [_VISION_MODEL]
    assert len(probe.renderers) == 1
    assert builder.renderer is probe.renderers[0]
    assert builder.renderer.image_cache_dir == str(tmp_path / "legacy-cache")


@dataclass(slots=True)
class _Sequence:
    tokens: list[int]


@dataclass(slots=True)
class _Sample:
    sequences: list[_Sequence]


@pytest.mark.asyncio
@pytest.mark.parametrize("tokenizer_name", [None, _VISION_TOKENIZER])
async def test_vision_evaluator_override_and_legacy_positionals(
    vision_runtime, tmp_path, tokenizer_name,
):
    options = {} if tokenizer_name is None else {"tokenizer_name": tokenizer_name}
    evaluator = vision.VisualSpatialAccuracyEvaluator(
        _VISION_MODEL, "review-renderer", 0, str(tmp_path), 1,
        **options,
    )
    parameter = inspect.signature(vision.VisualSpatialAccuracyEvaluator).parameters["tokenizer_name"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
    assert evaluator.preflight(model_context_length=1024).examples == 1

    class Sampler:
        __slots__ = ()

        async def sample_async(self, prompt, *, sampling_params):
            assert prompt.chunks[0].tokens == [10, 11, 12, 13]
            assert sampling_params.max_tokens == evaluator.max_tokens
            return _Sample([_Sequence([1])])

    metrics = await evaluator(Sampler())

    assert metrics["test/visual_spatial_accuracy"] == 1.0
    assert vision_runtime.tokenizer_names == [tokenizer_name or _VISION_MODEL]
    assert vision_runtime.processor_names == [_VISION_MODEL]


@pytest.mark.asyncio
@pytest.mark.parametrize("recipe", [train_sft_azure, train_rft_azure], ids=["sft", "rft"])
async def test_vision_entrypoints_forward_tokenizer_to_training_and_evaluation(
    vision_runtime, monkeypatch, tmp_path, recipe,
):
    recorded = []
    owner = recipe.train if recipe is train_sft_azure else recipe
    original_config = owner.Config

    def capture_config(**kwargs):
        config = original_config(**kwargs)
        recorded.append(config)
        return config

    monkeypatch.setattr(owner, "Config", capture_config)
    options = dict(
        project_endpoint="https://example.invalid/api/projects/review",
        model_name=_VISION_MODEL,
        tokenizer_name=_VISION_TOKENIZER,
        renderer_name="review-renderer",
        model_context_length=1024,
        log_path=str(tmp_path / "vision"),
        preflight_only=True,
    )
    if recipe is train_sft_azure:
        options["batch_size"] = 1
    await recipe.cli_main(recipe.CLIConfig(**options))

    assert len(recorded) == 1
    if recipe is train_sft_azure:
        assert recorded[0].dataset_builder.model_name == _VISION_MODEL
        assert (
            recorded[0].dataset_builder.common_config.model_name_for_tokenizer
            == _VISION_TOKENIZER
        )
    evaluator = recorded[0].evaluator_builders[0]()
    assert evaluator.tokenizer_name == _VISION_TOKENIZER
    assert evaluator.model_name == _VISION_MODEL
    assert evaluator.preflight(model_context_length=1024).examples == 2
    assert vision_runtime.tokenizer_names == [_VISION_TOKENIZER, _VISION_TOKENIZER]
    assert vision_runtime.processor_names == [_VISION_MODEL, _VISION_MODEL]