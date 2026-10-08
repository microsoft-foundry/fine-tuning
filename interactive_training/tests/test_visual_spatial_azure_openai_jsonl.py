import asyncio
import base64
import io
import json

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from azure.ai.finetuningsessions.models import ModelInputChunk
from PIL import Image

from interactive_training.recipes.visual_spatial import azure_openai_jsonl, train_rft_azure
from interactive_training.recipes.visual_spatial import training
from interactive_training.renderers import TrainOnWhat, base as renderer_base
from interactive_training.renderers.muse_glimmer import MuseGlimmerRenderer
from interactive_training.supervised.data import conversation_to_datum


class ReversibleTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8")


class ImageProcessor:
    merge_size = 1

    def get_number_of_image_patches(self, height, width, images_kwargs=None):
        assert (height, width) == (2, 3)
        return 6


def _conversation(image_url: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Answer briefly."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What defect is visible?"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
            {"role": "assistant", "content": answer},
        ]
    }


def _multi_image_conversation(image_urls: list[str], answer: str) -> dict:
    row = _conversation(image_urls[0], answer)
    content = row["messages"][1]["content"]
    content.extend(
        {"type": "image_url", "image_url": {"url": image_url}}
        for image_url in image_urls[1:]
    )
    return row


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (3, 2), color="red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_load_dataset_adapts_url_and_data_uri_conversations(tmp_path) -> None:
    rows = [
        _conversation("https://example.com/part.png", "crack"),
        _conversation("data:image/png;base64,iVBORw0KGgo=", "dent"),
    ]
    data_path = tmp_path / "images.jsonl"
    data_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    train, test = azure_openai_jsonl.load_dataset(
        str(data_path),
        seed=7,
        test_size=1,
    )
    adapted = [*train, *test]

    assert {row["answer"] for row in adapted} == {"crack", "dent"}
    assert {row["question"] for row in adapted} == {"What defect is visible?"}
    assert all(row["prompt_messages"][0]["role"] == "system" for row in adapted)
    assert all(row["prompt_messages"][-1]["role"] == "user" for row in adapted)
    image_urls = {
        part["image_url"]["url"]
        for row in adapted
        for part in row["prompt_messages"][-1]["content"]
        if part["type"] == "image_url"
    }
    assert image_urls == {
        "https://example.com/part.png",
        "data:image/png;base64,iVBORw0KGgo=",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("source_kind", ["public_url", "data_uri"])
async def test_train_preflight_with_data_path_renders_image_source(
    monkeypatch,
    tmp_path,
    source_kind,
) -> None:
    image_data = _png_bytes()
    downloaded_urls = []
    if source_kind == "public_url":
        image_url = "https://example.test/part.png"
        monkeypatch.setattr(
            "interactive_training.renderers.base._download_image_reference",
            lambda url: downloaded_urls.append(url) or image_data,
        )
    else:
        image_url = "data:image/png;base64," + base64.b64encode(image_data).decode()

    data_path = tmp_path / f"{source_kind}.jsonl"
    data_path.write_text(
        "".join(
            json.dumps(_conversation(image_url, answer)) + "\n"
            for answer in ("crack", "dent")
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        train_rft_azure.model_info,
        "require_vision_language_model",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(training, "get_tokenizer", lambda _name: ReversibleTokenizer())
    monkeypatch.setattr(training, "get_image_processor", lambda _name: ImageProcessor())

    await train_rft_azure.cli_main(
        train_rft_azure.CLIConfig(
            project_endpoint="https://example.test",
            model_name="example/vision-language-model",
            renderer_name="muse_glimmer",
            model_context_length=4096,
            data_path=str(data_path),
            log_path=str(tmp_path / "run"),
            preflight_only=True,
            max_tokens=64,
            eval_max_tokens=64,
        )
    )

    if source_kind == "public_url":
        assert downloaded_urls == [image_url]
    else:
        assert downloaded_urls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("source_kind", ["public_url", "data_uri"])
async def test_data_path_prepares_image_once_across_ten_epochs(
    monkeypatch,
    tmp_path,
    source_kind,
) -> None:
    image_data = _png_bytes()
    if source_kind == "public_url":
        image_url = "https://example.test/part.png"
    else:
        image_url = "data:image/png;base64," + base64.b64encode(image_data).decode()

    data_path = tmp_path / f"cached-{source_kind}.jsonl"
    data_path.write_text(
        "".join(
            json.dumps(_conversation(image_url, answer)) + "\n"
            for answer in ("crack", "dent")
        ),
        encoding="utf-8",
    )

    loaded_sources = []
    original_load_image_reference = renderer_base._load_image_reference

    def load_image_reference(source, **kwargs):
        loaded_sources.append(source)
        if source_kind == "public_url":
            return image_data
        return original_load_image_reference(source, **kwargs)

    monkeypatch.setattr(renderer_base, "_load_image_reference", load_image_reference)
    monkeypatch.setattr(training, "get_tokenizer", lambda _name: ReversibleTokenizer())
    monkeypatch.setattr(training, "get_image_processor", lambda _name: ImageProcessor())

    builder = training.VisualSpatialRLDatasetBuilder(
        batch_size=1,
        model_name="example/vision-language-model",
        tokenizer_name="example/vision-language-model",
        renderer_name="muse_glimmer",
        group_size=5,
        seed=0,
        num_epochs=10,
        data_path=str(data_path),
    )
    train_dataset, _ = await builder()

    await train_dataset.preflight(max_tokens=64, model_context_length=4096)
    for batch_index in range(len(train_dataset)):
        group = train_dataset.get_batch(batch_index)[0]
        environments = await group.make_envs()
        await asyncio.gather(
            *(environment.initial_observation() for environment in environments)
        )

    expected_loads = 1 if source_kind == "public_url" else 3
    assert loaded_sources == [image_url] * expected_loads


@pytest.mark.parametrize(
    ("messages", "error"),
    [
        (
            [{"role": "user", "content": "Where is the defect?"}],
            "final message must contain the assistant",
        ),
        (
            [
                {"role": "user", "content": "Where is the defect?"},
                {"role": "assistant", "content": "left"},
            ],
            "must contain an image_url",
        ),
    ],
)
def test_adapt_conversation_rejects_missing_contract(messages, error) -> None:
    with pytest.raises(ValueError, match=error):
        azure_openai_jsonl.adapt_conversation({"messages": messages})


def test_adapter_owns_reward_normalization() -> None:
    assert azure_openai_jsonl.answers_match("  CRACK\n", "crack")
    assert not azure_openai_jsonl.answers_match("small crack", "crack")


def test_prepare_image_cache_enforces_aggregate_encoded_byte_limit(
    monkeypatch,
    tmp_path,
) -> None:
    image_data = _png_bytes()
    row = azure_openai_jsonl.adapt_conversation(
        _multi_image_conversation(
            ["https://example.test/one.png", "https://example.test/two.png"],
            "crack",
        )
    )
    monkeypatch.setattr(
        renderer_base,
        "_load_image_reference",
        lambda *_args, **_kwargs: image_data,
    )
    monkeypatch.setattr(
        azure_openai_jsonl,
        "MAX_TOTAL_IMAGE_BYTES_PER_EXAMPLE",
        len(image_data) * 2 - 1,
    )

    with pytest.raises(ValueError, match="aggregate encoded image data"):
        azure_openai_jsonl.prepare_image_cache([row], str(tmp_path))


def test_prepare_image_cache_enforces_aggregate_decoded_pixel_limit(
    monkeypatch,
    tmp_path,
) -> None:
    image_data = _png_bytes()
    row = azure_openai_jsonl.adapt_conversation(
        _multi_image_conversation(
            ["https://example.test/one.png", "https://example.test/two.png"],
            "crack",
        )
    )
    monkeypatch.setattr(
        renderer_base,
        "_load_image_reference",
        lambda *_args, **_kwargs: image_data,
    )
    monkeypatch.setattr(
        azure_openai_jsonl,
        "MAX_TOTAL_IMAGE_PIXELS_PER_EXAMPLE",
        11,
    )

    with pytest.raises(ValueError, match="aggregate decoded image size"):
        azure_openai_jsonl.prepare_image_cache([row], str(tmp_path))


def test_rendered_visual_example_can_overfit_with_adapter_only() -> None:
    torch.manual_seed(7)
    image_url = "data:image/png;base64," + base64.b64encode(_png_bytes()).decode()
    row = azure_openai_jsonl.adapt_conversation(_conversation(image_url, "crack"))
    renderer = MuseGlimmerRenderer(
        ReversibleTokenizer(),
        image_processor=ImageProcessor(),
        current_date="2026-08-30",
    )
    datum = conversation_to_datum(
        azure_openai_jsonl.sft_messages(row),
        renderer,
        max_length=None,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )

    input_tokens = []
    for chunk in datum.model_input.chunks:
        input_tokens.extend(
            chunk.tokens
            if isinstance(chunk, ModelInputChunk)
            else [0] * chunk.length
        )
    inputs = torch.tensor(input_tokens, dtype=torch.long)
    targets = torch.tensor(
        datum.loss_fn_inputs["target_tokens"].data,
        dtype=torch.long,
    )
    weights = torch.tensor(datum.loss_fn_inputs["weights"].data) > 0

    assert inputs.shape == targets.shape == weights.shape
    assert weights.any()

    class TinyAdapterModel(nn.Module):
        def __init__(self, sequence_length: int) -> None:
            super().__init__()
            hidden_size = 32
            adapter_rank = 16
            self.token_embedding = nn.Embedding(256, hidden_size)
            self.position_embedding = nn.Embedding(sequence_length, hidden_size)
            self.base_head = nn.Linear(hidden_size, 256, bias=False)
            self.adapter_down = nn.Linear(hidden_size, adapter_rank, bias=False)
            self.adapter_up = nn.Linear(adapter_rank, 256, bias=False)
            nn.init.zeros_(self.adapter_up.weight)
            for parameter in (
                *self.token_embedding.parameters(),
                *self.position_embedding.parameters(),
                *self.base_head.parameters(),
            ):
                parameter.requires_grad_(False)

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            positions = torch.arange(token_ids.shape[0])
            hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
            return self.base_head(hidden) + self.adapter_up(self.adapter_down(hidden))

    model = TinyAdapterModel(len(inputs))
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    adapter_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.08,
    )

    def supervised_loss() -> torch.Tensor:
        return F.cross_entropy(model(inputs)[weights], targets[weights])

    initial_loss = supervised_loss().item()
    saw_adapter_gradient = False
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        loss = supervised_loss()
        loss.backward()
        saw_adapter_gradient = saw_adapter_gradient or any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        optimizer.step()

    final_loss = supervised_loss().item()
    predictions = model(inputs).argmax(dim=-1)
    assert saw_adapter_gradient
    assert final_loss < initial_loss * 0.1
    assert torch.equal(predictions[weights], targets[weights])
    assert all(
        torch.equal(parameter, frozen_before[name])
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    )
    assert any(
        not torch.equal(parameter, adapter_before[name])
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
