import io

from azure.ai.finetuningsessions.models import Datum, ModelInput, ModelInputChunk, TensorData
from PIL import Image

from interactive_training.image_types import ImageChunk
from interactive_training.supervised.train import _datum_token_count as _training_datum_token_count
from interactive_training.supervised.nll_evaluator import (
    _datum_loss_token_count,
    _datum_token_count,
)


def test_datum_token_counts_use_supervised_target_fields():
    datum = Datum(
        model_input=ModelInput(chunks=[ModelInputChunk(tokens=[10, 20])]),
        loss_fn_inputs={
            "target_tokens": TensorData(data=[20, 30]),
            "weights": TensorData(data=[1.0, 0.0]),
        },
    )

    assert _datum_token_count(datum) == 2
    assert _datum_loss_token_count(datum) == 1


def test_training_datum_token_count_includes_image_tokens():
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buffer, format="PNG")
    datum = Datum(
        model_input=ModelInput(
            chunks=[
                ModelInputChunk(tokens=[10, 20]),
                ImageChunk(data=buffer.getvalue(), format="png", expected_tokens=7),
            ]
        ),
        loss_fn_inputs={},
    )

    assert _training_datum_token_count(datum) == 9
