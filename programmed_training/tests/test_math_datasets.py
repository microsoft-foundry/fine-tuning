from unittest.mock import Mock

import pytest
from datasets import Dataset

from programmed_training import renderers
from programmed_training.recipes.math_rl import math_env


@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize(
    "examples,expected_batches",
    [(0, 0), (1, 1), (4, 1), (5, 2)],
)
def test_gsm8k_dataset_length_rounds_up_without_loading_remote_data(
    monkeypatch, split, examples, expected_batches
):
    rows = Dataset.from_dict({
        "question": ["What is one plus one?"] * examples,
        "answer": ["#### 2"] * examples,
    })
    loader = Mock(return_value=rows)
    monkeypatch.setattr(math_env, "load_dataset", loader)

    dataset = math_env.Gsm8kDataset(
        batch_size=4,
        group_size=2,
        renderer=Mock(spec=renderers.Renderer),
        split=split,
    )

    assert len(dataset) == expected_batches
    loader.assert_called_once_with("openai/gsm8k", name="main", split=split)
