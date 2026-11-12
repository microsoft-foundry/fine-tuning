"""Action-token KL aggregation at full-context position boundaries."""

import pytest
import torch
from azure.ai.finetuningsessions.models import Datum, ModelInput, ModelInputChunk, TensorData

from programmed_training.rl.metrics import compute_kl_sample_train


def datum(length, selected):
    logprobs = [0.0] * length
    mask = [0.0] * length
    for position, difference in selected.items():
        logprobs[position - 1] = difference
        mask[position - 1] = 1.0
    return Datum(
        model_input=ModelInput(chunks=[ModelInputChunk(tokens=[1] * length)]),
        loss_fn_inputs={
            "logprobs": TensorData(data=logprobs),
            "mask": TensorData(data=mask),
        },
    )


def test_kl_depth_boundaries_exclude_prompt_and_match_global_weighted_means():
    selected = {
        8191: 0.1, 8192: 0.2, 16383: 0.3, 16384: 0.4,
        32767: 0.5, 32768: 0.6, 65535: 0.7, 65536: 0.8,
    }
    result = compute_kl_sample_train([datum(65536, selected)], [torch.zeros(65536)])
    bins = {
        "0_8k": [0.1], "8_16k": [0.2, 0.3], "16_32k": [0.4, 0.5],
        "32_64k": [0.6, 0.7], "over_64k": [0.8],
    }
    for name, values in bins.items():
        prefix = f"optim/kl_sample_train_by_depth/{name}"
        assert result[f"{prefix}/tokens"] == len(values)
        assert result[f"{prefix}/v1"] == pytest.approx(sum(values) / len(values))
        assert result[f"{prefix}/v2"] == pytest.approx(
            sum(value * value for value in values) / (2 * len(values))
        )
    assert result["optim/kl_sample_train_v1"] == pytest.approx(0.45)
    assert result["optim/kl_sample_train_action_tokens"] == 8
    assert result["optim/kl_sample_train_max_action_position"] == 65536
    assert result["optim/kl_sample_train_v2"] == pytest.approx(
        sum(value * value for value in selected.values()) / 16
    )


def test_kl_depth_pools_tokens_instead_of_averaging_sequence_means():
    result = compute_kl_sample_train(
        [datum(4, {1: 0.2}), datum(4, {1: 0.4, 2: 0.6, 3: 0.8})],
        [torch.zeros(4), torch.zeros(4)],
    )
    prefix = "optim/kl_sample_train_by_depth/0_8k"
    assert result[f"{prefix}/tokens"] == 4
    assert result[f"{prefix}/v1"] == pytest.approx(0.5)
    assert result[f"{prefix}/v2"] == pytest.approx(0.15)
    assert result[f"{prefix}/v1"] == result["optim/kl_sample_train_v1"]
    empty = "optim/kl_sample_train_by_depth/over_64k"
    assert result[f"{empty}/tokens"] == 0
    assert f"{empty}/v1" not in result
    assert f"{empty}/v2" not in result


def test_kl_depth_preserves_signed_estimator_and_empty_batch_behavior():
    assert compute_kl_sample_train([], []) == {}
    result = compute_kl_sample_train([datum(1, {1: -0.2})], [torch.zeros(1)])
    assert result["optim/kl_sample_train_v1"] == pytest.approx(-0.2)
    assert result["optim/kl_sample_train_v2"] == pytest.approx(0.02)


def test_kl_depth_records_actual_256k_boundary_position():
    length = 262143
    result = compute_kl_sample_train(
        [datum(length, {length: 0.01})], [torch.zeros(length)],
    )
    assert result["optim/kl_sample_train_max_action_position"] == length
    assert result["optim/kl_sample_train_by_depth/over_64k/tokens"] == 1
