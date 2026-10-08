"""Tests for interactive_training.tensor_utils."""

import pytest
import torch
from azure.ai.finetuningsessions.models import TensorData

from interactive_training.tensor_utils import tensor_data_from_torch, tensor_data_to_torch


class TestTensorDataToTorch:
    def test_basic(self):
        td = TensorData(data=[1.0, 2.0, 3.0])
        t = tensor_data_to_torch(td)
        assert torch.equal(t, torch.tensor([1.0, 2.0, 3.0]))
        assert t.dtype == torch.float32
        assert t.requires_grad is False

    def test_empty(self):
        t = tensor_data_to_torch(TensorData(data=[]))
        assert t.shape == (0,)

    def test_negative_values(self):
        t = tensor_data_to_torch(TensorData(data=[-1.5, 2.3, -0.0]))
        assert t.tolist() == pytest.approx([-1.5, 2.3, 0.0])

    def test_custom_dtype(self):
        t = tensor_data_to_torch(TensorData(data=[1.0, 2.0]), dtype=torch.float64)
        assert t.dtype == torch.float64


class TestTensorDataFromTorch:
    def test_basic_1d(self):
        td = tensor_data_from_torch(torch.tensor([1.0, 2.0, 3.0]))
        assert td.data == [1.0, 2.0, 3.0]

    def test_scalar(self):
        td = tensor_data_from_torch(torch.tensor(3.14))
        assert td.data == pytest.approx([3.14])

    def test_detaches_grad(self):
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        td = tensor_data_from_torch(t)
        assert td.data == [1.0, 2.0]

    def test_downcasts_to_float32(self):
        # Pass a float64 tensor; output list should still be plain floats.
        td = tensor_data_from_torch(torch.tensor([1.5, 2.5], dtype=torch.float64))
        assert td.data == [1.5, 2.5]

    def test_rejects_2d(self):
        with pytest.raises(ValueError, match="must be 0-D or 1-D"):
            tensor_data_from_torch(torch.zeros(2, 3))

    def test_rejects_3d(self):
        with pytest.raises(ValueError, match="Multi-dimensional"):
            tensor_data_from_torch(torch.zeros(2, 3, 4))


class TestRoundTrip:
    def test_to_from_torch(self):
        original = TensorData(data=[0.5, -1.5, 2.0])
        roundtrip = tensor_data_from_torch(tensor_data_to_torch(original))
        assert roundtrip.data == original.data
