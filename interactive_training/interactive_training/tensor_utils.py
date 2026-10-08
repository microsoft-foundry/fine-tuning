"""Helpers for round-tripping between the Azure SDK ``TensorData`` and
``torch.Tensor``.

The Azure ``TensorData`` schema is 1-D ``list[float]``; these helpers exist
so custom-loss authors using ``forward_backward_custom_async`` don't repeat
the ``torch.tensor(td.data)`` / ``td.detach().cpu().tolist()`` boilerplate.

Example
-------
.. code-block:: python

    from interactive_training.tensor_utils import tensor_data_to_torch, tensor_data_from_torch

    def my_loss(data, logprobs_list):
        weights = tensor_data_to_torch(data[0].loss_fn_inputs["weights"])
        ...
"""

from __future__ import annotations

import torch
from azure.ai.finetuningsessions.models import TensorData


def tensor_data_to_torch(
    td: TensorData, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Convert a ``TensorData`` (1-D ``list[float]``) to a 1-D ``torch.Tensor``.

    The returned tensor is detached and has ``requires_grad=False``.
    """
    return torch.tensor(td.data, dtype=dtype)


def tensor_data_from_torch(t: torch.Tensor) -> TensorData:
    """Serialise a 1-D ``torch.Tensor`` as ``TensorData``.

    Raises
    ------
    ValueError
        If ``t`` has rank > 1.  The Azure SDK's ``TensorData`` schema is 1-D
        only; multi-dimensional tensors are not supported (see
        ``docs/loss_functions.md`` for the constraint).
    """
    if t.dim() > 1:
        raise ValueError(
            f"tensor_data_from_torch: input must be 0-D or 1-D, got shape "
            f"{tuple(t.shape)}. Multi-dimensional TensorData is not supported "
            "by the current SDK schema."
        )
    return TensorData(
        data=t.detach().cpu().to(torch.float32).flatten().tolist()
    )
