import logging

import torch
from azure.ai.finetuningsessions.models import Datum, ModelInput, ModelInputChunk, TensorData

from interactive_training.image_types import ImageAssetPointerChunk, ImageChunk


def _chunk_len(chunk: ModelInputChunk | ImageChunk | ImageAssetPointerChunk) -> int:
    """Get the token length of a chunk (text or image)."""
    if isinstance(chunk, ModelInputChunk):
        return len(chunk.tokens)
    return chunk.length

logger = logging.getLogger(__name__)


def compute_mean_nll(
    logprobs_list: list[TensorData], weights_list: list[TensorData]
) -> float:
    """Compute weighted mean negative log likelihood."""
    total_weighted_logprobs = 0.0
    total_weights = 0.0

    for logprobs, weights in zip(logprobs_list, weights_list, strict=True):
        # Explicitly cast to float32.  Logprobs are floating-point by nature,
        # but JSON round-tripping (especially through Cosmos DB) can demote
        # exact-integer values like 0 or -0.0 to Python ints, which makes
        # torch.tensor() produce a Long tensor.  The cast is lossless:
        # float32 has 23 mantissa bits (~7 decimal digits), more than enough
        # for log-probabilities, and any genuine float values pass through
        # unchanged.  Mirrors the renderer's float32 fix for weights
        # (commit dcc02baa).
        logprobs_torch = torch.tensor(logprobs.data, dtype=torch.float)
        weights_torch = torch.tensor(weights.data, dtype=torch.float)
        total_weighted_logprobs += logprobs_torch.dot(weights_torch)
        total_weights += weights_torch.sum()

    if total_weights == 0:
        logger.warning("No valid weights found for NLL computation")
        return float("nan")

    return float(-total_weighted_logprobs / total_weights)


def create_rightshifted_model_input_and_leftshifted_targets(
    chunks: list[ModelInputChunk],
) -> tuple[ModelInput, list[int]]:
    """
    Given a full sequence of model input chunks, create
     "inputs" (with last token removed); these are also list[ModelInputChunk] because text+images
     "targets" (with first token removed); these are list[int] text tokens
    """
    assert len(chunks) >= 1, "must have at least one chunk"

    last_chunk = chunks[-1]
    if not isinstance(last_chunk, ModelInputChunk):
        raise ValueError(
            "The last chunk must be a text chunk. This is because images are 0-loss anyways, so we should remove them beforehand."
        )

    total_length = sum(_chunk_len(c) for c in chunks)
    if total_length < 2:
        raise ValueError("need at least 2 tokens for input/target split")

    # Build input chunks: all but last, then append truncated last chunk
    input_chunks: list[ModelInputChunk] = list(chunks[:-1])
    if _chunk_len(last_chunk) > 1:
        input_chunks.append(ModelInputChunk(tokens=last_chunk.tokens[:-1]))

    # Build target tokens: collect all tokens, then slice off first
    all_tokens: list[int] = []
    for chunk in chunks:
        if isinstance(chunk, ModelInputChunk):
            all_tokens.extend(chunk.tokens)
        else:
            all_tokens.extend([0] * _chunk_len(chunk))
    target_tokens = all_tokens[1:]

    return ModelInput(chunks=input_chunks), target_tokens


def datum_from_model_input_weights(
    model_input: ModelInput,
    weights: torch.Tensor,
    max_length: int | None = None,
) -> Datum:
    """
    Create a Datum from a ModelInput and weights tensor.

    Performs max_length truncation and next-token slicing to create input and target.
    Text chunks can be truncated, but image chunks must be wholly discarded to stay
    within max_length.

    Args:
        model_input: The model input containing a sequence of text and/or image chunks
        weights: The weights tensor aligned with the model_input length
        max_length: Optional maximum sequence length. If provided, truncates to this length.
                   Image chunks are discarded entirely if they would exceed max_length.

    Returns:
        A Datum with model_input (input tokens) and loss_fn_inputs (target tokens and weights)
    """

    model_input_chunks = list(model_input.chunks)

    # Truncate to max_length by popping from end
    if max_length is not None:
        total_length = sum(_chunk_len(chunk) for chunk in model_input_chunks)

        while total_length > max_length and model_input_chunks:
            last = model_input_chunks[-1]
            if isinstance(last, ModelInputChunk):
                overflow = total_length - max_length
                if overflow < _chunk_len(last):
                    # Partial truncation of text chunk
                    model_input_chunks[-1] = ModelInputChunk(
                        tokens=list(last.tokens[:-overflow])
                    )
                    total_length = max_length
                else:
                    # Remove entire text chunk
                    model_input_chunks.pop()
                    total_length -= _chunk_len(last)
            else:
                # Image chunk - must remove entirely
                model_input_chunks.pop()
                total_length -= _chunk_len(last)

    # Remove trailing images (no text to predict after them)
    while model_input_chunks and isinstance(
        model_input_chunks[-1], (ImageChunk, ImageAssetPointerChunk)
    ):
        model_input_chunks.pop()

    input_model_input, target_tokens = create_rightshifted_model_input_and_leftshifted_targets(
        model_input_chunks
    )
    weights = weights[1 : len(target_tokens) + 1]

    return Datum(
        model_input=input_model_input,
        loss_fn_inputs={
            "weights": TensorData(
                data=weights.tolist(),
            ),
            "target_tokens": TensorData(
                data=target_tokens,
            ),
        },
    )
