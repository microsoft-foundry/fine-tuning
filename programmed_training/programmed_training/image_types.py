"""Compatibility types for image handling."""

from dataclasses import dataclass

from azure.ai.finetuningsessions.models import ImageChunk


@dataclass
class ImageAssetPointerChunk:
    """Represents a pointer to an image asset in a model input sequence."""

    asset_id: str
    expected_tokens: int

    @property
    def length(self) -> int:
        return self.expected_tokens
