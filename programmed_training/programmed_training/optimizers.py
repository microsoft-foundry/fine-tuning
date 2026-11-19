"""Shared, recipe-overridable Adam settings for Programmed Training."""

import chz
from azure.ai.finetuningsessions.models import AdamParams


@chz.chz
class AdamConfig:
    """Defaults preserve the effective settings of canonical Loom trainers."""

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.0

    def adam_kwargs(self) -> dict[str, float]:
        return {
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_eps": self.adam_eps,
            "adam_weight_decay": self.adam_weight_decay,
        }

    def to_adam_params(self, learning_rate: float) -> AdamParams:
        return AdamParams(
            learning_rate=learning_rate,
            beta1=self.adam_beta1,
            beta2=self.adam_beta2,
            eps=self.adam_eps,
            weight_decay=self.adam_weight_decay,
        )
